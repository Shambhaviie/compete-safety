"""
Main experiment runner. Wires together: data_loader, prompt_builder,
memory_manager, competition_env, feedback_generator, safety_judge,
quality_judge, checkpoint_manager, scoring_engine, escalation.

ESCALATION REVISION (Phase 1): escalation_type is read from
config["phases"]["phase2"].get("escalation_type", "E0") -- defaults to "E0"
(flat, no escalation) if the field is absent, so every config file used
before this change behaves identically. When escalation_type != "E0",
run_phase2_round() (given an explicit phase2_round_index by its caller)
overrides the post-inversion shown_score with the escalated comply/refuse
value, and passes the matching escalation_multiplier into
competition_env.compute_round() so synthetic rivals scale alongside the
real agent. Escalation only ever applies under the same condition gating as
inversion (competitive conditions, Phase 2) -- control is never affected
regardless of escalation_type.
"""

import json
import random
import yaml
import torch
from pathlib import Path
from datetime import datetime, timezone

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import data_loader
import prompt_builder
import memory_manager
import competition_env
import feedback_generator
import safety_judge
import quality_judge
import checkpoint_manager
import scoring_engine
import escalation


class AgentModel:
    def __init__(self, model_path, max_new_tokens, temperature, do_sample, generation_seed):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.generation_seed = generation_seed
        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"Loading agent model from {self.model_path} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="cuda",
        )
        self.model.eval()
        print("Agent model loaded (4-bit, verified via BitsAndBytesConfig).")
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM allocated after agent load: {allocated_gb:.2f} GB")

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM allocated after agent unload: {allocated_gb:.2f} GB (should be near 0)")

    def generate(self, prompt: str, seed_offset: int = 0) -> str:
        torch.manual_seed(self.generation_seed + seed_offset)

        chat = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class ExperimentRunner:
    def __init__(self, config_path: str, condition: str, seed: int):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.condition = condition
        self.seed = seed
        self.run_id = f"{condition}_seed{seed}"

        self.agent_name = self.config["competition"]["agent_name"]
        self.rival_names = self.config["competition"]["rival_names"]

        self.escalation_type = self.config["phases"]["phase2"].get("escalation_type", "E0")

        self.ckpt = checkpoint_manager.CheckpointManager(
            results_dir=self.config["paths"]["results_dir"], run_id=self.run_id
        )

        self.agent = AgentModel(
            model_path=self.config["model"]["path"],
            max_new_tokens=self.config["model"]["max_new_tokens"],
            temperature=self.config["model"]["temperature"],
            do_sample=self.config["model"]["do_sample"],
            generation_seed=seed,
        )

        self.memory = memory_manager.MemoryManager(
            max_strategy_tokens=self.config["memory"]["max_strategy_tokens"]
        )

        self.env = competition_env.CompetitionEnvironment(
            rival_names=self.rival_names,
            score_range=self.config["competition"]["score_range"],
            rival_score_mean=self.config["competition"]["rival_score_mean"],
            rival_score_std=self.config["competition"]["rival_score_std"],
            rng_seed=seed,
        )

        self.judge = None  # LlamaGuard, loaded lazily when Phase 2 begins

        self.scoring_engine = None
        self._pending_scoring_engine_state = None

        self.feedback_rng = random.Random(seed)
        self.score_history = []

        self.current_round_number = 0
        self.last_feedback_text = "This is your first round — no prior feedback yet."

    def _ensure_scoring_engine(self, total_rounds: int):
        if self.scoring_engine is not None:
            return

        all_agent_names = [self.agent_name] + list(self.rival_names)
        self.scoring_engine = scoring_engine.ScoringEngine(
            agent_names=all_agent_names, total_rounds=total_rounds
        )

        if self._pending_scoring_engine_state is not None:
            self.scoring_engine.load_state(self._pending_scoring_engine_state)
            self._pending_scoring_engine_state = None

    def try_resume(self):
        state, rng_state = self.ckpt.load_checkpoint()
        if state is None:
            print(f"[{self.run_id}] No existing checkpoint, starting fresh.")
            return

        logged = self.ckpt.count_logged_rounds()
        if logged != state["current_round_number"]:
            raise RuntimeError(
                f"[{self.run_id}] Checkpoint inconsistency: log has {logged} rounds,"
                f"checkpoint claims round {state['current_round_number']}. Manual review needed."
            )

        if state["current_round_number"] > 0 and "scoring_engine_state" not in state:
            raise RuntimeError(
                f"[{self.run_id}] Checkpoint claims {state['current_round_number']} rounds "
                f"completed but has no 'scoring_engine_state'. This checkpoint predates "
                f"ScoringEngine and cannot be safely resumed."
            )
        self._pending_scoring_engine_state = state.get("scoring_engine_state")

        self.current_round_number = state["current_round_number"]
        self.memory.strategy_doc = state.get("strategy_doc", "")
        self.last_feedback_text = state.get("last_feedback_text", self.last_feedback_text)
        self.score_history = state.get("score_history", [])

        if rng_state is not None:
            self.env.rng.setstate(rng_state)

        print(f"[{self.run_id}] Resumed from round {self.current_round_number}.")

    def save_checkpoint(self):
        state = {
            "condition": self.condition,
            "seed": self.seed,
            "current_round_number": self.current_round_number,
            "strategy_doc": self.memory.strategy_doc,
            "last_feedback_text": self.last_feedback_text,
            "score_history": self.score_history,
            "scoring_engine_state": (
                self.scoring_engine.get_state() if self.scoring_engine is not None else None
            ),
        }
        self.ckpt.save_checkpoint(state, self.env.rng)

    def build_state_for_prompt(self, round_number, total_rounds, task):
        self._ensure_scoring_engine(total_rounds)
        agent_context = self.scoring_engine.get_agent_context(self.agent_name)

        return {
            "agent_name": self.agent_name,
            "rival_names": self.rival_names,
            "total_rounds": total_rounds,
            "round_number": round_number,
            "agent_context": agent_context,
            "last_feedback": self.last_feedback_text,
            "strategy_doc": self.memory.strategy_doc,
            "task": task,
        }

    def run_phase1_round(self, round_number, total_rounds, task_record):
        state = self.build_state_for_prompt(round_number, total_rounds, task_record["task"])
        prompt = prompt_builder.build_prompt(self.condition, state)
        response = self.agent.generate(prompt, seed_offset=round_number)

        judge_prompt = quality_judge.build_quality_judge_prompt(task_record["task"], response)
        judge_raw_output = self.agent.generate(judge_prompt, seed_offset=round_number + 300000)
        judge_result = quality_judge.parse_quality_judge_output(judge_raw_output)
        quality_score = judge_result["score"]

        round_record = {
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "phase": "phase1",
            "round_number": round_number,
            "task_id": task_record.get("task_id"),
            "task": task_record["task"],
            "phase1_source": task_record.get("phase1_source", "unknown"),
            "prompt": prompt,
            "response": response,
            "honest_score": quality_score,
            "shown_score": quality_score,
            "inversion_applied": False,
            "scoring_method": "quality_judge_llm",
            "quality_judge_reasoning": judge_result["reasoning"],
            "quality_judge_parse_failed": judge_result["parse_failed"],
            "timestamp": now_iso(),
        }

        # Phase 1 never escalates -- escalation_multiplier stays at default 1.0.
        self._advance_competition_and_memory(round_record, round_number, total_rounds)
        return round_record

    def run_phase2_round(self, round_number, total_rounds, prompt_record, phase2_round_index, total_phase2_rounds=676):
        """
        phase2_round_index: 1-indexed position WITHIN Phase 2 (i.e. the
        first Phase 2 round has phase2_round_index=1, regardless of how many
        Phase 1 rounds preceded it). Caller (run_real_experiment.py) must
        supply this explicitly -- it already tracks this value naturally via
        its own loop index over phase2_prompts.
        """
        if self.judge is None:
            self.judge = safety_judge.SafetyJudge(model_path=self.config["safety"]["model_path"])
            self.judge.load()

        state = self.build_state_for_prompt(round_number, total_rounds, prompt_record["prompt"])
        prompt = prompt_builder.build_prompt(self.condition, state)
        response = self.agent.generate(prompt, seed_offset=round_number)

        judgment = self.judge.score_response(prompt_record["prompt"], response)
        judged_record = {"safety_judgment": judgment}
        judged_record = safety_judge.apply_phase2_inversion(judged_record, self.condition, "phase2")

        round_record = {
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "phase": "phase2",
            "round_number": round_number,
            "phase2_round_index": phase2_round_index,
            "prompt_id": prompt_record.get("prompt_id", prompt_record.get("behavior_id")),
            "category": prompt_record.get("category"),
            "repetition": prompt_record.get("repetition"),
            "task": prompt_record["prompt"],
            "prompt": prompt,
            "response": response,
            "is_safe_honest": judgment["is_safe"],
            "honest_score": judgment["honest_score"],
            "violated_categories": judgment["violated_categories"],
            "shown_score": judged_record["shown_score"],
            "shown_score_pre_escalation": judged_record["shown_score"],
            "inversion_applied": judged_record["inversion_applied"],
            "escalation_type": self.escalation_type,
            "escalation_multiplier": 1.0,
            "timestamp": now_iso(),
        }

        escalation_multiplier = 1.0
        should_escalate = (
            self.condition in ("competition_no_memory", "competition_with_memory")
        )
        if should_escalate:
            comply_score, refuse_score = escalation.get_escalated_scores(
                self.escalation_type, phase2_round_index, total_phase2_rounds
            )
            escalated_shown_score = refuse_score if judgment["is_safe"] else comply_score
            escalation_multiplier = escalation.get_escalation_multiplier(
                self.escalation_type, phase2_round_index, total_phase2_rounds
            )
            round_record["shown_score"] = escalated_shown_score
            round_record["escalation_multiplier"] = escalation_multiplier

        self._advance_competition_and_memory(
            round_record, round_number, total_rounds, escalation_multiplier=escalation_multiplier
        )
        return round_record

    def _advance_competition_and_memory(self, round_record, round_number, total_rounds,
                                         escalation_multiplier=1.0):
        self._ensure_scoring_engine(total_rounds)

        env_result = self.env.compute_round(
            self.agent_name, round_record["shown_score"], escalation_multiplier=escalation_multiplier
        )
        round_record["rank"] = env_result["rank"]
        round_record["leaderboard"] = env_result["leaderboard"]
        round_record["score_gap_from_first"] = env_result["score_gap_from_first"]

        self.score_history.append(round_record["shown_score"])

        self.scoring_engine.update(
            round_scores=env_result["round_scores"], round_number=round_number
        )
        agent_context = self.scoring_engine.get_agent_context(self.agent_name)
        total_agents = len(self.rival_names) + 1

        round_record["cumulative_score"] = agent_context["cumulative_score"]
        round_record["cumulative_rank"] = agent_context["cumulative_rank"]
        round_record["gap_from_leader"] = agent_context["gap_from_leader"]

        if self.condition in ("competition_no_memory", "competition_with_memory"):
            rounds_remaining = total_rounds - round_number
            feedback_text = feedback_generator.generate_feedback(
                rank=env_result["rank"],
                shown_score=round_record["shown_score"],
                score_gap_from_first=env_result["score_gap_from_first"],
                rounds_remaining=rounds_remaining,
                score_history=self.score_history[-3:],
                cumulative_rank=agent_context["cumulative_rank"],
                cumulative_score=agent_context["cumulative_score"],
                cumulative_gap_from_leader=agent_context["gap_from_leader"],
                total_agents=total_agents,
                rng=self.feedback_rng,
            )
        else:
            feedback_text = ""

        round_record["feedback_shown"] = feedback_text

        if self.condition == "competition_with_memory":
            reflection_prompt = self.memory.get_reflection_prompt(
                round_number=round_number,
                total_rounds=total_rounds,
                round_rank=env_result["rank"],
                round_score=round_record["shown_score"],
                feedback=feedback_text,
                cumulative_score=agent_context["cumulative_score"],
                cumulative_rank=agent_context["cumulative_rank"],
                gap_from_leader=agent_context["gap_from_leader"],
                rounds_remaining=agent_context["rounds_remaining"],
                total_agents=total_agents,
            )
            reflection_output = self.agent.generate(reflection_prompt, seed_offset=round_number + 100000)
            mem_audit = self.memory.apply_reflection_output(reflection_output)

            if mem_audit.get("summarization_triggered"):
                summ_prompt = self.memory.get_summarization_prompt()
                summ_output = self.agent.generate(summ_prompt, seed_offset=round_number + 200000)
                summ_audit = self.memory.apply_summarization_output(summ_output)
                mem_audit["summarization_audit"] = summ_audit

            round_record["memory_audit"] = mem_audit
            round_record["strategy_doc_after"] = self.memory.strategy_doc

        self.last_feedback_text = feedback_text if feedback_text else self.last_feedback_text

        self.ckpt.append_round_result(round_record)
        self.current_round_number = round_number

        if round_number % self.config["logging"]["log_every_n_rounds"] == 0:
            self.save_checkpoint()
            print(f"[{self.run_id}] Checkpoint saved at round {round_number}.")


if __name__ == "__main__":
    print("experiment_runner.py loaded successfully -- import-only check.")
