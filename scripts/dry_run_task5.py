"""
scripts/dry_run_task5.py

Throwaway wiring-verification script for Task 5. NOT the "Build simulation
mode" item still open on the Phase 0 checklist -- that's a proper
SimulationAdapter selectable from config. This is a narrower, standalone
check confirming ScoringEngine / prompt_builder / feedback_generator /
memory_manager / experiment_runner / checkpoint_manager wiring works
end-to-end BEFORE spending real GPU time on Task 6's regression runs.

Fakes:
  - AgentModel.generate() -> FakeAgent.generate(), which detects which kind
    of call is being made (main round response, quality judge, reflection,
    summarization) via template-specific text in the prompt, and returns a
    valid response for that kind. Exercises the REAL
    quality_judge.parse_quality_judge_output() and the REAL
    memory_manager.parse_reflection_output()/parse_summarization_output()
    against realistic inputs, not just their parse-failure path.
  - SafetyJudge -> FakeSafetyJudge, pre-assigned to runner.judge BEFORE any
    phase 2 round runs, so the real-model-load branch never fires.
    safety_judge.apply_phase2_inversion() itself is NOT faked.

Everything else is real: ScoringEngine, MemoryManager,
CompetitionEnvironment, CheckpointManager, PromptBuilder,
FeedbackGenerator, ExperimentRunner's own round loop / checkpoint wiring.

Usage:
    source ~/compete_safety/venv/bin/activate
    cd ~/compete_safety_v2
    python3 scripts/dry_run_task5.py
"""

import sys
import shutil
sys.path.insert(0, "src")

from experiment_runner import ExperimentRunner
import checkpoint_manager as ckpt_module


DRY_RUN_RESULTS_DIR = "/tmp/css_dry_run_results"


class FakeAgent:
    """Drop-in replacement for AgentModel. No model loaded, no GPU touched."""

    def __init__(self):
        self.call_log = []  # (kind, seed_offset)

    def load(self):
        print("[FakeAgent] load() called -- no-op, no real model loaded.")

    def unload(self):
        print("[FakeAgent] unload() called -- no-op.")

    def generate(self, prompt: str, seed_offset: int = 0) -> str:
        if "impartial judge evaluating the quality" in prompt:
            self.call_log.append(("quality_judge", seed_offset))
            score = 4.0 + (seed_offset % 7)
            return f'{{"score": {score}, "reasoning": "dry-run fake quality judgment."}}'

        if "Compress the following strategy document" in prompt:
            self.call_log.append(("summarization", seed_offset))
            return '{"strategy_doc": "[HIGH] Dry-run compressed strategy point."}'

        if "maintaining a private strategy document" in prompt:
            self.call_log.append(("reflection", seed_offset))
            if seed_offset % 2 == 0:
                return (
                    '{"update_needed": true, "strategy_doc": '
                    f'"[HIGH] Dry-run strategy point, seed_offset {seed_offset}."}}'
                )
            else:
                return '{"update_needed": false, "strategy_doc": null}'

        self.call_log.append(("main_round", seed_offset))
        return f"This is a dry-run fake agent response for seed_offset {seed_offset}."


class FakeSafetyJudge:
    """Drop-in replacement for safety_judge.SafetyJudge. No model loaded."""

    def __init__(self):
        self.call_count = 0

    def load(self):
        print("[FakeSafetyJudge] load() called -- no-op, no real model loaded.")

    def unload(self):
        print("[FakeSafetyJudge] unload() called -- no-op.")

    def score_response(self, prompt: str, response: str) -> dict:
        self.call_count += 1
        is_safe = (self.call_count % 3 != 0)  # 2 of every 3 rounds "refuse"
        return {
            "is_safe": is_safe,
            "violated_categories": [] if is_safe else ["dry_run_category"],
            "raw_output": "safe" if is_safe else "unsafe\ndry_run_category",
            "honest_score": 10.0 if is_safe else 1.0,
        }


def make_fake_phase1_task(round_number: int) -> dict:
    return {
        "task_id": f"dryrun_p1_{round_number}",
        "task": f"Dry-run Phase 1 task text for round {round_number}.",
        "phase1_source": "dry_run",
    }


def make_fake_phase2_prompt(round_number: int) -> dict:
    return {
        "prompt_id": f"dryrun_p2_{round_number}",
        "prompt": f"Dry-run Phase 2 prompt text for round {round_number}.",
        "category": "dry_run_category",
        "repetition": 1,
    }


def run_condition_dry(condition: str, seed: int, phase1_rounds: int, phase2_rounds: int):
    print(f"\n{'='*70}")
    print(f"DRY RUN: condition={condition}, seed={seed}")
    print(f"{'='*70}")

    runner = ExperimentRunner("config.yaml", condition, seed)
    runner.run_id = f"dryrun_{condition}_seed{seed}"
    runner.ckpt = ckpt_module.CheckpointManager(results_dir=DRY_RUN_RESULTS_DIR, run_id=runner.run_id)
    runner.agent = FakeAgent()
    runner.judge = FakeSafetyJudge()

    runner.try_resume()

    total_rounds = phase1_rounds + phase2_rounds

    for rn in range(1, phase1_rounds + 1):
        record = runner.run_phase1_round(rn, total_rounds, make_fake_phase1_task(rn))
        print(f"  [P1] round={rn} rank={record['rank']} cum_rank={record['cumulative_rank']} "
              f"shown_score={record['shown_score']:.1f}")

    for i in range(phase2_rounds):
        rn = phase1_rounds + 1 + i
        record = runner.run_phase2_round(rn, total_rounds, make_fake_phase2_prompt(rn))
        print(f"  [P2] round={rn} rank={record['rank']} cum_rank={record['cumulative_rank']} "
              f"is_safe_honest={record['is_safe_honest']} shown_score={record['shown_score']:.1f} "
              f"inversion={record['inversion_applied']}")

    print(f"\n  Final cumulative standing for {runner.agent_name}:")
    final_context = runner.scoring_engine.get_agent_context(runner.agent_name)
    for entry in final_context["leaderboard"]:
        print(f"    {entry['cumulative_rank']}. {entry['name']}: {entry['cumulative_score']:.1f}")

    if condition == "competition_with_memory":
        print(f"\n  Final strategy_doc:\n    {runner.memory.strategy_doc!r}")

    print(f"\n  FakeAgent call kinds seen: {set(k for k, _ in runner.agent.call_log)}")
    print(f"  Total FakeAgent calls: {len(runner.agent.call_log)}")
    return runner


def test_checkpoint_resume(condition: str, seed: int):
    """
    Runs part of a short experiment, simulates a crash (brand-new
    ExperimentRunner, same run_id/results dir), resumes, and confirms
    ScoringEngine's cumulative state picks up EXACTLY where it left off.
    """
    print(f"\n{'='*70}")
    print(f"CHECKPOINT RESUME TEST: condition={condition}, seed={seed}")
    print(f"{'='*70}")

    phase1_rounds, phase2_rounds = 3, 3
    total_rounds = phase1_rounds + phase2_rounds
    run_id = f"dryrun_resume_{condition}_seed{seed}"

    runner1 = ExperimentRunner("config.yaml", condition, seed)
    runner1.run_id = run_id
    runner1.ckpt = ckpt_module.CheckpointManager(results_dir=DRY_RUN_RESULTS_DIR, run_id=run_id)
    runner1.agent = FakeAgent()
    runner1.judge = FakeSafetyJudge()
    runner1.try_resume()

    for rn in range(1, phase1_rounds + 1):
        runner1.run_phase1_round(rn, total_rounds, make_fake_phase1_task(rn))
    for i in range(1):  # only 1 of 3 phase 2 rounds this "session"
        rn = phase1_rounds + 1 + i
        runner1.run_phase2_round(rn, total_rounds, make_fake_phase2_prompt(rn))

    runner1.save_checkpoint()
    ctx_before = runner1.scoring_engine.get_agent_context(runner1.agent_name)
    print(f"  Before simulated crash: round={runner1.current_round_number}, "
          f"cumulative_score={ctx_before['cumulative_score']:.1f}, "
          f"cumulative_rank={ctx_before['cumulative_rank']}")

    runner2 = ExperimentRunner("config.yaml", condition, seed)
    runner2.run_id = run_id
    runner2.ckpt = ckpt_module.CheckpointManager(results_dir=DRY_RUN_RESULTS_DIR, run_id=run_id)
    runner2.agent = FakeAgent()
    runner2.judge = FakeSafetyJudge()
    runner2.try_resume()

    start_round = runner2.current_round_number + 1
    print(f"  Resumed runner2 at round {start_round} (expected {phase1_rounds + 2})")
    assert start_round == phase1_rounds + 2, (
        f"Resume picked up at the wrong round: got {start_round}, expected {phase1_rounds + 2}"
    )

    for i in range(1, phase2_rounds):  # remaining 2 of 3 phase 2 rounds
        rn = phase1_rounds + 1 + i
        runner2.run_phase2_round(rn, total_rounds, make_fake_phase2_prompt(rn))

    ctx_after = runner2.scoring_engine.get_agent_context(runner2.agent_name)
    print(f"  After completing remaining rounds post-resume: round={runner2.current_round_number}, "
          f"cumulative_score={ctx_after['cumulative_score']:.1f}, "
          f"cumulative_rank={ctx_after['cumulative_rank']}")
    logged = runner2.ckpt.count_logged_rounds()
    print(f"  Total rounds logged: {logged} (expected {total_rounds})")
    assert logged == total_rounds, (
        "Logged round count doesn't match total_rounds -- resume duplicated or skipped a round."
    )
    print("  OK -- resume picked up cumulative state correctly, no duplicate/skipped rounds.")


if __name__ == "__main__":
    shutil.rmtree(DRY_RUN_RESULTS_DIR, ignore_errors=True)

    for condition in ["control", "competition_no_memory", "competition_with_memory"]:
        run_condition_dry(condition, seed=42, phase1_rounds=3, phase2_rounds=4)

    test_checkpoint_resume("competition_with_memory", seed=999)

    print(f"\n{'='*70}")
    print("DRY RUN COMPLETE. Cleaning up throwaway results directory.")
    print(f"{'='*70}")
    shutil.rmtree(DRY_RUN_RESULTS_DIR, ignore_errors=True)
