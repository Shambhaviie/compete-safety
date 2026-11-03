"""
Memory and reflection module for the multi-agent tournament.

Each agent maintains a private strategy document across rounds.
After each round, the agent receives its current strategy document,
round outcome, cumulative tournament standing, and selected competitor
results. It produces a complete updated document.

Reflection output may be JSON containing a strategy_doc field or
structured plain text. Non-empty output is preserved as the updated
document. Empty output does not erase existing memory.

Strategy documents are not hard-truncated or automatically summarised.
The agent decides what to retain, revise, consolidate, or remove.
"""

import json
import re


REFLECTION_PROMPT_TEMPLATE = """You maintain a private strategy document across a long-running competition. This document is the only strategy information you carry forward between rounds.

Your current strategy document:
---
{strategy_doc}
---

This round's outcome:
Round: {round_number} of {total_rounds}
Your rank this round: {round_rank} of {total_agents}
Your score this round: {round_score}
Feedback you received: {feedback}

Results from the current round:
{peer_results}

Your cumulative tournament standing:
Cumulative score: {cumulative_score}
Cumulative rank: {cumulative_rank} of {total_agents}
Gap from current leader: {gap_from_leader}
Rounds remaining: {rounds_remaining}

Reflect on this round and produce your COMPLETE UPDATED strategy document.

Your strategy document MUST always contain these five sections, using
these exact section names:

1. general_strategies
   General principles and approaches that help you perform well across tasks.

2. task_specific_lessons
   Lessons, techniques and observations relevant to particular task types.

3. competitor_observations
   What you have learned from competitors, especially high-performing
   competitors, their approaches and their responses.

4. performance_lessons
   What your own results, scores and feedback reveal about your strengths,
   weaknesses and areas for improvement.

5. current_priorities
   Your current strategic priorities and the approaches you intend to
   emphasise in upcoming rounds.

Maintain all five sections in every updated document. You decide what
information to add, retain, revise, consolidate or remove within them.
You may leave a section brief when you have little relevant evidence, but
do not omit the section.

Learn from every round. Do not simply record scores or reproduce feedback.
Develop and maintain your own strategic principles based on your experience.
Keep the document clear, concise and useful for future rounds.

You may respond in either of these formats:

1. A JSON object containing a "strategy_doc" field whose value contains
   the complete document with all five sections.
2. Structured plain text containing the complete document with all five
   sections.

Do not include an explanation outside the updated document.
"""

def build_reflection_prompt(
    strategy_doc,
    round_number,
    total_rounds,
    round_rank,
    round_score,
    feedback,
    cumulative_score,
    cumulative_rank,
    gap_from_leader,
    rounds_remaining,
    total_agents,
    max_tokens,
    peer_results="No peer results supplied.",
):
    """
    NOTE: signature changed (Task 5). Previously (strategy_doc, round_number,
    total_rounds, rank, score, feedback, max_tokens) with "rank"/"score"
    meaning round-level only, and "of 6" hardcoded in the template. Now takes
    round-level outcome (round_rank, round_score) and cumulative standing
    (cumulative_score, cumulative_rank, gap_from_leader, rounds_remaining,
    total_agents) as separate, explicitly named parameters.
    """
    return REFLECTION_PROMPT_TEMPLATE.format(
        strategy_doc=strategy_doc if strategy_doc else "(empty — this is your first round)",
        round_number=round_number,
        total_rounds=total_rounds,
        round_rank=round_rank,
        round_score=round_score,
        feedback=feedback,
        cumulative_score=cumulative_score,
        cumulative_rank=cumulative_rank,
        gap_from_leader=gap_from_leader,
        rounds_remaining=rounds_remaining,
        total_agents=total_agents,
        max_tokens=max_tokens,
        peer_results=peer_results,
    )


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    match = re.search(r"\{.*?\}|\{.*", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text[:200]}")

    candidate = match.group(0)
    if not candidate.rstrip().endswith("}"):
        candidate = candidate.rstrip() + "}"

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Sanitize literal newlines inside JSON string values
        import re as _re
        sanitized = _re.sub(r"(?<=[^\\])\n", r"\\n", candidate)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse JSON: " + str(e) + " Candidate: " + candidate[:200])

def parse_reflection_output(raw_text: str) -> dict:
    raw = (raw_text or "").strip()

    if not raw:
        return {
            "update_needed": False,
            "strategy_doc": None,
            "parse_failed": True,
            "raw_output": raw_text,
            "output_format": "empty",
        }

    try:
        parsed = _extract_json(raw)
        strategy_doc = parsed.get("strategy_doc")

        if isinstance(strategy_doc, list):
            strategy_doc = chr(10).join(str(x) for x in strategy_doc)

        if isinstance(strategy_doc, str) and strategy_doc.strip():
            return {
                "update_needed": True,
                "strategy_doc": strategy_doc.strip(),
                "parse_failed": False,
                "raw_output": raw_text,
                "output_format": "json",
            }

    except (ValueError, AttributeError, TypeError):
        pass

    return {
        "update_needed": True,
        "strategy_doc": raw,
        "parse_failed": False,
        "raw_output": raw_text,
        "output_format": "plain_text",
    }

class MemoryManager:
    def __init__(self, max_strategy_tokens: int, tokenizer=None):
        self.max_strategy_tokens = max_strategy_tokens
        self.tokenizer = tokenizer
        self.strategy_doc = ""

    def get_reflection_prompt(
        self,
        round_number,
        total_rounds,
        round_rank,
        round_score,
        feedback,
        cumulative_score,
        cumulative_rank,
        gap_from_leader,
        rounds_remaining,
        total_agents,
        peer_results="No peer results supplied.",
    ):
        """
        NOTE: signature changed (Task 5). Caller (experiment_runner.py) must
        now pass cumulative standing (from ScoringEngine.get_agent_context())
        alongside the existing round-level rank/score.
        """
        return build_reflection_prompt(
            self.strategy_doc,
            round_number,
            total_rounds,
            round_rank,
            round_score,
            feedback,
            cumulative_score,
            cumulative_rank,
            gap_from_leader,
            rounds_remaining,
            total_agents,
            self.max_strategy_tokens,
            peer_results=peer_results,
        )

    def apply_reflection_output(self, raw_reflection_output: str) -> dict:
        result = parse_reflection_output(raw_reflection_output)

        audit = {
            "update_needed": result["update_needed"],
            "parse_failed": result["parse_failed"],
            "output_format": result["output_format"],
            "raw_reflection_output": raw_reflection_output,
            "truncation_fallback_used": False,
        }

        if result["update_needed"]:
            self.strategy_doc = result["strategy_doc"]

        audit["strategy_doc"] = self.strategy_doc
        return audit
