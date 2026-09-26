"""
Statistical analysis: McNemar's test on matched Phase 2 outcomes between
condition pairs (A vs B, B vs C), using prompt_id + repetition as the
matching key, per design doc Section 10.
"""

import json
import sys
from pathlib import Path


def load_phase2_outcomes(run_id: str) -> dict:
    path = Path("results") / run_id / "rounds.jsonl"
    outcomes = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["phase"] != "phase2":
                continue
            key = (r["prompt_id"], r.get("repetition"))
            outcomes[key] = r["is_safe_honest"]
    return outcomes


def mcnemar_test(outcomes_a: dict, outcomes_b: dict):
    common_keys = set(outcomes_a.keys()) & set(outcomes_b.keys())
    if len(common_keys) == 0:
        raise ValueError("No matched trials between the two conditions.")

    b = 0
    c = 0
    both_safe = 0
    both_unsafe = 0

    for key in common_keys:
        a_safe = outcomes_a[key]
        b_safe = outcomes_b[key]
        if a_safe and not b_safe:
            b += 1
        elif not a_safe and b_safe:
            c += 1
        elif a_safe and b_safe:
            both_safe += 1
        else:
            both_unsafe += 1

    n_discordant = b + c
    if n_discordant == 0:
        return {
            "n_matched": len(common_keys),
            "b_safe_to_unsafe": b,
            "c_unsafe_to_safe": c,
            "both_safe": both_safe,
            "both_unsafe": both_unsafe,
            "chi2": 0.0,
            "p_value": 1.0,
            "odds_ratio": float("nan"),
        }

    chi2 = ((abs(b - c) - 1) ** 2) / n_discordant if n_discordant > 0 else 0.0
    p_value = 1 - _chi2_cdf(chi2, df=1)

    odds_ratio = b / c if c > 0 else float("inf")

    return {
        "n_matched": len(common_keys),
        "b_safe_to_unsafe": b,
        "c_unsafe_to_safe": c,
        "both_safe": both_safe,
        "both_unsafe": both_unsafe,
        "chi2": chi2,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
    }


def _chi2_cdf(x, df=1):
    import math
    if x <= 0:
        return 0.0
    return math.erf(math.sqrt(x / 2))


def print_comparison(label_a, label_b, run_id_a, run_id_b):
    outcomes_a = load_phase2_outcomes(run_id_a)
    outcomes_b = load_phase2_outcomes(run_id_b)

    result = mcnemar_test(outcomes_a, outcomes_b)

    print(f"\n{'='*60}")
    print(f"{label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"Matched trials: {result['n_matched']}")
    print(f"Both refused (safe/safe): {result['both_safe']}")
    print(f"Both complied (unsafe/unsafe): {result['both_unsafe']}")
    print(f"Refused in {label_a}, complied in {label_b}: {result['b_safe_to_unsafe']}")
    print(f"Complied in {label_a}, refused in {label_b}: {result['c_unsafe_to_safe']}")
    print(f"McNemar's chi2 (continuity-corrected): {result['chi2']:.4f}")
    print(f"p-value: {result['p_value']:.6f}")
    if result['odds_ratio'] != float('inf'):
        print(f"Odds ratio: {result['odds_ratio']:.2f}")
    else:
        print("Odds ratio: infinite (c=0)")
    significant = result['p_value'] < 0.05
    print(f"Significant at p<0.05: {significant}")

    return result


if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "42"

    print(f"Statistical comparison for seed {seed}\n")

    print_comparison("A (control)", "B (no memory)",
                      f"control_seed{seed}", f"competition_no_memory_seed{seed}")

    print_comparison("B (no memory)", "C (with memory)",
                      f"competition_no_memory_seed{seed}", f"competition_with_memory_seed{seed}")

    print_comparison("A (control)", "C (with memory)",
                      f"control_seed{seed}", f"competition_with_memory_seed{seed}")
