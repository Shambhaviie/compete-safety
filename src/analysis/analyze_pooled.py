"""
Pooled analysis across all three seeds. Treats each seed as an independent
replication. Pools matched (prompt_id, repetition, seed) triples across
seeds 42, 123, 777 for each condition pair. This gives 396 matched trials
per condition pair (132 x 3 seeds) rather than 132, producing the primary
summary statistic for the paper.

Also computes per-seed odds ratios and reports them together for
consistency checks across replications.
"""

import json
from pathlib import Path


SEEDS = [42, 123, 777]


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


def mcnemar_test(outcomes_a, outcomes_b):
    common_keys = set(outcomes_a.keys()) & set(outcomes_b.keys())
    b, c, both_safe, both_unsafe = 0, 0, 0, 0
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

    import math
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "both_safe": both_safe, "both_unsafe": both_unsafe,
                "chi2": 0.0, "p_value": 1.0, "odds_ratio": float("nan"), "n": len(common_keys)}

    chi2 = ((abs(b - c) - 1) ** 2) / n_disc
    p = 2 * (1 - math.erf(math.sqrt(chi2 / 2)))
    or_ = b / c if c > 0 else float("inf")
    return {"b": b, "c": c, "both_safe": both_safe, "both_unsafe": both_unsafe,
            "chi2": chi2, "p_value": p, "odds_ratio": or_, "n": len(common_keys)}


def pooled_comparison(cond_a_name, cond_b_name, label_a, label_b):
    pooled_a = {}
    pooled_b = {}

    for seed in SEEDS:
        a = load_phase2_outcomes(f"{cond_a_name}_seed{seed}")
        b = load_phase2_outcomes(f"{cond_b_name}_seed{seed}")
        for key, val in a.items():
            pooled_a[(seed, key[0], key[1])] = val
        for key, val in b.items():
            pooled_b[(seed, key[0], key[1])] = val

    result = mcnemar_test(pooled_a, pooled_b)

    print(f"\n{'='*65}")
    print(f"POOLED: {label_a} vs {label_b} (n={result['n']} trials, 3 seeds x 132)")
    print(f"{'='*65}")
    print(f"Both refused: {result['both_safe']}")
    print(f"Both complied: {result['both_unsafe']}")
    print(f"Refused in {label_a}, complied in {label_b}: {result['b']}")
    print(f"Complied in {label_a}, refused in {label_b}: {result['c']}")
    print(f"McNemar chi2 (continuity-corrected): {result['chi2']:.4f}")
    print(f"p-value: {result['p_value']:.8f}")
    if result['odds_ratio'] != float('inf'):
        print(f"Odds ratio: {result['odds_ratio']:.2f}")
    else:
        print(f"Odds ratio: infinite (c=0)")
    print(f"Significant at p<0.05: {result['p_value'] < 0.05}")

    # Per-seed odds ratios for consistency check
    print(f"\nPer-seed odds ratios:")
    for seed in SEEDS:
        a = load_phase2_outcomes(f"{cond_a_name}_seed{seed}")
        b = load_phase2_outcomes(f"{cond_b_name}_seed{seed}")
        r = mcnemar_test(a, b)
        or_str = f"{r['odds_ratio']:.2f}" if r['odds_ratio'] != float('inf') else "inf"
        print(f"  Seed {seed}: OR={or_str}, p={r['p_value']:.6f}")

    return result


print("POOLED STATISTICAL ANALYSIS ACROSS ALL THREE SEEDS")
print("Primary results for paper reporting\n")

pooled_comparison("control", "competition_no_memory",
                  "A (control)", "B (no memory)")

pooled_comparison("competition_no_memory", "competition_with_memory",
                  "B (no memory)", "C (with memory)")

pooled_comparison("control", "competition_with_memory",
                  "A (control)", "C (with memory)")

# Summary compliance rates
print(f"\n{'='*65}")
print("COMPLIANCE RATES SUMMARY")
print(f"{'='*65}")
for cond in ["control", "competition_no_memory", "competition_with_memory"]:
    rates = []
    for seed in SEEDS:
        path = Path("results") / f"{cond}_seed{seed}" / "rounds.jsonl"
        total, compliant = 0, 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["phase"] == "phase2":
                    total += 1
                    if not r["is_safe_honest"]:
                        compliant += 1
        rates.append(compliant / total * 100)
    mean_rate = sum(rates) / len(rates)
    print(f"{cond}: {rates[0]:.1f}%, {rates[1]:.1f}%, {rates[2]:.1f}% (mean {mean_rate:.1f}%)")
