"""
Category-level compliance breakdown across conditions and seeds.
For each HarmBench harm category, computes compliance rate per condition,
pooled across all three seeds. Also runs Fisher's exact test per category
for A vs C (the full effect comparison) to identify which categories
show the strongest signal.
"""

import json
from pathlib import Path
from collections import defaultdict


SEEDS = [42, 123, 777]
CONDITIONS = ["control", "competition_no_memory", "competition_with_memory"]


def load_phase2_by_category(run_id: str) -> dict:
    """Returns {category: {"total": n, "compliant": n}}"""
    path = Path("results") / run_id / "rounds.jsonl"
    stats = defaultdict(lambda: {"total": 0, "compliant": 0})
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["phase"] != "phase2":
                continue
            cat = r.get("category", "unknown")
            stats[cat]["total"] += 1
            if not r["is_safe_honest"]:
                stats[cat]["compliant"] += 1
    return dict(stats)


def fisher_exact_2x2(a, b, c, d):
    """
    Fisher's exact test for a 2x2 table:
    [[a, b], [c, d]]
    where a = complied in cond1, b = refused in cond1,
          c = complied in cond2, d = refused in cond2.
    Uses scipy if available, otherwise returns approximate p via chi2.
    """
    try:
        from scipy.stats import fisher_exact
        import numpy as np
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return p
    except ImportError:
        import math
        n = a + b + c + d
        if n == 0:
            return 1.0
        chi2 = n * (a*d - b*c)**2 / ((a+b)*(c+d)*(a+c)*(b+d) + 1e-10)
        p = 2 * (1 - math.erf(math.sqrt(chi2 / 2)))
        return min(p, 1.0)


# Pool across seeds
pooled = {}
for cond in CONDITIONS:
    pooled[cond] = defaultdict(lambda: {"total": 0, "compliant": 0})
    for seed in SEEDS:
        run_id = f"{cond}_seed{seed}"
        cat_stats = load_phase2_by_category(run_id)
        for cat, stats in cat_stats.items():
            pooled[cond][cat]["total"] += stats["total"]
            pooled[cond][cat]["compliant"] += stats["compliant"]

# Get all categories
all_cats = sorted(set(
    cat for cond in CONDITIONS
    for cat in pooled[cond].keys()
))

print("CATEGORY-LEVEL COMPLIANCE BREAKDOWN")
print("Pooled across seeds 42, 123, 777")
print(f"{'='*75}\n")

print(f"{'Category':<30} {'Control':>10} {'No Memory':>12} {'With Memory':>12} {'Fisher p':>10}")
print(f"{'':.<30} {'(A)':>10} {'(B)':>12} {'(C)':>12} {'(A vs C)':>10}")
print("-" * 75)

category_results = []

for cat in all_cats:
    a_stats = pooled["control"][cat]
    b_stats = pooled["competition_no_memory"][cat]
    c_stats = pooled["competition_with_memory"][cat]

    a_rate = a_stats["compliant"] / a_stats["total"] * 100 if a_stats["total"] > 0 else 0
    b_rate = b_stats["compliant"] / b_stats["total"] * 100 if b_stats["total"] > 0 else 0
    c_rate = c_stats["compliant"] / c_stats["total"] * 100 if c_stats["total"] > 0 else 0

    # Fisher's exact for A vs C
    a_compliant = a_stats["compliant"]
    a_refused = a_stats["total"] - a_compliant
    c_compliant = c_stats["compliant"]
    c_refused = c_stats["total"] - c_compliant

    p = fisher_exact_2x2(a_compliant, a_refused, c_compliant, c_refused)
    sig = "*" if p < 0.05 else ""

    category_results.append((cat, a_rate, b_rate, c_rate, p, a_stats["total"]))

    print(f"{cat:<30} {a_rate:>9.1f}% {b_rate:>11.1f}% {c_rate:>11.1f}% {p:>9.4f}{sig}")

print(f"\n* = significant at p<0.05 (Fisher's exact, A vs C, two-sided)")
print(f"\nNote: trial counts per category vary. Total trials per condition = 396 (132 x 3 seeds).")

# Sort by effect size (C rate - A rate) and show top categories
print(f"\n{'='*75}")
print("CATEGORIES RANKED BY EFFECT SIZE (C compliance rate - A compliance rate)")
print(f"{'='*75}")
sorted_by_effect = sorted(category_results, key=lambda x: -(x[3] - x[1]))
for cat, a_rate, b_rate, c_rate, p, n_total in sorted_by_effect:
    effect = c_rate - a_rate
    sig = "*" if p < 0.05 else " "
    print(f"{cat:<30} effect={effect:+.1f}pp  A={a_rate:.1f}%  B={b_rate:.1f}%  C={c_rate:.1f}%  p={p:.4f}{sig}")

print(f"\npp = percentage points")
