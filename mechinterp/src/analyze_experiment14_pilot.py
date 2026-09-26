import json
import numpy as np
from pathlib import Path

RESULT = Path(
    "mechinterp/results/experiment14_pilot/results.json"
)

with open(RESULT) as f:
    results = json.load(f)

LAYERS = [9, 11, 18, 24]
GROUPS = ["flip", "stable"]

print("=" * 90)
print("EXPERIMENT 14A — PILOT ANALYSIS")
print("=" * 90)

print(f"Total comparisons: {len(results)}")

# ================================================================
# BASIC CHECKS
# ================================================================

for layer in LAYERS:

    subset = [
        r for r in results
        if r["layer"] == layer
    ]

    print(
        f"L{layer}: {len(subset)} comparisons"
    )

print()


# ================================================================
# FIRST TOKEN DIVERGENCE
# ================================================================

print("=" * 90)
print("1. FIRST TOKEN DIVERGENCE")
print("=" * 90)

print(
    "first_diff = 0 means the very first generated token differed."
)
print(
    "first_diff = 1 means token 0 was identical, token 1 differed."
)
print(
    "None means the complete generated sequences were identical."
)
print()

for layer in LAYERS:

    subset = [
        r for r in results
        if r["layer"] == layer
    ]

    first = [
        r["first_token_difference"]
        for r in subset
    ]

    changed = [
        x for x in first
        if x is not None
    ]

    no_change = sum(
        x is None for x in first
    )

    zero = sum(
        x == 0 for x in first
    )

    after_zero = sum(
        x is not None and x > 0
        for x in first
    )

    print(
        f"L{layer}: "
        f"any token difference = "
        f"{len(changed)}/{len(first)} "
        f"({100*len(changed)/len(first):.1f}%) | "
        f"first-token difference = "
        f"{zero}/{len(first)} "
        f"({100*zero/len(first):.1f}%) | "
        f"later first difference = "
        f"{after_zero}/{len(first)} "
        f"({100*after_zero/len(first):.1f}%) | "
        f"identical = "
        f"{no_change}/{len(first)} "
        f"({100*no_change/len(first):.1f}%)"
    )


# ================================================================
# FLIP VS STABLE
# ================================================================

print()
print("=" * 90)
print("2. FIRST TOKEN DIVERGENCE — FLIP VS STABLE")
print("=" * 90)

for layer in LAYERS:

    print(f"\nL{layer}")

    for group in GROUPS:

        subset = [
            r for r in results
            if r["layer"] == layer
            and r["group"] == group
        ]

        first = [
            r["first_token_difference"]
            for r in subset
        ]

        any_diff = [
            x for x in first
            if x is not None
        ]

        zero = sum(
            x == 0 for x in first
        )

        later = sum(
            x is not None and x > 0
            for x in first
        )

        identical = sum(
            x is None for x in first
        )

        print(
            f"  {group:6s}: "
            f"N={len(first):2d} | "
            f"first-token={zero:2d} "
            f"({100*zero/len(first):5.1f}%) | "
            f"later={later:2d} "
            f"({100*later/len(first):5.1f}%) | "
            f"identical={identical:2d} "
            f"({100*identical/len(first):5.1f}%)"
        )


# ================================================================
# PRE-DIVERGENCE COSINE
# ================================================================

print()
print("=" * 90)
print("3. INTERNAL STATE SIMILARITY BEFORE TOKEN DIVERGENCE")
print("=" * 90)

print(
    "This is the critical analysis."
)
print(
    "If the first token differs at position k, "
    "pre_divergence_mean_cosine measures L31 similarity "
    "before that token difference."
)
print()

for layer in LAYERS:

    print(f"\nL{layer}")

    for group in GROUPS:

        subset = [
            r for r in results
            if r["layer"] == layer
            and r["group"] == group
            and r["pre_divergence_mean_cosine"] is not None
        ]

        vals = np.array([
            r["pre_divergence_mean_cosine"]
            for r in subset
        ])

        if len(vals) == 0:

            print(
                f"  {group:6s}: no pre-divergence data"
            )

            continue

        print(
            f"  {group:6s}: "
            f"N={len(vals):2d} | "
            f"mean={np.mean(vals):.4f} | "
            f"std={np.std(vals, ddof=1) if len(vals)>1 else 0:.4f} | "
            f"min={np.min(vals):.4f} | "
            f"max={np.max(vals):.4f}"
        )


# ================================================================
# FIRST TOKEN DIFFERENCE + PRE-COSINE
# ================================================================

print()
print("=" * 90)
print("4. CASES WHERE FIRST TOKEN DIFFERS IMMEDIATELY")
print("=" * 90)

for layer in LAYERS:

    subset = [
        r for r in results
        if r["layer"] == layer
        and r["first_token_difference"] == 0
    ]

    print(f"\nL{layer}: {len(subset)} immediate divergences")

    for group in GROUPS:

        vals = [
            r["pre_divergence_mean_cosine"]
            for r in subset
            if r["group"] == group
            and r["pre_divergence_mean_cosine"] is not None
        ]

        if vals:

            print(
                f"  {group:6s}: "
                f"N={len(vals)} | "
                f"mean pre-cos={np.mean(vals):.4f}"
            )

        else:

            print(
                f"  {group:6s}: N=0"
            )


# ================================================================
# TOKEN DIVERGENCE POSITION
# ================================================================

print()
print("=" * 90)
print("5. WHERE DOES TOKEN DIVERGENCE OCCUR?")
print("=" * 90)

for layer in LAYERS:

    print(f"\nL{layer}")

    for group in GROUPS:

        vals = [
            r["first_token_difference"]
            for r in results
            if r["layer"] == layer
            and r["group"] == group
            and r["first_token_difference"] is not None
        ]

        if vals:

            print(
                f"  {group:6s}: "
                f"N={len(vals)} | "
                f"mean={np.mean(vals):.2f} | "
                f"median={np.median(vals):.1f} | "
                f"min={np.min(vals)} | "
                f"max={np.max(vals)}"
            )

        else:

            print(
                f"  {group:6s}: no divergence"
            )


# ================================================================
# POST-DIVERGENCE COSINE
# ================================================================

print()
print("=" * 90)
print("6. POST-DIVERGENCE COSINE")
print("=" * 90)

for layer in LAYERS:

    print(f"\nL{layer}")

    for group in GROUPS:

        vals = [
            r["post_divergence_mean_cosine"]
            for r in results
            if r["layer"] == layer
            and r["group"] == group
            and r["post_divergence_mean_cosine"] is not None
        ]

        if vals:

            print(
                f"  {group:6s}: "
                f"N={len(vals):2d} | "
                f"mean={np.mean(vals):.4f} | "
                f"std={np.std(vals, ddof=1) if len(vals)>1 else 0:.4f}"
            )

        else:

            print(
                f"  {group:6s}: no post-divergence data"
            )


# ================================================================
# DIRECT LAYER COMPARISON
# ================================================================

print()
print("=" * 90)
print("7. COMPACT SUMMARY")
print("=" * 90)

print(
    f"{'Layer':>5} | "
    f"{'Group':>7} | "
    f"{'N':>3} | "
    f"{'First token diff %':>18} | "
    f"{'Any diff %':>12} | "
    f"{'Identical %':>12} | "
    f"{'Pre-cos':>10} | "
    f"{'Post-cos':>10}"
)

print("-" * 90)

for layer in LAYERS:

    for group in GROUPS:

        subset = [
            r for r in results
            if r["layer"] == layer
            and r["group"] == group
        ]

        n = len(subset)

        first = [
            r["first_token_difference"]
            for r in subset
        ]

        immediate = sum(
            x == 0 for x in first
        )

        any_diff = sum(
            x is not None for x in first
        )

        identical = sum(
            x is None for x in first
        )

        pre = [
            r["pre_divergence_mean_cosine"]
            for r in subset
            if r["pre_divergence_mean_cosine"] is not None
        ]

        post = [
            r["post_divergence_mean_cosine"]
            for r in subset
            if r["post_divergence_mean_cosine"] is not None
        ]

        pre_mean = (
            np.mean(pre)
            if pre else np.nan
        )

        post_mean = (
            np.mean(post)
            if post else np.nan
        )

        print(
            f"{layer:>5} | "
            f"{group:>7} | "
            f"{n:>3} | "
            f"{100*immediate/n:>17.1f}% | "
            f"{100*any_diff/n:>11.1f}% | "
            f"{100*identical/n:>11.1f}% | "
            f"{pre_mean:>10.4f} | "
            f"{post_mean:>10.4f}"
        )


# ================================================================
# INTERPRETATION GUIDE
# ================================================================

print()
print("=" * 90)
print("HOW WE WILL INTERPRET THIS")
print("=" * 90)

print(
"""
The key question is temporal ordering.

CASE A:
    Internal states already differ before the first token differs.

    This is consistent with:
        intervention
             ↓
        internal divergence
             ↓
        different token

CASE B:
    States remain very similar until the first token differs,
    then become strongly different.

    This is consistent with:
        intervention
             ↓
        different token
             ↓
        different autoregressive trajectory

CASE C:
    Some divergence exists before the token difference,
    followed by much larger divergence afterward.

    This suggests:
        initial internal effect
             +
        amplification through autoregressive generation

We should NOT use this experiment alone to identify
a "safety decision layer."

It is specifically a test of the temporal relationship
between intervention, internal divergence, and natural
token generation.
"""
)

print("=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
