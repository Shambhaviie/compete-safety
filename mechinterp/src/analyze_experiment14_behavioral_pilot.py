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

print("=" * 110)
print("EXPERIMENT 14A — BEHAVIOURAL OUTCOME ANALYSIS")
print("=" * 110)

# ------------------------------------------------------------------
# IMPORTANT:
# safe=True  = REFUSAL
# safe=False = COMPLIANCE
# ------------------------------------------------------------------

def label(x):
    return "refusal" if bool(x) else "compliance"


# ------------------------------------------------------------------
# 1. BASIC BEHAVIOURAL OUTCOMES
# ------------------------------------------------------------------

print()
print("=" * 110)
print("1. BEHAVIOURAL OUTCOMES AFTER ABLATION")
print("=" * 110)

print(
    f"{'Layer':>5} | "
    f"{'Group':>7} | "
    f"{'N':>3} | "
    f"{'Natural':>11} | "
    f"{'Abl Ref':>8} | "
    f"{'Abl Comp':>8} | "
    f"{'Behaviour flip':>15}"
)
print("-" * 110)

behaviour_rows = []

for layer in LAYERS:

    for group in GROUPS:

        subset = [
            r for r in results
            if int(r["layer"]) == layer
            and r["group"] == group
        ]

        n = len(subset)

        if group == "flip":
            # Natural C behaviour = compliance
            natural = "compliance"

            flipped = sum(
                bool(r["safety_label"])
                for r in subset
            )

        else:
            # Natural C behaviour = refusal
            natural = "refusal"

            flipped = sum(
                not bool(r["safety_label"])
                for r in subset
            )

        abl_ref = sum(
            bool(r["safety_label"])
            for r in subset
        )

        abl_comp = n - abl_ref

        print(
            f"{layer:>5} | "
            f"{group:>7} | "
            f"{n:>3} | "
            f"{natural:>11} | "
            f"{abl_ref:>8} | "
            f"{abl_comp:>8} | "
            f"{flipped:>5}/{n:<3} "
            f"({100*flipped/n:>5.1f}%)"
        )

        behaviour_rows.append({
            "layer": layer,
            "group": group,
            "n": n,
            "natural_behaviour": natural,
            "ablated_refusal": abl_ref,
            "ablated_compliance": abl_comp,
            "behaviour_flips": flipped,
            "behaviour_flip_pct": 100 * flipped / n,
        })


# ------------------------------------------------------------------
# 2. BEHAVIOUR × FIRST TOKEN DIVERGENCE
# ------------------------------------------------------------------

print()
print("=" * 110)
print("2. BEHAVIOURAL FLIP × FIRST TOKEN DIVERGENCE")
print("=" * 110)

print(
    "This is the important analysis: token divergence is NOT "
    "treated as behavioural divergence."
)
print()

print(
    f"{'Layer':>5} | "
    f"{'Group':>7} | "
    f"{'Outcome':>20} | "
    f"{'N':>3} | "
    f"{'Token diff':>10} | "
    f"{'Immediate':>10} | "
    f"{'Later':>8} | "
    f"{'Identical':>10}"
)

print("-" * 110)

cross_rows = []

for layer in LAYERS:

    for group in GROUPS:

        subset = [
            r for r in results
            if int(r["layer"]) == layer
            and r["group"] == group
        ]

        for outcome_name, is_flip in [
            ("BEHAVIOUR FLIPPED", True),
            ("BEHAVIOUR PRESERVED", False),
        ]:

            selected = []

            for r in subset:

                safety = bool(r["safety_label"])

                if group == "flip":
                    # Natural = compliance
                    behavioural_flip = safety
                else:
                    # Natural = refusal
                    behavioural_flip = not safety

                if behavioural_flip == is_flip:
                    selected.append(r)

            n = len(selected)

            if n == 0:
                print(
                    f"{layer:>5} | "
                    f"{group:>7} | "
                    f"{outcome_name:>20} | "
                    f"{0:>3} |"
                )
                continue

            first = [
                r["first_token_difference"]
                for r in selected
            ]

            any_diff = sum(
                x is not None
                for x in first
            )

            immediate = sum(
                x == 0
                for x in first
            )

            later = sum(
                x is not None and x > 0
                for x in first
            )

            identical = sum(
                x is None
                for x in first
            )

            print(
                f"{layer:>5} | "
                f"{group:>7} | "
                f"{outcome_name:>20} | "
                f"{n:>3} | "
                f"{any_diff:>4}/{n:<4} "
                f"({100*any_diff/n:>4.1f}%) | "
                f"{immediate:>4}/{n:<4} "
                f"({100*immediate/n:>4.1f}%) | "
                f"{later:>3}/{n:<3} "
                f"({100*later/n:>4.1f}%) | "
                f"{identical:>4}/{n:<4} "
                f"({100*identical/n:>4.1f}%)"
            )

            cross_rows.append({
                "layer": layer,
                "group": group,
                "outcome": outcome_name,
                "n": n,
                "any_token_difference": any_diff,
                "any_token_difference_pct":
                    100 * any_diff / n,
                "immediate_difference": immediate,
                "immediate_difference_pct":
                    100 * immediate / n,
                "later_difference": later,
                "later_difference_pct":
                    100 * later / n,
                "identical": identical,
                "identical_pct":
                    100 * identical / n,
            })


# ------------------------------------------------------------------
# 3. BEHAVIOURAL FLIP TABLE
# ------------------------------------------------------------------

print()
print("=" * 110)
print("3. BEHAVIOURAL FLIP TABLE")
print("=" * 110)

print(
    f"{'Layer':>5} | "
    f"{'Flip→Refusal':>15} | "
    f"{'Stable→Compliance':>19} | "
    f"{'Total flips':>12}"
)
print("-" * 110)

for layer in LAYERS:

    flip_subset = [
        r for r in results
        if int(r["layer"]) == layer
        and r["group"] == "flip"
    ]

    stable_subset = [
        r for r in results
        if int(r["layer"]) == layer
        and r["group"] == "stable"
    ]

    flip_to_refusal = sum(
        bool(r["safety_label"])
        for r in flip_subset
    )

    stable_to_compliance = sum(
        not bool(r["safety_label"])
        for r in stable_subset
    )

    print(
        f"{layer:>5} | "
        f"{flip_to_refusal:>5}/"
        f"{len(flip_subset):<5} "
        f"({100*flip_to_refusal/len(flip_subset):>5.1f}%) | "
        f"{stable_to_compliance:>5}/"
        f"{len(stable_subset):<5} "
        f"({100*stable_to_compliance/len(stable_subset):>5.1f}%) | "
        f"{flip_to_refusal + stable_to_compliance:>5}"
    )


# ------------------------------------------------------------------
# 4. BEHAVIOURAL FLIP VS TOKEN DIVERGENCE
# ------------------------------------------------------------------

print()
print("=" * 110)
print("4. KEY QUESTION: DOES TOKEN DIVERGENCE PREDICT BEHAVIOURAL CHANGE?")
print("=" * 110)

print(
    "For each layer/group, compare the proportion of token-divergent"
)
print(
    "cases among behavioural flips versus behaviourally preserved cases."
)
print()

print(
    f"{'Layer':>5} | "
    f"{'Group':>7} | "
    f"{'Flip any-diff':>14} | "
    f"{'Preserved any-diff':>19} | "
    f"{'Difference':>12}"
)

print("-" * 110)

for layer in LAYERS:

    for group in GROUPS:

        rows = [
            r for r in cross_rows
            if r["layer"] == layer
            and r["group"] == group
        ]

        flip_row = next(
            (r for r in rows
             if r["outcome"] == "BEHAVIOUR FLIPPED"),
            None
        )

        preserved_row = next(
            (r for r in rows
             if r["outcome"] == "BEHAVIOUR PRESERVED"),
            None
        )

        if flip_row is None or preserved_row is None:
            continue

        diff = (
            flip_row["any_token_difference_pct"]
            - preserved_row["any_token_difference_pct"]
        )

        print(
            f"{layer:>5} | "
            f"{group:>7} | "
            f"{flip_row['any_token_difference_pct']:>13.1f}% | "
            f"{preserved_row['any_token_difference_pct']:>18.1f}% | "
            f"{diff:>11.1f} pp"
        )


# ------------------------------------------------------------------
# 5. SAVE MACHINE-READABLE RESULTS
# ------------------------------------------------------------------

OUT = Path(
    "mechinterp/results/experiment14_pilot/"
    "behavioral_analysis.json"
)

with open(OUT, "w") as f:
    json.dump(
        {
            "behavioural_summary": behaviour_rows,
            "behaviour_vs_token_divergence": cross_rows,
        },
        f,
        indent=2
    )

print()
print("=" * 110)
print(f"Saved: {OUT}")
print("=" * 110)

