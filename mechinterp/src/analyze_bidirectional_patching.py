#!/usr/bin/env python3

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

A2C = Path(
    "mechinterp/results/activation_patch_A_to_C_full_163/"
    "full_A_to_C_20260907_per_prompt.jsonl"
)

C2A = Path(
    "mechinterp/results/activation_patch_C_to_A/"
    "C_to_A_per_prompt.jsonl"
)

OUT = Path(
    "mechinterp/results/activation_patch_bidirectional_analysis"
)
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


a2c = load(A2C)
c2a = load(C2A)

print("=" * 80)
print("BIDIRECTIONAL ACTIVATION-PATCHING ANALYSIS")
print("=" * 80)
print(f"A→C rows: {len(a2c)}")
print(f"C→A rows: {len(c2a)}")


# ------------------------------------------------------------
# Index by prompt_id + layer
# ------------------------------------------------------------

a2c_map = {
    (r["prompt_id"], int(r["layer"])): r
    for r in a2c
}

c2a_map = {
    (r["prompt_id"], int(r["layer"])): r
    for r in c2a
}

prompts = sorted(set(r["prompt_id"] for r in a2c))

print(f"Unique A→C prompts: {len(prompts)}")


# ------------------------------------------------------------
# Define actual causal events
# ------------------------------------------------------------

# A→C:
# A naturally COMPLIES.
# Inject C activation.
# Patched response becomes REFUSAL.
#
# This is the correct event for the stored A→C run.
def a2c_flip(r):
    return (
        r["natural_compliance"] is True
        and r["patched_refusal"] is True
    )


# C→A:
# C naturally REFUSES.
# Inject A activation.
# Patched response becomes COMPLIANCE.
#
# This is the reversal event.
def c2a_flip(r):
    return (
        r["natural_refusal"] is True
        and r["patched_compliance"] is True
    )


A_LAYERS = [14, 16, 18, 20]


# ------------------------------------------------------------
# Per-prompt analysis
# ------------------------------------------------------------

rows = []

for pid in prompts:

    arows = {
        int(r["layer"]): r
        for (p, l), r in a2c_map.items()
        if p == pid
    }

    crows = {
        int(r["layer"]): r
        for (p, l), r in c2a_map.items()
        if p == pid
    }

    first_a = next(iter(arows.values()))

    row = {
        "prompt_id": pid,
        "index": first_a["index"],
        "category": first_a["category"],

        # Natural states in the two conditions
        "A_natural_compliance": first_a["natural_compliance"],
        "A_natural_refusal": first_a["natural_refusal"],

        "C_natural_compliance":
            crows.get(0, {}).get("natural_compliance"),
        "C_natural_refusal":
            crows.get(0, {}).get("natural_refusal"),
    }

    # A→C effects
    for layer in A_LAYERS:

        r = arows[layer]

        row[f"A2C_L{layer}_flip"] = a2c_flip(r)

        row[f"A2C_L{layer}_patched_refusal"] = \
            r["patched_refusal"]

        row[f"A2C_L{layer}_delta"] = \
            r["patch_max_abs_change"]

    # C→A effects at every layer
    for layer in range(32):

        r = crows[layer]

        row[f"C2A_L{layer}_flip"] = c2a_flip(r)

    # Any effect in each direction
    row["A2C_any_flip"] = any(
        row[f"A2C_L{l}_flip"]
        for l in A_LAYERS
    )

    row["C2A_any_flip"] = any(
        row[f"C2A_L{l}_flip"]
        for l in range(32)
    )

    # Earliest A→C
    row["A2C_earliest_layer"] = next(
        (
            l for l in A_LAYERS
            if row[f"A2C_L{l}_flip"]
        ),
        None
    )

    # Earliest C→A
    row["C2A_earliest_layer"] = next(
        (
            l for l in range(32)
            if row[f"C2A_L{l}_flip"]
        ),
        None
    )

    # Bidirectional
    if row["A2C_any_flip"] and row["C2A_any_flip"]:
        row["class"] = "both"
    elif row["A2C_any_flip"]:
        row["class"] = "A_to_C_only"
    elif row["C2A_any_flip"]:
        row["class"] = "C_to_A_only"
    else:
        row["class"] = "neither"

    rows.append(row)


# ------------------------------------------------------------
# Save prompt-level data
# ------------------------------------------------------------

csv_path = OUT / "prompt_level_bidirectional.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved:")
print(f"  {csv_path}")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

n = len(rows)

a_any = sum(r["A2C_any_flip"] for r in rows)
c_any = sum(r["C2A_any_flip"] for r in rows)

both = sum(r["class"] == "both" for r in rows)
a_only = sum(r["class"] == "A_to_C_only" for r in rows)
c_only = sum(r["class"] == "C_to_A_only" for r in rows)
neither = sum(r["class"] == "neither" for r in rows)

print("\n" + "=" * 80)
print("BIDIRECTIONAL SUMMARY")
print("=" * 80)

print(f"N prompts: {n}")

print(
    f"A→C: {a_any}/{n} "
    f"({100*a_any/n:.1f}%)"
)

print(
    f"C→A: {c_any}/{n} "
    f"({100*c_any/n:.1f}%)"
)

print("\nPrompt classification:")
print(f"  Both:        {both}")
print(f"  A→C only:    {a_only}")
print(f"  C→A only:    {c_only}")
print(f"  Neither:     {neither}")


# ------------------------------------------------------------
# A→C layer effects
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("A→C: COMPETITIVE STATE INJECTED INTO A")
print("=" * 80)

a_layer_counts = {}

for layer in A_LAYERS:

    count = sum(
        r[f"A2C_L{layer}_flip"]
        for r in rows
    )

    a_layer_counts[layer] = count

    print(
        f"L{layer}: "
        f"{count}/{n} "
        f"({100*count/n:.1f}%)"
    )


# ------------------------------------------------------------
# C→A layer effects
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("C→A: ISOLATED REFUSAL STATE INJECTED INTO C")
print("=" * 80)

c_layer_counts = {}

for layer in range(32):

    count = sum(
        r[f"C2A_L{layer}_flip"]
        for r in rows
    )

    c_layer_counts[layer] = count

    if count:
        print(
            f"L{layer}: "
            f"{count}/{n} "
            f"({100*count/n:.1f}%)"
        )


# ------------------------------------------------------------
# Earliest layers
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("EARLIEST FLIP DISTRIBUTIONS")
print("=" * 80)

a_earliest = Counter(
    r["A2C_earliest_layer"]
    for r in rows
    if r["A2C_earliest_layer"] is not None
)

c_earliest = Counter(
    r["C2A_earliest_layer"]
    for r in rows
    if r["C2A_earliest_layer"] is not None
)

print("\nA→C:")
for l, c in sorted(a_earliest.items()):
    print(f"  L{l}: {c}")

print("\nC→A:")
for l, c in sorted(c_earliest.items()):
    print(f"  L{l}: {c}")


# ------------------------------------------------------------
# Natural-state consistency
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("NATURAL STATE CHECK")
print("=" * 80)

print(
    "\nA natural state:"
)
print(
    Counter(
        (r["A_natural_refusal"], r["A_natural_compliance"])
        for r in rows
    )
)

print(
    "\nC natural state:"
)
print(
    Counter(
        (r["C_natural_refusal"], r["C_natural_compliance"])
        for r in rows
    )
)


# ------------------------------------------------------------
# Category analysis
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CATEGORY ANALYSIS")
print("=" * 80)

cats = sorted(set(r["category"] for r in rows))

cat_rows = []

for cat in cats:

    rs = [r for r in rows if r["category"] == cat]
    m = len(rs)

    aa = sum(r["A2C_any_flip"] for r in rs)
    cc = sum(r["C2A_any_flip"] for r in rs)
    bb = sum(r["class"] == "both" for r in rs)

    out = {
        "category": cat,
        "n": m,
        "A2C": aa,
        "A2C_pct": 100 * aa / m,
        "C2A": cc,
        "C2A_pct": 100 * cc / m,
        "both": bb,
        "both_pct": 100 * bb / m,
    }

    cat_rows.append(out)

    print(
        f"{cat:30s} "
        f"N={m:3d} "
        f"A→C={aa:3d} ({100*aa/m:5.1f}%) "
        f"C→A={cc:3d} ({100*cc/m:5.1f}%) "
        f"both={bb:3d} ({100*bb/m:5.1f}%)"
    )

cat_path = OUT / "category_summary.csv"

with open(cat_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cat_rows[0].keys())
    writer.writeheader()
    writer.writerows(cat_rows)


# ------------------------------------------------------------
# Save summary JSON
# ------------------------------------------------------------

summary = {
    "n_prompts": n,
    "A_to_C": {
        "n_any_flip": a_any,
        "pct_any_flip": 100 * a_any / n,
        "layer_counts": a_layer_counts,
        "earliest_layer_counts": dict(a_earliest),
    },
    "C_to_A": {
        "n_any_flip": c_any,
        "pct_any_flip": 100 * c_any / n,
        "layer_counts": c_layer_counts,
        "earliest_layer_counts": dict(c_earliest),
    },
    "classification": {
        "both": both,
        "A_to_C_only": a_only,
        "C_to_A_only": c_only,
        "neither": neither,
    },
}

with open(OUT / "bidirectional_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved:")
print(f"  {OUT / 'category_summary.csv'}")
print(f"  {OUT / 'bidirectional_summary.json'}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
