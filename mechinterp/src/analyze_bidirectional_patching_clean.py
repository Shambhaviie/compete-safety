#!/usr/bin/env python3

import json
import csv
from pathlib import Path
from collections import Counter

A2C_PATH = Path(
    "mechinterp/results/activation_patch_A_to_C_full_163/"
    "full_A_to_C_20260907_per_prompt.jsonl"
)

C2A_PATH = Path(
    "mechinterp/results/activation_patch_C_to_A/"
    "C_to_A_per_prompt.jsonl"
)

OUT = Path(
    "mechinterp/results/activation_patch_bidirectional_analysis"
)
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


a2c = load_jsonl(A2C_PATH)
c2a = load_jsonl(C2A_PATH)

print("=" * 80)
print("CLEAN BIDIRECTIONAL ACTIVATION-PATCHING ANALYSIS")
print("=" * 80)

print(f"A→C rows: {len(a2c)}")
print(f"C→A rows: {len(c2a)}")


# ------------------------------------------------------------------
# Index by (prompt_id, layer)
# ------------------------------------------------------------------

a2c_map = {
    (r["prompt_id"], int(r["layer"])): r
    for r in a2c
}

c2a_map = {
    (r["prompt_id"], int(r["layer"])): r
    for r in c2a
}

prompts = sorted(set(r["prompt_id"] for r in a2c))

print(f"Unique prompts: {len(prompts)}")


# ------------------------------------------------------------------
# Sanity check experimental semantics
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("SEMANTIC SANITY CHECK")
print("=" * 80)

# A→C: target C should naturally comply.
a2c_natural = [
    next(
        r for r in a2c
        if r["prompt_id"] == pid
        and int(r["layer"]) == 14
    )
    for pid in prompts
]

print("\nA→C target natural state:")
print(
    Counter(
        (
            r["natural_refusal"],
            r["natural_compliance"],
            r["natural_safe"]
        )
        for r in a2c_natural
    )
)

# C→A: target A should naturally refuse.
c2a_natural = [
    next(
        r for r in c2a
        if r["prompt_id"] == pid
        and int(r["layer"]) == 0
    )
    for pid in prompts
]

print("\nC→A target natural state:")
print(
    Counter(
        (
            r["natural_refusal"],
            r["natural_compliance"],
            r["natural_safe"]
        )
        for r in c2a_natural
    )
)


# ------------------------------------------------------------------
# Explicit causal events
# ------------------------------------------------------------------

def A_to_C_flip(r):
    """
    C naturally complies.
    A activation is injected into C.
    C becomes refusal.
    """
    return (
        r["natural_compliance"] is True
        and r["patched_refusal"] is True
    )


def C_to_A_flip(r):
    """
    A naturally refuses.
    C activation is injected into A.
    A becomes compliance.
    """
    return (
        r["natural_refusal"] is True
        and r["patched_compliance"] is True
    )


A2C_LAYERS = [14, 16, 18, 20]


# ------------------------------------------------------------------
# Build paired prompt table
# ------------------------------------------------------------------

rows = []

for pid in prompts:

    arows = {
        int(r["layer"]): r
        for r in a2c
        if r["prompt_id"] == pid
    }

    crows = {
        int(r["layer"]): r
        for r in c2a
        if r["prompt_id"] == pid
    }

    base = arows[14]

    row = {
        "prompt_id": pid,
        "index": base["index"],
        "category": base["category"],
    }

    # A→C
    for layer in A2C_LAYERS:
        r = arows[layer]

        row[f"A2C_L{layer}"] = A_to_C_flip(r)

    # C→A
    for layer in range(32):
        r = crows[layer]

        row[f"C2A_L{layer}"] = C_to_A_flip(r)

    # Any effect
    row["A2C_any"] = any(
        row[f"A2C_L{layer}"]
        for layer in A2C_LAYERS
    )

    row["C2A_any"] = any(
        row[f"C2A_L{layer}"]
        for layer in range(32)
    )

    # Earliest
    row["A2C_earliest"] = next(
        (
            layer
            for layer in A2C_LAYERS
            if row[f"A2C_L{layer}"]
        ),
        None
    )

    row["C2A_earliest"] = next(
        (
            layer
            for layer in range(32)
            if row[f"C2A_L{layer}"]
        ),
        None
    )

    # Explicit classification
    if row["A2C_any"] and row["C2A_any"]:
        row["classification"] = "bidirectional"
    elif row["A2C_any"]:
        row["classification"] = "A_to_C_only"
    elif row["C2A_any"]:
        row["classification"] = "C_to_A_only"
    else:
        row["classification"] = "neither"

    rows.append(row)


# ------------------------------------------------------------------
# Save complete paired table
# ------------------------------------------------------------------

all_fields = list(rows[0].keys())

with open(OUT / "paired_prompt_analysis.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(rows)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

n = len(rows)

a_any = sum(r["A2C_any"] for r in rows)
c_any = sum(r["C2A_any"] for r in rows)

both = sum(
    r["classification"] == "bidirectional"
    for r in rows
)

a_only = sum(
    r["classification"] == "A_to_C_only"
    for r in rows
)

c_only = sum(
    r["classification"] == "C_to_A_only"
    for r in rows
)

neither = sum(
    r["classification"] == "neither"
    for r in rows
)

print("\n" + "=" * 80)
print("PAIRED CAUSAL RESULTS")
print("=" * 80)

print(f"\nN = {n}")

print(
    f"A→C: {a_any}/{n} "
    f"({100*a_any/n:.1f}%)"
)

print(
    f"C→A: {c_any}/{n} "
    f"({100*c_any/n:.1f}%)"
)

print("\nClassification:")
print(f"  Bidirectional : {both}")
print(f"  A→C only      : {a_only}")
print(f"  C→A only      : {c_only}")
print(f"  Neither       : {neither}")


# ------------------------------------------------------------------
# A→C layer counts
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("A→C: A STATE INJECTED INTO C")
print("=" * 80)

for layer in A2C_LAYERS:

    count = sum(
        r[f"A2C_L{layer}"]
        for r in rows
    )

    print(
        f"L{layer}: "
        f"{count}/{n} "
        f"({100*count/n:.1f}%)"
    )


# ------------------------------------------------------------------
# C→A layer counts
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("C→A: C STATE INJECTED INTO A")
print("=" * 80)

for layer in range(32):

    count = sum(
        r[f"C2A_L{layer}"]
        for r in rows
    )

    if count:
        print(
            f"L{layer}: "
            f"{count}/{n} "
            f"({100*count/n:.1f}%)"
        )


# ------------------------------------------------------------------
# Earliest layer distributions
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("EARLIEST EFFECTIVE LAYERS")
print("=" * 80)

a_earliest = Counter(
    r["A2C_earliest"]
    for r in rows
    if r["A2C_earliest"] is not None
)

c_earliest = Counter(
    r["C2A_earliest"]
    for r in rows
    if r["C2A_earliest"] is not None
)

print("\nA→C:")
for layer, count in sorted(a_earliest.items()):
    print(f"  L{layer}: {count}")

print("\nC→A:")
for layer, count in sorted(c_earliest.items()):
    print(f"  L{layer}: {count}")


# ------------------------------------------------------------------
# Bidirectional cases
# ------------------------------------------------------------------

bi = [
    r for r in rows
    if r["classification"] == "bidirectional"
]

print("\n" + "=" * 80)
print("BIDIRECTIONAL CASES")
print("=" * 80)

print(f"\nN = {len(bi)}")

for r in bi:
    print(
        f'{r["prompt_id"]:12s} '
        f'idx={r["index"]:3d} '
        f'{r["category"]:25s} '
        f'A→C=L{r["A2C_earliest"]:<2} '
        f'C→A=L{r["C2A_earliest"]:<2}'
    )


# ------------------------------------------------------------------
# Compare earliest layers for bidirectional cases
# ------------------------------------------------------------------

if bi:

    differences = [
        r["C2A_earliest"] - r["A2C_earliest"]
        for r in bi
    ]

    differences_sorted = sorted(differences)

    mean_diff = sum(differences) / len(differences)
    median_diff = differences_sorted[len(differences)//2]

    print("\n" + "=" * 80)
    print("EARLIEST-LAYER ASYMMETRY")
    print("=" * 80)

    print(
        f"\nC→A minus A→C:"
    )

    print(f"  Mean   = {mean_diff:.2f} layers")
    print(f"  Median = {median_diff:.1f} layers")

    print(
        f"  A→C earlier: "
        f"{sum(d > 0 for d in differences)}"
    )

    print(
        f"  C→A earlier: "
        f"{sum(d < 0 for d in differences)}"
    )

    print(
        f"  Same layer: "
        f"{sum(d == 0 for d in differences)}"
    )


# ------------------------------------------------------------------
# Save bidirectional cases
# ------------------------------------------------------------------

bi_fields = [
    "prompt_id",
    "index",
    "category",
    "A2C_earliest",
    "C2A_earliest",
]

with open(
    OUT / "bidirectional_cases.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(f, fieldnames=bi_fields)
    writer.writeheader()

    for r in bi:
        writer.writerow({
            k: r[k]
            for k in bi_fields
        })


# ------------------------------------------------------------------
# Save summary
# ------------------------------------------------------------------

summary = {
    "n_prompts": n,
    "A_to_C_any": a_any,
    "C_to_A_any": c_any,
    "bidirectional": both,
    "A_to_C_only": a_only,
    "C_to_A_only": c_only,
    "neither": neither,
    "A_to_C_layer_counts": {
        str(l): sum(r[f"A2C_L{l}"] for r in rows)
        for l in A2C_LAYERS
    },
    "C_to_A_layer_counts": {
        str(l): sum(r[f"C2A_L{l}"] for r in rows)
        for l in range(32)
    },
    "A_to_C_earliest": dict(a_earliest),
    "C_to_A_earliest": dict(c_earliest),
}

with open(
    OUT / "clean_bidirectional_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)


print("\nSaved:")
print(f"  {OUT / 'paired_prompt_analysis.csv'}")
print(f"  {OUT / 'bidirectional_cases.csv'}")
print(f"  {OUT / 'clean_bidirectional_summary.json'}")

print("\nDONE")
