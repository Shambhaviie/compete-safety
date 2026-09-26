#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULTS = Path("mechinterp/results/A_vs_C_same_prompt")

INPUT = RESULTS / "prompt_level_results.csv"

FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

flip = df[df["group"] == "flip_A_refusal_to_C_compliance"]
stable = df[df["group"] == "stable_refusal"]

layers = sorted(df["layer"].unique())


# ============================================================
# 1. COSINE SIMILARITY
# ============================================================

summary = (
    df[df["group"].isin([
        "flip_A_refusal_to_C_compliance",
        "stable_refusal"
    ])]
    .groupby(["layer", "group"])["cosine_A_C"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(9, 5))

for group, label in [
    ("flip_A_refusal_to_C_compliance", "A→C flips"),
    ("stable_refusal", "Stable refusals"),
]:

    x = summary[summary["group"] == group]

    plt.plot(
        x["layer"],
        x["mean"],
        marker="o",
        label=label
    )

plt.xlabel("Layer")
plt.ylabel("Cosine similarity")
plt.title("Same-prompt hidden-state similarity: A vs C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "01_cosine_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. L2 DISTANCE
# ============================================================

summary = (
    df[df["group"].isin([
        "flip_A_refusal_to_C_compliance",
        "stable_refusal"
    ])]
    .groupby(["layer", "group"])["l2_A_C"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(9, 5))

for group, label in [
    ("flip_A_refusal_to_C_compliance", "A→C flips"),
    ("stable_refusal", "Stable refusals"),
]:

    x = summary[summary["group"] == group]

    plt.plot(
        x["layer"],
        x["mean"],
        marker="o",
        label=label
    )

plt.xlabel("Layer")
plt.ylabel("L2 distance")
plt.title("Magnitude of same-prompt activation change: A → C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "02_L2_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. RELATIVE L2 DISTANCE
# ============================================================

summary = (
    df[df["group"].isin([
        "flip_A_refusal_to_C_compliance",
        "stable_refusal"
    ])]
    .groupby(["layer", "group"])["relative_l2_A_C"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(9, 5))

for group, label in [
    ("flip_A_refusal_to_C_compliance", "A→C flips"),
    ("stable_refusal", "Stable refusals"),
]:

    x = summary[summary["group"] == group]

    plt.plot(
        x["layer"],
        x["mean"],
        marker="o",
        label=label
    )

plt.xlabel("Layer")
plt.ylabel("Relative L2 change")
plt.title("Relative same-prompt activation change: A → C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "03_relative_L2_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. INDIVIDUAL PROMPT DISTRIBUTIONS
# ============================================================

# These layers are especially useful for inspecting the
# transition region and later representation.
key_layers = [7, 9, 11, 18, 24, 31]

for layer in key_layers:

    if layer not in layers:
        continue

    flip_values = flip[
        flip["layer"] == layer
    ]["cosine_A_C"].values

    stable_values = stable[
        stable["layer"] == layer
    ]["cosine_A_C"].values

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [flip_values, stable_values],
        labels=["A→C flips", "Stable refusals"],
        showfliers=False
    )

    # Overlay individual observations with jitter.
    rng = np.random.default_rng(42)

    for x_pos, values in [
        (1, flip_values),
        (2, stable_values)
    ]:

        jitter = rng.normal(
            0,
            0.045,
            size=len(values)
        )

        plt.scatter(
            np.full(len(values), x_pos) + jitter,
            values,
            alpha=0.18,
            s=10
        )

    plt.ylabel("Cosine similarity: A vs C")
    plt.title(
        f"Prompt-level A vs C similarity — Layer {layer}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / f"04_cosine_distribution_L{layer}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. EFFECT SIZE
# ============================================================

layer_summary = pd.read_csv(
    RESULTS / "layer_summary.csv"
)

plt.figure(figsize=(9, 5))

plt.plot(
    layer_summary["layer"],
    layer_summary["cohens_d_cosine"],
    marker="o"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Layer")
plt.ylabel("Cohen's d")
plt.title(
    "Flip vs stable separation in A→C cosine similarity"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "05_effect_size_cosine.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. L2 EFFECT SIZE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    layer_summary["layer"],
    layer_summary["cohens_d_l2"],
    marker="o"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Layer")
plt.ylabel("Cohen's d")
plt.title(
    "Flip vs stable separation in A→C activation distance"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "06_effect_size_L2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. SUMMARY
# ============================================================

print("=" * 80)
print("PLOTS CREATED")
print("=" * 80)

for path in sorted(FIGURES.glob("*.png")):
    print(path)

print("\nTotal figures:", len(list(FIGURES.glob("*.png"))))
print("\nDone.")
