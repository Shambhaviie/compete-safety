#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

BASE = Path("mechinterp/activations")
OUT = Path("mechinterp/results/A_vs_C_flips")
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

A_FILE = BASE / "activations_prehook_A_seed42.npz"
C_FILE = BASE / "activations_prehook_C_seed42.npz"

A_META = BASE / "metadata_prehook_A_seed42.json"
C_META = BASE / "metadata_prehook_C_seed42.json"


# ============================================================
# HELPERS
# ============================================================

def cosine_rows(x, y):
    numerator = np.sum(x * y, axis=1)
    denominator = (
        np.linalg.norm(x, axis=1)
        * np.linalg.norm(y, axis=1)
    )
    return numerator / np.maximum(denominator, 1e-12)


def safe_mean(x):
    return float(np.mean(x))


def safe_sd(x):
    return float(np.std(x, ddof=1))


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("A -> C FLIP REPRESENTATION ANALYSIS")
print("163 SAME-PROMPT REFUSAL -> COMPLIANCE FLIPS")
print("=" * 80)

A = np.load(A_FILE, allow_pickle=True)
C = np.load(C_FILE, allow_pickle=True)

with open(A_META) as f:
    A_meta = json.load(f)

with open(C_META) as f:
    C_meta = json.load(f)

A_acts = A["prefill_acts"].astype(np.float32)
C_acts = C["prefill_acts"].astype(np.float32)

A_labels = A["replayed_labels"].astype(bool)
C_labels = C["replayed_labels"].astype(bool)

A_ids = [x["prompt_id"] for x in A_meta]
C_ids = [x["prompt_id"] for x in C_meta]


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

assert A_ids == C_ids, "Prompt IDs do not align between A and C."
assert A_acts.shape == C_acts.shape, "Activation shapes differ."

flip_mask = A_labels & (~C_labels)
flip_indices = np.where(flip_mask)[0]

A_flip = A_acts[flip_indices]
C_flip = C_acts[flip_indices]

prompt_ids = [A_ids[i] for i in flip_indices]

n_flips, n_layers, hidden_dim = A_flip.shape

print(f"\nTotal prompts: {len(A_ids)}")
print(f"A -> C flips: {n_flips}")
print(f"Layers: {n_layers}")
print(f"Hidden dimension: {hidden_dim}")

assert n_flips == 163, f"Expected 163 flips, found {n_flips}."


# ============================================================
# ANALYSIS 1:
# SAME-PROMPT A -> C DISTANCE AND COSINE
# ============================================================

layer_rows = []

all_delta = []
all_cosine = []
all_l2 = []
all_relative_l2 = []

for layer in range(n_layers):

    A_layer = A_flip[:, layer, :]
    C_layer = C_flip[:, layer, :]

    delta = C_layer - A_layer

    cosine = cosine_rows(A_layer, C_layer)
    l2 = np.linalg.norm(delta, axis=1)

    A_norm = np.linalg.norm(A_layer, axis=1)
    relative_l2 = l2 / np.maximum(A_norm, 1e-12)

    all_delta.append(delta)
    all_cosine.append(cosine)
    all_l2.append(l2)
    all_relative_l2.append(relative_l2)

    layer_rows.append({
        "layer": layer,

        "cosine_mean": safe_mean(cosine),
        "cosine_sd": safe_sd(cosine),
        "cosine_median": float(np.median(cosine)),

        "l2_mean": safe_mean(l2),
        "l2_sd": safe_sd(l2),
        "l2_median": float(np.median(l2)),

        "relative_l2_mean": safe_mean(relative_l2),
        "relative_l2_sd": safe_sd(relative_l2),
        "relative_l2_median": float(np.median(relative_l2)),
    })


layer_df = pd.DataFrame(layer_rows)

layer_df.to_csv(
    OUT / "layer_representation_change.csv",
    index=False
)


# ============================================================
# ANALYSIS 2:
# IS THE A -> C CHANGE IN A COMMON DIRECTION?
# ============================================================

direction_rows = []

alignment_by_layer = []

for layer in range(n_layers):

    delta = all_delta[layer]

    # Mean displacement across the 163 prompts
    mean_delta = np.mean(delta, axis=0)

    mean_delta_norm = np.linalg.norm(mean_delta)

    # Individual displacement vectors aligned with
    # the mean displacement direction
    mean_direction = mean_delta / max(mean_delta_norm, 1e-12)

    alignment = delta @ mean_direction

    delta_norms = np.linalg.norm(delta, axis=1)

    cosine_to_mean = (
        alignment /
        np.maximum(delta_norms, 1e-12)
    )

    alignment_by_layer.append(cosine_to_mean)

    direction_rows.append({
        "layer": layer,

        "mean_delta_norm": float(mean_delta_norm),

        "alignment_mean":
            float(np.mean(cosine_to_mean)),

        "alignment_sd":
            float(np.std(cosine_to_mean, ddof=1)),

        "alignment_median":
            float(np.median(cosine_to_mean)),

        "alignment_positive_fraction":
            float(np.mean(cosine_to_mean > 0)),

        "alignment_above_0.5_fraction":
            float(np.mean(cosine_to_mean > 0.5)),

        "alignment_above_0.8_fraction":
            float(np.mean(cosine_to_mean > 0.8)),
    })


direction_df = pd.DataFrame(direction_rows)

direction_df.to_csv(
    OUT / "layer_displacement_direction_consistency.csv",
    index=False
)


# ============================================================
# ANALYSIS 3:
# PCA / SVD OF THE 163 DISPLACEMENT VECTORS
# ============================================================

pca_rows = []

# Store PCA results for selected layers
selected_layers = [7, 9, 11, 14, 18, 24, 31]

pca_details = {}


for layer in range(n_layers):

    delta = all_delta[layer]

    # Center across prompts before PCA
    centered = delta - np.mean(delta, axis=0, keepdims=True)

    # SVD
    U, S, Vt = np.linalg.svd(
        centered,
        full_matrices=False
    )

    variance = S ** 2

    total_variance = np.sum(variance)

    # If all centered displacement vectors are identical,
    # there is no prompt-level variance for PCA to explain.
    if total_variance <= 1e-12:
        explained = np.full_like(
            variance,
            np.nan,
            dtype=np.float64
        )

        cumulative = np.full_like(
            variance,
            np.nan,
            dtype=np.float64
        )

        def variance_at(k):
            return np.nan

        n50 = np.nan
        n80 = np.nan
        n90 = np.nan
        n95 = np.nan

    else:
        explained = variance / total_variance
        cumulative = np.cumsum(explained)

        def variance_at(k):
            if k <= len(cumulative):
                return float(cumulative[k - 1])
            return 1.0

        n50 = int(np.searchsorted(cumulative, 0.50) + 1)
        n80 = int(np.searchsorted(cumulative, 0.80) + 1)
        n90 = int(np.searchsorted(cumulative, 0.90) + 1)
        n95 = int(np.searchsorted(cumulative, 0.95) + 1)

    pca_rows.append({
        "layer": layer,

        # Cumulative variance explained.
        # PC1 is the first component alone.
        # PC2 means PC1 + PC2.
        "PC1": variance_at(1),
        "PC2": variance_at(2),
        "PC5": variance_at(5),
        "PC10": variance_at(10),
        "PC20": variance_at(20),
        "PC50": variance_at(50),
        "PC100": variance_at(100),

        "n_components_50pct": n50,
        "n_components_80pct": n80,
        "n_components_90pct": n90,
        "n_components_95pct": n95,
    })

    if layer in selected_layers:
        pca_details[layer] = {
            "U": U,
            "S": S,
            "Vt": Vt,
            "explained": explained,
            "cumulative": cumulative,
            "centered_delta": centered,
        }


pca_df = pd.DataFrame(pca_rows)

pca_df.to_csv(
    OUT / "layer_PCA_displacement.csv",
    index=False
)


# ============================================================
# ANALYSIS 4:
# SAVE PROMPT-LEVEL MEASUREMENTS
# ============================================================

prompt_rows = []

for i, pid in enumerate(prompt_ids):

    for layer in range(n_layers):

        prompt_rows.append({
            "prompt_id": pid,
            "layer": layer,

            "cosine_A_C":
                float(all_cosine[layer][i]),

            "l2_A_C":
                float(all_l2[layer][i]),

            "relative_l2_A_C":
                float(all_relative_l2[layer][i]),

            "delta_alignment_to_mean":
                float(alignment_by_layer[layer][i]),
        })


prompt_df = pd.DataFrame(prompt_rows)

prompt_df.to_csv(
    OUT / "prompt_level_flip_trajectories.csv",
    index=False
)


# ============================================================
# FIGURE 1:
# COSINE A vs C FOR FLIPS
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    layer_df["layer"],
    layer_df["cosine_mean"],
    marker="o"
)

plt.xlabel("Layer")
plt.ylabel("Cosine similarity")
plt.title(
    "Same-prompt representation similarity: A vs C\n"
    "163 refusal → compliance flips"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "01_flip_cosine_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2:
# L2 CHANGE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    layer_df["layer"],
    layer_df["l2_mean"],
    marker="o"
)

plt.xlabel("Layer")
plt.ylabel("L2 distance")
plt.title(
    "Magnitude of A → C representation change\n"
    "163 refusal → compliance flips"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "02_flip_L2_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3:
# RELATIVE L2 CHANGE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    layer_df["layer"],
    layer_df["relative_l2_mean"],
    marker="o"
)

plt.xlabel("Layer")
plt.ylabel("Relative L2 change")
plt.title(
    "Relative A → C representation change\n"
    "163 refusal → compliance flips"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "03_flip_relative_L2_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4:
# COMMON-DIRECTION CONSISTENCY
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    direction_df["layer"],
    direction_df["alignment_mean"],
    marker="o"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Layer")
plt.ylabel("Cosine to mean A → C displacement")
plt.title(
    "Consistency of A → C displacement direction\n"
    "163 refusal → compliance flips"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "04_displacement_direction_consistency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5:
# FRACTION OF PROMPTS ALIGNED WITH COMMON DIRECTION
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    direction_df["layer"],
    direction_df["alignment_positive_fraction"],
    marker="o",
    label="Cosine > 0"
)

plt.plot(
    direction_df["layer"],
    direction_df["alignment_above_0.5_fraction"],
    marker="o",
    label="Cosine > 0.5"
)

plt.plot(
    direction_df["layer"],
    direction_df["alignment_above_0.8_fraction"],
    marker="o",
    label="Cosine > 0.8"
)

plt.xlabel("Layer")
plt.ylabel("Fraction of flipped prompts")
plt.title(
    "Consistency of competition-induced displacement"
)

plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "05_displacement_alignment_fractions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 6:
# PCA VARIANCE EXPLAINED
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    pca_df["layer"],
    pca_df["PC1"],
    marker="o",
    label="PC1"
)

plt.plot(
    pca_df["layer"],
    pca_df["PC2"],
    marker="o",
    label="PC1 + PC2"
)

plt.plot(
    pca_df["layer"],
    pca_df["PC5"],
    marker="o",
    label="PC1–5"
)

plt.plot(
    pca_df["layer"],
    pca_df["PC10"],
    marker="o",
    label="PC1–10"
)

plt.xlabel("Layer")
plt.ylabel("Cumulative variance explained")
plt.title(
    "Dimensionality of A → C displacement"
)

plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "06_PCA_variance_by_layer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 7:
# PCA COMPONENTS REQUIRED
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    pca_df["layer"],
    pca_df["n_components_50pct"],
    marker="o",
    label="50%"
)

plt.plot(
    pca_df["layer"],
    pca_df["n_components_80pct"],
    marker="o",
    label="80%"
)

plt.plot(
    pca_df["layer"],
    pca_df["n_components_90pct"],
    marker="o",
    label="90%"
)

plt.xlabel("Layer")
plt.ylabel("Number of components")
plt.title(
    "Dimensionality of competition-induced displacement"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG / "07_PCA_components_required.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 8:
# DISTRIBUTION OF DIRECTIONAL CONSISTENCY
# ============================================================

for layer in selected_layers:

    values = direction_df[
        direction_df["layer"] == layer
    ]

    if len(values) == 0:
        continue

    alignment = alignment_by_layer[layer]

    plt.figure(figsize=(7, 5))

    plt.hist(
        alignment,
        bins=25
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Cosine to mean A → C displacement"
    )

    plt.ylabel("Number of flipped prompts")

    plt.title(
        f"Displacement-direction consistency — Layer {layer}"
    )

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        FIG / f"08_alignment_distribution_L{layer}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 9:
# A AND C STATES PROJECTED ONTO PCA OF DELTA
# ============================================================

for layer in selected_layers:

    if layer not in pca_details:
        continue

    details = pca_details[layer]

    Vt = details["Vt"]

    # First two PCs of the displacement space
    components = Vt[:2]

    A_layer = A_flip[:, layer, :]
    C_layer = C_flip[:, layer, :]

    # Center states using the same center used for PCA
    delta_center = np.mean(
        C_layer - A_layer,
        axis=0
    )

    A_centered = A_layer - delta_center
    C_centered = C_layer - delta_center

    A_proj = A_centered @ components.T
    C_proj = C_centered @ components.T

    plt.figure(figsize=(7, 7))

    # Draw paired A -> C trajectories
    for i in range(n_flips):

        plt.plot(
            [A_proj[i, 0], C_proj[i, 0]],
            [A_proj[i, 1], C_proj[i, 1]],
            linewidth=0.5,
            alpha=0.12
        )

    plt.scatter(
        A_proj[:, 0],
        A_proj[:, 1],
        s=12,
        alpha=0.35,
        label="Condition A"
    )

    plt.scatter(
        C_proj[:, 0],
        C_proj[:, 1],
        s=12,
        alpha=0.35,
        label="Condition C"
    )

    plt.xlabel("PC1 of A → C displacement")
    plt.ylabel("PC2 of A → C displacement")

    plt.title(
        f"A → C representation trajectories — Layer {layer}\n"
        "163 flipped prompts"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        FIG / f"09_A_C_PCA_trajectories_L{layer}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "same_prompt_A_to_C_flip_representation_analysis",

    "seed": 42,

    "n_flips": int(n_flips),

    "n_layers": int(n_layers),

    "hidden_dimension": int(hidden_dim),

    "activation_type": "prefill_acts",

    "definition":
        "A refusal -> C compliance",

    "files": {
        "A": str(A_FILE),
        "C": str(C_FILE),
    },

    "selected_layers_for_distribution":
        selected_layers,
}

with open(
    OUT / "summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("KEY RESULTS")
print("=" * 80)

print("\nLargest L2 displacement:")
top_l2 = layer_df.sort_values(
    "l2_mean",
    ascending=False
).head(5)

print(
    top_l2[
        ["layer", "l2_mean", "relative_l2_mean"]
    ].to_string(index=False)
)

print("\nStrongest common-direction consistency:")
top_align = direction_df.sort_values(
    "alignment_mean",
    ascending=False
).head(5)

print(
    top_align[
        [
            "layer",
            "alignment_mean",
            "alignment_positive_fraction",
            "alignment_above_0.5_fraction",
        ]
    ].to_string(index=False)
)

print("\nMost low-dimensional layers:")
top_pca = pca_df.sort_values(
    "n_components_90pct"
).head(5)

print(
    top_pca[
        [
            "layer",
            "PC1",
            "PC2",
            "PC5",
            "PC10",
            "n_components_90pct",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

print(f"Results: {OUT}")
print(f"Figures: {FIG}")

print("\nDone.")
