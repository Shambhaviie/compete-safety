#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIG
# ============================================================

BASE = Path("mechinterp/activations")
RESULTS = Path("mechinterp/results/A_vs_C_same_prompt")
RESULTS.mkdir(parents=True, exist_ok=True)

A_FILE = BASE / "activations_prehook_A_seed42.npz"
C_FILE = BASE / "activations_prehook_C_seed42.npz"

A_META = BASE / "metadata_prehook_A_seed42.json"
C_META = BASE / "metadata_prehook_C_seed42.json"


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a, b):
    numerator = np.sum(a * b, axis=1)
    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
    )
    return numerator / np.maximum(denominator, 1e-12)


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy)
        / (nx + ny - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("SAME-PROMPT CONDITION A vs C HIDDEN-STATE ANALYSIS")
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

assert A_ids == C_ids, "A/C prompt ordering does not match!"
assert A_acts.shape == C_acts.shape, "A/C activation shapes differ!"

n_prompts, n_layers, hidden_dim = A_acts.shape

print(f"\nPrompts: {n_prompts}")
print(f"Layers: {n_layers}")
print(f"Hidden dimension: {hidden_dim}")

flip_mask = A_labels & ~C_labels
stable_mask = A_labels & C_labels

print(f"\nA refusal -> C compliance: {flip_mask.sum()}")
print(f"A refusal -> C refusal:    {stable_mask.sum()}")
print(f"A compliance:              {(~A_labels).sum()}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

rows = []

all_cosines = np.zeros(
    (n_prompts, n_layers),
    dtype=np.float32
)

all_l2 = np.zeros(
    (n_prompts, n_layers),
    dtype=np.float32
)

all_relative_l2 = np.zeros(
    (n_prompts, n_layers),
    dtype=np.float32
)


for layer in range(n_layers):

    A_layer = A_acts[:, layer, :]
    C_layer = C_acts[:, layer, :]

    # Same-prompt cosine similarity
    cosine = cosine_similarity(A_layer, C_layer)

    # A -> C difference vector
    delta = C_layer - A_layer

    # Absolute magnitude of change
    l2 = np.linalg.norm(delta, axis=1)

    # Change relative to A representation magnitude
    A_norm = np.linalg.norm(A_layer, axis=1)

    relative_l2 = l2 / np.maximum(A_norm, 1e-12)

    all_cosines[:, layer] = cosine
    all_l2[:, layer] = l2
    all_relative_l2[:, layer] = relative_l2

    # --------------------------------------------------------
    # GROUPS
    # --------------------------------------------------------

    flip_cos = cosine[flip_mask]
    stable_cos = cosine[stable_mask]

    flip_l2 = l2[flip_mask]
    stable_l2 = l2[stable_mask]

    flip_rel = relative_l2[flip_mask]
    stable_rel = relative_l2[stable_mask]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    u_cos, p_cos = mannwhitneyu(
        flip_cos,
        stable_cos,
        alternative="two-sided"
    )

    u_l2, p_l2 = mannwhitneyu(
        flip_l2,
        stable_l2,
        alternative="two-sided"
    )

    u_rel, p_rel = mannwhitneyu(
        flip_rel,
        stable_rel,
        alternative="two-sided"
    )

    rows.append({
        "layer": layer,

        "all_cosine_mean": float(np.mean(cosine)),
        "all_cosine_sd": float(np.std(cosine, ddof=1)),

        "flip_n": int(len(flip_cos)),
        "flip_cosine_mean": float(np.mean(flip_cos)),
        "flip_cosine_sd": float(np.std(flip_cos, ddof=1)),

        "flip_l2_mean": float(np.mean(flip_l2)),
        "flip_l2_sd": float(np.std(flip_l2, ddof=1)),

        "flip_relative_l2_mean": float(np.mean(flip_rel)),
        "flip_relative_l2_sd": float(np.std(flip_rel, ddof=1)),

        "stable_n": int(len(stable_cos)),
        "stable_cosine_mean": float(np.mean(stable_cos)),
        "stable_cosine_sd": float(np.std(stable_cos, ddof=1)),

        "stable_l2_mean": float(np.mean(stable_l2)),
        "stable_l2_sd": float(np.std(stable_l2, ddof=1)),

        "stable_relative_l2_mean": float(np.mean(stable_rel)),
        "stable_relative_l2_sd": float(np.std(stable_rel, ddof=1)),

        "cosine_difference_flip_minus_stable":
            float(np.mean(flip_cos) - np.mean(stable_cos)),

        "l2_difference_flip_minus_stable":
            float(np.mean(flip_l2) - np.mean(stable_l2)),

        "relative_l2_difference_flip_minus_stable":
            float(np.mean(flip_rel) - np.mean(stable_rel)),

        "cohens_d_cosine":
            float(cohens_d(flip_cos, stable_cos)),

        "cohens_d_l2":
            float(cohens_d(flip_l2, stable_l2)),

        "cohens_d_relative_l2":
            float(cohens_d(flip_rel, stable_rel)),

        "mannwhitney_u_cosine": float(u_cos),
        "mannwhitney_p_cosine": float(p_cos),

        "mannwhitney_u_l2": float(u_l2),
        "mannwhitney_p_l2": float(p_l2),

        "mannwhitney_u_relative_l2": float(u_rel),
        "mannwhitney_p_relative_l2": float(p_rel),
    })


results_df = pd.DataFrame(rows)


# ============================================================
# SAVE LAYER SUMMARY
# ============================================================

csv_path = RESULTS / "layer_summary.csv"
results_df.to_csv(csv_path, index=False)


# ============================================================
# SAVE PROMPT-LEVEL RESULTS
# ============================================================

prompt_rows = []

for i, pid in enumerate(A_ids):

    if flip_mask[i]:
        group = "flip_A_refusal_to_C_compliance"
    elif stable_mask[i]:
        group = "stable_refusal"
    else:
        group = "A_compliance"

    for layer in range(n_layers):

        prompt_rows.append({
            "prompt_id": pid,
            "group": group,
            "layer": layer,
            "cosine_A_C": float(all_cosines[i, layer]),
            "l2_A_C": float(all_l2[i, layer]),
            "relative_l2_A_C":
                float(all_relative_l2[i, layer]),
        })

prompt_df = pd.DataFrame(prompt_rows)

prompt_csv = RESULTS / "prompt_level_results.csv"
prompt_df.to_csv(prompt_csv, index=False)


# ============================================================
# PLOT 1: COSINE
# ============================================================

layers = np.arange(n_layers)

plt.figure(figsize=(8, 5))

plt.plot(
    layers,
    results_df["flip_cosine_mean"],
    marker="o",
    label="A→C flips"
)

plt.plot(
    layers,
    results_df["stable_cosine_mean"],
    marker="o",
    label="Stable refusals"
)

plt.xlabel("Layer")
plt.ylabel("Cosine similarity: A vs C")
plt.title(
    "Same-prompt hidden-state similarity: Condition A vs C"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

cosine_plot = RESULTS / "A_vs_C_cosine_by_layer.png"
plt.savefig(cosine_plot, dpi=300)
plt.close()


# ============================================================
# PLOT 2: L2 DISTANCE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    layers,
    results_df["flip_l2_mean"],
    marker="o",
    label="A→C flips"
)

plt.plot(
    layers,
    results_df["stable_l2_mean"],
    marker="o",
    label="Stable refusals"
)

plt.xlabel("Layer")
plt.ylabel("L2 distance: C − A")
plt.title(
    "Same-prompt activation change: Condition A → C"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

l2_plot = RESULTS / "A_vs_C_L2_by_layer.png"
plt.savefig(l2_plot, dpi=300)
plt.close()


# ============================================================
# PLOT 3: EFFECT SIZE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    layers,
    results_df["cohens_d_cosine"],
    marker="o",
    label="Cosine similarity"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Layer")
plt.ylabel("Cohen's d")
plt.title(
    "Flip vs stable difference in A→C representation change"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

effect_plot = RESULTS / "A_vs_C_effect_size.png"
plt.savefig(effect_plot, dpi=300)
plt.close()


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary = {
    "analysis":
        "same_prompt_A_vs_C_hidden_state_comparison",
    "seed": 42,
    "activation_type": "prefill_acts",
    "n_prompts": int(n_prompts),
    "n_layers": int(n_layers),
    "hidden_dim": int(hidden_dim),
    "n_flip": int(flip_mask.sum()),
    "n_stable_refusal": int(stable_mask.sum()),
    "n_A_compliance": int((~A_labels).sum()),
    "files": {
        "A": str(A_FILE),
        "C": str(C_FILE),
    },
}

with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(
    "\nLayer | Flip cosine | Stable cosine | "
    "Flip L2 | Stable L2 | d(cos) | p(cos)"
)

print("-" * 80)

for _, r in results_df.iterrows():

    print(
        f"{int(r['layer']):5d} | "
        f"{r['flip_cosine_mean']:.4f}      | "
        f"{r['stable_cosine_mean']:.4f}        | "
        f"{r['flip_l2_mean']:.3f}   | "
        f"{r['stable_l2_mean']:.3f}      | "
        f"{r['cohens_d_cosine']:.3f}  | "
        f"{r['mannwhitney_p_cosine']:.3e}"
    )


print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

print(csv_path)
print(prompt_csv)
print(cosine_plot)
print(l2_plot)
print(effect_plot)
print(RESULTS / "summary.json")

print("\nDone.")
