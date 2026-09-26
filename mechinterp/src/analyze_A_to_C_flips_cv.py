import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ============================================================
# CONFIG
# ============================================================

BASE = "mechinterp"

A_ACT_PATH = os.path.join(
    BASE, "activations", "activations_prehook_A_seed42.npz"
)

C_ACT_PATH = os.path.join(
    BASE, "activations", "activations_prehook_C_seed42.npz"
)

A_META_PATH = os.path.join(
    BASE, "activations", "metadata_prehook_A_seed42.json"
)

C_META_PATH = os.path.join(
    BASE, "activations", "metadata_prehook_C_seed42.json"
)

OUT_DIR = os.path.join(
    BASE, "results", "A_vs_C_flips_cv"
)

FIG_DIR = os.path.join(OUT_DIR, "figures")

N_SPLITS = 5
SEED = 42


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity_matrix(X, direction):
    """
    Cosine similarity between every row of X and one direction.

    X:         [N, D]
    direction: [D]
    """
    X_norm = np.linalg.norm(X, axis=1)
    d_norm = np.linalg.norm(direction)

    denom = X_norm * d_norm

    result = np.full(X.shape[0], np.nan, dtype=np.float64)

    valid = denom > 0
    result[valid] = (
        X[valid] @ direction
    ) / denom[valid]

    return result


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "frac_gt_0": np.nan,
            "frac_gt_05": np.nan,
            "frac_gt_08": np.nan,
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "frac_gt_0": float(np.mean(values > 0.0)),
        "frac_gt_05": float(np.mean(values > 0.5)),
        "frac_gt_08": float(np.mean(values > 0.8)),
    }


# ============================================================
# LOAD DATA
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 80)
print("5-FOLD CROSS-VALIDATED A -> C DISPLACEMENT ANALYSIS")
print("=" * 80)

print("\nLoading activations...")

A_npz = np.load(A_ACT_PATH)
C_npz = np.load(C_ACT_PATH)

print("A keys:", A_npz.files)
print("C keys:", C_npz.files)

# We deliberately use PREFILL activations because this matches
# the previous A-vs-C flip analysis.
A = A_npz["prefill_acts"].astype(np.float64)
C = C_npz["prefill_acts"].astype(np.float64)

print("A shape:", A.shape)
print("C shape:", C.shape)

assert A.shape == C.shape

with open(A_META_PATH, "r") as f:
    A_meta = json.load(f)

with open(C_META_PATH, "r") as f:
    C_meta = json.load(f)

print("A metadata entries:", len(A_meta))
print("C metadata entries:", len(C_meta))


# ============================================================
# ALIGN PROMPTS
# ============================================================

def metadata_pid(item):
    return str(item["prompt_id"])


A_ids = [metadata_pid(x) for x in A_meta]
C_ids = [metadata_pid(x) for x in C_meta]

assert A_ids == C_ids, "A and C prompt ordering differs."

# Prefer metadata labels, but verify against NPZ labels if present.
if "replayed_labels" in A_npz:
    A_labels = A_npz["replayed_labels"]
elif "historical_labels" in A_npz:
    A_labels = A_npz["historical_labels"]
else:
    A_labels = None

if "replayed_labels" in C_npz:
    C_labels = C_npz["replayed_labels"]
elif "historical_labels" in C_npz:
    C_labels = C_npz["historical_labels"]
else:
    C_labels = None

# A and C labels are expected to differ because the experiment
# specifically studies behavioral changes between conditions.
# Do NOT assert equality between them.


# ============================================================
# IDENTIFY THE 163 A -> C FLIPS
# ============================================================

# Label convention used in the existing experiment:
# 0 = refusal
# 1 = compliance
#
# We also verify this using metadata where possible.

if A_labels is not None:
    # Actual label convention:
    #   1 = refusal
    #   0 = compliance
    #
    # We want A refusal -> C compliance.
    A_refusal = (A_labels == 1)
    C_compliance = (C_labels == 0)

    flip_mask = A_refusal & C_compliance
else:
    raise RuntimeError(
        "Could not find replayed_labels/historical_labels in activation files."
    )

flip_indices = np.where(flip_mask)[0]

print("\nNumber of A -> C flips:", len(flip_indices))

assert len(flip_indices) == 163, (
    f"Expected 163 A->C flips, found {len(flip_indices)}"
)

flip_ids = [A_ids[i] for i in flip_indices]

print("First 10 flip prompt IDs:")
for pid in flip_ids[:10]:
    print(" ", pid)


# ============================================================
# EXTRACT FLIP DISPLACEMENTS
# ============================================================

# A and C:
#   [N, layers, hidden_dim]
#
# For each prompt and layer:
#
#   Delta_i,l = C_i,l - A_i,l
#
# Result:
#   [163, 32, 4096]

A_flip = A[flip_indices]
C_flip = C[flip_indices]

D = C_flip - A_flip

N, N_LAYERS, D_MODEL = D.shape

print("\nFlip displacement shape:", D.shape)


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print("\nRunning 5-fold cross-validation...")
print("Each fold: 80% train / 20% held-out test.")
print("The displacement direction is estimated ONLY from training prompts.")
print("Held-out prompts are never used to estimate their test direction.")

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

# Store held-out scores for every prompt/layer.
#
# Shape:
#   [163, 32]
#
# Each prompt gets exactly one held-out score at each layer.
cv_test_cosines = np.full(
    (N, N_LAYERS),
    np.nan,
    dtype=np.float64
)

# Store training scores as well, mainly to quantify
# train-vs-test optimism.
cv_train_cosines = np.full(
    (N, N_LAYERS),
    np.nan,
    dtype=np.float64
)

# Store the normalized direction estimated from each training fold.
#
# [fold, layer, hidden_dim]
cv_directions = np.zeros(
    (N_SPLITS, N_LAYERS, D_MODEL),
    dtype=np.float64
)

# Fold assignment for every prompt.
fold_assignment = np.full(N, -1, dtype=int)

fold_summaries = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):

    print(f"\nFold {fold_idx + 1}/{N_SPLITS}")
    print("  Train:", len(train_idx))
    print("  Test :", len(test_idx))

    fold_assignment[test_idx] = fold_idx

    for layer in range(N_LAYERS):

        train_delta = D[train_idx, layer, :]
        test_delta = D[test_idx, layer, :]

        # ----------------------------------------------------
        # Estimate common displacement direction ONLY on train
        # ----------------------------------------------------

        mean_train_delta = np.mean(
            train_delta,
            axis=0,
            dtype=np.float64
        )

        norm = np.linalg.norm(mean_train_delta)

        if norm == 0:
            continue

        direction = mean_train_delta / norm

        cv_directions[
            fold_idx, layer, :
        ] = direction

        # ----------------------------------------------------
        # Evaluate on TRAIN
        # ----------------------------------------------------

        train_cos = cosine_similarity_matrix(
            train_delta,
            direction
        )

        cv_train_cosines[
            train_idx, layer
        ] = train_cos

        # ----------------------------------------------------
        # Evaluate on HELD-OUT TEST
        # ----------------------------------------------------

        test_cos = cosine_similarity_matrix(
            test_delta,
            direction
        )

        cv_test_cosines[
            test_idx, layer
        ] = test_cos


# ============================================================
# SUMMARIZE CROSS-VALIDATED RESULTS
# ============================================================

rows = []

for layer in range(N_LAYERS):

    train_summary = summarize(
        cv_train_cosines[:, layer]
    )

    test_summary = summarize(
        cv_test_cosines[:, layer]
    )

    rows.append({
        "layer": layer,

        # Train
        "train_mean_cosine": train_summary["mean"],
        "train_std_cosine": train_summary["std"],
        "train_median_cosine": train_summary["median"],

        # Held-out
        "test_mean_cosine": test_summary["mean"],
        "test_std_cosine": test_summary["std"],
        "test_median_cosine": test_summary["median"],
        "test_q25_cosine": test_summary["q25"],
        "test_q75_cosine": test_summary["q75"],

        # Held-out fractions
        "test_fraction_cosine_gt_0":
            test_summary["frac_gt_0"],

        "test_fraction_cosine_gt_0.5":
            test_summary["frac_gt_05"],

        "test_fraction_cosine_gt_0.8":
            test_summary["frac_gt_08"],

        # Train/test difference
        "train_minus_test_mean":
            train_summary["mean"] - test_summary["mean"],
    })


summary_df = pd.DataFrame(rows)

summary_path = os.path.join(
    OUT_DIR,
    "cv_displacement_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PER-PROMPT RESULTS
# ============================================================

prompt_rows = []

for i in range(N):

    for layer in range(N_LAYERS):

        prompt_rows.append({
            "prompt_index": int(i),
            "prompt_id": flip_ids[i],
            "fold": int(fold_assignment[i]) + 1,

            "layer": layer,

            "heldout_cosine": float(
                cv_test_cosines[i, layer]
            ),

            "train_cosine": float(
                cv_train_cosines[i, layer]
            ),
        })


prompt_df = pd.DataFrame(prompt_rows)

prompt_path = os.path.join(
    OUT_DIR,
    "cv_prompt_level_results.csv"
)

prompt_df.to_csv(
    prompt_path,
    index=False
)


# ============================================================
# SAVE CROSS-VALIDATED DIRECTIONS
# ============================================================

direction_path = os.path.join(
    OUT_DIR,
    "cv_train_directions.npy"
)

np.save(
    direction_path,
    cv_directions
)


# ============================================================
# SAVE FOLD ASSIGNMENTS
# ============================================================

fold_path = os.path.join(
    OUT_DIR,
    "fold_assignments.npy"
)

np.save(
    fold_path,
    fold_assignment
)


# ============================================================
# PLOTS
# ============================================================

import matplotlib.pyplot as plt


layers = summary_df["layer"].to_numpy()


# ------------------------------------------------------------
# Figure 1: Held-out mean cosine
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    layers,
    summary_df["test_mean_cosine"],
    marker="o",
    label="Held-out mean cosine"
)

plt.plot(
    layers,
    summary_df["train_mean_cosine"],
    marker="o",
    linestyle="--",
    label="Training mean cosine"
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Layer")
plt.ylabel("Cosine similarity")
plt.title(
    "Cross-validated consistency of A → C displacement direction\n"
    "163 refusal → compliance flips"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

fig1 = os.path.join(
    FIG_DIR,
    "01_cv_mean_cosine.png"
)

plt.savefig(fig1, dpi=200)
plt.close()


# ------------------------------------------------------------
# Figure 2: Held-out fractions above thresholds
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    layers,
    summary_df["test_fraction_cosine_gt_0"],
    marker="o",
    label="Cosine > 0"
)

plt.plot(
    layers,
    summary_df["test_fraction_cosine_gt_0.5"],
    marker="o",
    label="Cosine > 0.5"
)

plt.plot(
    layers,
    summary_df["test_fraction_cosine_gt_0.8"],
    marker="o",
    label="Cosine > 0.8"
)

plt.xlabel("Layer")
plt.ylabel("Fraction of held-out prompts")
plt.ylim(-0.02, 1.05)

plt.title(
    "Cross-validated alignment with learned A → C displacement\n"
    "Held-out prompts only"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

fig2 = os.path.join(
    FIG_DIR,
    "02_cv_alignment_fractions.png"
)

plt.savefig(fig2, dpi=200)
plt.close()


# ------------------------------------------------------------
# Figure 3: Train-test gap
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

plt.plot(
    layers,
    summary_df["train_minus_test_mean"],
    marker="o"
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Layer")
plt.ylabel("Train − held-out mean cosine")

plt.title(
    "Cross-validation generalization gap\n"
    "A → C displacement direction"
)

plt.grid(alpha=0.3)
plt.tight_layout()

fig3 = os.path.join(
    FIG_DIR,
    "03_cv_generalization_gap.png"
)

plt.savefig(fig3, dpi=200)
plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("CROSS-VALIDATED RESULTS")
print("=" * 80)

print(
    summary_df[
        [
            "layer",
            "train_mean_cosine",
            "test_mean_cosine",
            "test_fraction_cosine_gt_0",
            "test_fraction_cosine_gt_0.5",
            "test_fraction_cosine_gt_0.8",
            "train_minus_test_mean",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary_json = {
    "analysis": "5-fold cross-validated A->C displacement direction",
    "seed": SEED,
    "n_flips": int(N),
    "n_layers": int(N_LAYERS),
    "hidden_dim": int(D_MODEL),
    "n_splits": N_SPLITS,
    "activation_type": "prefill_acts",
    "displacement_definition": "C - A",
    "direction_estimation": "mean displacement on training prompts",
    "evaluation": "cosine similarity on held-out prompts",
    "files": {
        "summary_csv": summary_path,
        "prompt_level_csv": prompt_path,
        "directions_npy": direction_path,
        "fold_assignments_npy": fold_path,
        "figure_mean_cosine": fig1,
        "figure_alignment_fractions": fig2,
        "figure_generalization_gap": fig3,
    },
}

json_path = os.path.join(
    OUT_DIR,
    "summary.json"
)

with open(json_path, "w") as f:
    json.dump(
        summary_json,
        f,
        indent=2
    )


print("\n")
print("=" * 80)
print("FILES WRITTEN")
print("=" * 80)

print("Results :", OUT_DIR)
print("Figures :", FIG_DIR)
print()
print(summary_path)
print(prompt_path)
print(direction_path)
print(fold_path)
print(json_path)

print("\nDone.")
