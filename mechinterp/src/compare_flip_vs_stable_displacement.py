#!/usr/bin/env python3

"""
COMPARE FLIP VS STABLE A->C DISPLACEMENT

Question:
    Does the A->C displacement direction learned from the 163
    refusal->compliance flips also describe the 512 prompts that
    remain refusals?

Design:
    - 676 prompts total
    - 163 A->C flips: A refusal -> C compliance
    - 512 stable refusals: A refusal -> C refusal
    - 1 A-compliance prompt is excluded
    - 5-fold cross-fitting over the 163 flip prompts
    - At each layer:
        * learn mean A->C displacement direction on 4/5 flip prompts
        * evaluate held-out flip prompts
        * evaluate ALL 512 stable prompts
    - Compare flip vs stable alignment distributions

Important:
    The displacement direction is estimated ONLY from flip prompts.
    Stable prompts are never used to estimate the direction.

Outputs:
    mechinterp/results/A_vs_C_flip_vs_stable/
        flip_vs_stable_summary.csv
        prompt_level_alignment.csv
        layer_statistics.csv
        summary.json
        figures/
"""

import os
import json
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

A_PATH = "mechinterp/activations/activations_prehook_A_seed42.npz"
C_PATH = "mechinterp/activations/activations_prehook_C_seed42.npz"

A_META_PATH = "mechinterp/activations/metadata_prehook_A_seed42.json"
C_META_PATH = "mechinterp/activations/metadata_prehook_C_seed42.json"

OUT_DIR = "mechinterp/results/A_vs_C_flip_vs_stable"
FIG_DIR = os.path.join(OUT_DIR, "figures")

N_SPLITS = 5
RANDOM_SEED = 42

# Number of bootstrap samples for confidence intervals.
N_BOOTSTRAP = 2000

# Layers shown in distribution plots.
SELECTED_LAYERS = [7, 9, 11, 14, 18, 24, 31]


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def load_npz(path):
    data = np.load(path, allow_pickle=True)

    print(f"\nLoaded: {path}")
    print("Keys:", list(data.keys()))

    for key in data.keys():
        arr = data[key]
        if hasattr(arr, "shape"):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    return data


def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)


def get_labels(npz_data, metadata):
    """
    Robustly retrieve behavioural labels.

    Expected convention from current project:
        1 = refusal
        0 = compliance

    We prefer replayed_labels if available.
    """

    possible_keys = [
        "replayed_labels",
        "historical_labels",
        "labels",
    ]

    labels = None

    for key in possible_keys:
        if key in npz_data:
            labels = np.asarray(npz_data[key]).astype(int)
            print(f"Using labels from NPZ key: {key}")
            break

    if labels is None:
        # Try metadata.
        candidate_fields = [
            "replayed_label",
            "historical_label",
            "label",
            "safety_label",
            "refusal_label",
        ]

        extracted = []

        for item in metadata:
            value = None
            for field in candidate_fields:
                if field in item:
                    value = item[field]
                    break

            if value is None:
                extracted = []
                break

            extracted.append(value)

        if extracted:
            labels = np.asarray(extracted).astype(int)
            print("Using labels extracted from metadata.")
        else:
            raise RuntimeError(
                "Could not find behavioural labels in NPZ or metadata."
            )

    unique = np.unique(labels)

    print("Unique labels:", unique)

    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Expected binary labels 0/1, found {unique}"
        )

    return labels


def get_prompt_ids(metadata):
    ids = []

    for item in metadata:
        if "prompt_id" in item:
            ids.append(str(item["prompt_id"]))
        elif "id" in item:
            ids.append(str(item["id"]))
        else:
            raise RuntimeError(
                "Could not find prompt_id/id in metadata."
            )

    return np.asarray(ids)


def normalize(v, eps=1e-12):
    norm = np.linalg.norm(v)

    if norm < eps:
        return None

    return v / norm


def cosine_rows(X, direction):
    """
    Cosine similarity between every row of X and direction.
    """

    direction_norm = np.linalg.norm(direction)

    if direction_norm < 1e-12:
        return np.full(X.shape[0], np.nan)

    X_norms = np.linalg.norm(X, axis=1)

    denom = X_norms * direction_norm

    numer = X @ direction

    out = np.full(X.shape[0], np.nan)

    valid = denom > 1e-12

    out[valid] = numer[valid] / denom[valid]

    return out


def cohens_d(x, y):
    """
    Cohen's d using pooled standard deviation.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)

    pooled = np.sqrt(
        ((nx - 1) * sx**2 + (ny - 1) * sy**2)
        / (nx + ny - 2)
    )

    if pooled < 1e-12:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled


def bootstrap_mean_difference(x, y, rng, n_bootstrap=2000):
    """
    Bootstrap CI for:

        mean(x) - mean(y)

    Resamples each group independently.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan

    diffs = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)

        diffs[b] = np.mean(xb) - np.mean(yb)

    observed = np.mean(x) - np.mean(y)

    lower, upper = np.percentile(diffs, [2.5, 97.5])

    return observed, lower, upper


# =============================================================================
# MAIN
# =============================================================================

def main():

    ensure_dirs()

    print("=" * 88)
    print("FLIP VS STABLE A->C DISPLACEMENT COMPARISON")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    A = load_npz(A_PATH)
    C = load_npz(C_PATH)

    with open(A_META_PATH, "r") as f:
        A_meta = json.load(f)

    with open(C_META_PATH, "r") as f:
        C_meta = json.load(f)

    A_ids = get_prompt_ids(A_meta)
    C_ids = get_prompt_ids(C_meta)

    if len(A_ids) != len(C_ids):
        raise ValueError("A and C prompt counts differ.")

    if not np.array_equal(A_ids, C_ids):
        raise ValueError(
            "A and C prompt ordering / prompt IDs do not match."
        )

    A_labels = get_labels(A, A_meta)
    C_labels = get_labels(C, C_meta)

    if len(A_labels) != len(C_labels):
        raise ValueError("A/C label lengths differ.")

    # -------------------------------------------------------------------------
    # Activation arrays
    # -------------------------------------------------------------------------

    if "prefill_acts" not in A or "prefill_acts" not in C:
        raise RuntimeError(
            "Expected prefill_acts in both activation files."
        )

    A_acts = np.asarray(A["prefill_acts"], dtype=np.float32)
    C_acts = np.asarray(C["prefill_acts"], dtype=np.float32)

    if A_acts.shape != C_acts.shape:
        raise ValueError(
            f"A/C activation shapes differ: "
            f"{A_acts.shape} vs {C_acts.shape}"
        )

    n_prompts, n_layers, hidden_dim = A_acts.shape

    print("\n" + "=" * 88)
    print("DATASET")
    print("=" * 88)

    print(f"Prompts: {n_prompts}")
    print(f"Layers: {n_layers}")
    print(f"Hidden dimension: {hidden_dim}")

    # -------------------------------------------------------------------------
    # Define groups
    #
    # Current project convention:
    #   1 = refusal
    #   0 = compliance
    #
    # Flip:
    #   A refusal -> C compliance
    #
    # Stable:
    #   A refusal -> C refusal
    # -------------------------------------------------------------------------

    flip_mask = (A_labels == 1) & (C_labels == 0)
    stable_mask = (A_labels == 1) & (C_labels == 1)

    flip_idx = np.where(flip_mask)[0]
    stable_idx = np.where(stable_mask)[0]

    other_idx = np.where(~(flip_mask | stable_mask))[0]

    print("\nGroup counts:")
    print(f"  A refusal -> C compliance flips: {len(flip_idx)}")
    print(f"  A refusal -> C refusal stable:   {len(stable_idx)}")
    print(f"  Other cases:                      {len(other_idx)}")

    if len(flip_idx) != 163:
        print(
            f"\nWARNING: expected 163 flips, found {len(flip_idx)}."
        )

    if len(stable_idx) != 512:
        print(
            f"WARNING: expected 512 stable refusals, found {len(stable_idx)}."
        )

    if len(flip_idx) < N_SPLITS:
        raise RuntimeError("Not enough flip prompts for 5-fold CV.")

    # -------------------------------------------------------------------------
    # Precompute A->C displacements
    # -------------------------------------------------------------------------

    print("\nComputing A->C displacement...")

    displacement = C_acts - A_acts

    # Shape:
    #   [prompts, layers, hidden_dim]

    # -------------------------------------------------------------------------
    # Cross-fitting
    # -------------------------------------------------------------------------

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_assignments = np.full(len(flip_idx), -1, dtype=int)

    # Results:
    #
    # flip_alignment[prompt, layer]
    # stable_alignment[prompt, layer]
    #
    flip_alignment = np.full(
        (len(flip_idx), n_layers),
        np.nan,
        dtype=np.float32,
    )

    stable_alignment = np.full(
        (len(stable_idx), n_layers),
        np.nan,
        dtype=np.float32,
    )

    # For diagnostics:
    direction_norms = np.full(n_layers, np.nan)

    fold_counter = 0

    for train_pos, test_pos in kfold.split(flip_idx):

        train_global = flip_idx[train_pos]
        test_global = flip_idx[test_pos]

        fold_assignments[test_pos] = fold_counter

        print(
            f"\nFold {fold_counter + 1}/{N_SPLITS}: "
            f"train={len(train_global)}, test={len(test_global)}"
        )

        for layer in range(n_layers):

            # -------------------------------------------------------------
            # Estimate direction ONLY from training flips
            # -------------------------------------------------------------

            train_delta = displacement[
                train_global,
                layer,
                :
            ]

            mean_direction = np.mean(train_delta, axis=0)

            direction_norm = np.linalg.norm(mean_direction)

            direction_norms[layer] = direction_norm

            if direction_norm < 1e-12:
                continue

            direction = mean_direction / direction_norm

            # -------------------------------------------------------------
            # Held-out flip alignment
            # -------------------------------------------------------------

            test_delta = displacement[
                test_global,
                layer,
                :
            ]

            test_cos = cosine_rows(
                test_delta,
                direction,
            )

            # Store according to fold test positions.
            flip_alignment[test_pos, layer] = test_cos

            # -------------------------------------------------------------
            # Stable-prompt alignment
            #
            # Stable prompts were NOT used to estimate direction.
            # -------------------------------------------------------------

            stable_delta = displacement[
                stable_idx,
                layer,
                :
            ]

            stable_cos = cosine_rows(
                stable_delta,
                direction,
            )

            # We have one stable estimate per fold.
            # We average the fold-specific values below.
            if fold_counter == 0:
                stable_alignment[:, layer] = stable_cos
            else:
                # Running average across folds.
                previous = stable_alignment[:, layer]

                stable_alignment[:, layer] = (
                    previous * fold_counter + stable_cos
                ) / (fold_counter + 1)

        fold_counter += 1

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for layer in range(n_layers):

        flip_vals = flip_alignment[:, layer]
        stable_vals = stable_alignment[:, layer]

        flip_vals = flip_vals[np.isfinite(flip_vals)]
        stable_vals = stable_vals[np.isfinite(stable_vals)]

        if len(flip_vals) == 0 or len(stable_vals) == 0:
            continue

        flip_mean = np.mean(flip_vals)
        stable_mean = np.mean(stable_vals)

        flip_median = np.median(flip_vals)
        stable_median = np.median(stable_vals)

        flip_std = np.std(flip_vals, ddof=1)
        stable_std = np.std(stable_vals, ddof=1)

        # Welch t-test.
        t_stat, t_p = ttest_ind(
            flip_vals,
            stable_vals,
            equal_var=False,
        )

        # Mann-Whitney U.
        u_stat, mw_p = mannwhitneyu(
            flip_vals,
            stable_vals,
            alternative="two-sided",
        )

        d = cohens_d(
            flip_vals,
            stable_vals,
        )

        diff, ci_low, ci_high = bootstrap_mean_difference(
            flip_vals,
            stable_vals,
            rng,
            n_bootstrap=N_BOOTSTRAP,
        )

        # AUC:
        # Can alignment itself discriminate flips from stable prompts?
        y = np.concatenate([
            np.ones(len(flip_vals)),
            np.zeros(len(stable_vals)),
        ])

        scores = np.concatenate([
            flip_vals,
            stable_vals,
        ])

        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = np.nan

        rows.append({
            "layer": layer,
            "flip_n": len(flip_vals),
            "stable_n": len(stable_vals),

            "flip_mean_cosine": flip_mean,
            "stable_mean_cosine": stable_mean,

            "flip_median_cosine": flip_median,
            "stable_median_cosine": stable_median,

            "flip_sd": flip_std,
            "stable_sd": stable_std,

            "mean_difference_flip_minus_stable": diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,

            "cohens_d": d,

            "welch_t": t_stat,
            "welch_p": t_p,

            "mannwhitney_u": u_stat,
            "mannwhitney_p": mw_p,

            "alignment_auc": auc,

            "flip_fraction_cos_gt_0":
                np.mean(flip_vals > 0),

            "stable_fraction_cos_gt_0":
                np.mean(stable_vals > 0),

            "flip_fraction_cos_gt_05":
                np.mean(flip_vals > 0.5),

            "stable_fraction_cos_gt_05":
                np.mean(stable_vals > 0.5),

            "flip_fraction_cos_gt_08":
                np.mean(flip_vals > 0.8),

            "stable_fraction_cos_gt_08":
                np.mean(stable_vals > 0.8),
        })

    stats_df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Multiple-comparison correction: Benjamini-Hochberg
    # -------------------------------------------------------------------------

    def bh_fdr(p_values):

        p = np.asarray(p_values, dtype=float)

        q = np.full_like(p, np.nan)

        valid = np.isfinite(p)

        if not np.any(valid):
            return q

        pv = p[valid]

        order = np.argsort(pv)
        ranked = pv[order]

        m = len(ranked)

        adjusted = ranked * m / np.arange(1, m + 1)

        adjusted = np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]

        adjusted = np.minimum(adjusted, 1.0)

        result = np.empty(m)
        result[order] = adjusted

        q[valid] = result

        return q

    stats_df["mannwhitney_q_bh"] = bh_fdr(
        stats_df["mannwhitney_p"].values
    )

    stats_df["welch_q_bh"] = bh_fdr(
        stats_df["welch_p"].values
    )

    # -------------------------------------------------------------------------
    # Prompt-level CSV
    # -------------------------------------------------------------------------

    prompt_rows = []

    for pos, global_idx in enumerate(flip_idx):

        for layer in range(n_layers):

            prompt_rows.append({
                "prompt_index": int(global_idx),
                "prompt_id": str(A_ids[global_idx]),
                "group": "flip",
                "layer": layer,
                "cv_fold": int(fold_assignments[pos]),
                "alignment_cosine": float(
                    flip_alignment[pos, layer]
                ),
            })

    for pos, global_idx in enumerate(stable_idx):

        for layer in range(n_layers):

            prompt_rows.append({
                "prompt_index": int(global_idx),
                "prompt_id": str(A_ids[global_idx]),
                "group": "stable",
                "layer": layer,
                "cv_fold": -1,
                "alignment_cosine": float(
                    stable_alignment[pos, layer]
                ),
            })

    prompt_df = pd.DataFrame(prompt_rows)

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    stats_path = os.path.join(
        OUT_DIR,
        "layer_statistics.csv",
    )

    prompt_path = os.path.join(
        OUT_DIR,
        "prompt_level_alignment.csv",
    )

    stats_df.to_csv(stats_path, index=False)
    prompt_df.to_csv(prompt_path, index=False)

    # -------------------------------------------------------------------------
    # Figure 1:
    # Mean alignment across layers
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    layers = stats_df["layer"].values

    plt.plot(
        layers,
        stats_df["flip_mean_cosine"].values,
        marker="o",
        label="A→C flips (cross-validated)",
    )

    plt.plot(
        layers,
        stats_df["stable_mean_cosine"].values,
        marker="o",
        label="Stable refusals",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Layer")
    plt.ylabel(
        "Cosine alignment with flip-derived A→C displacement direction"
    )

    plt.title(
        "Flip vs stable alignment with the competition-induced displacement"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig1_path = os.path.join(
        FIG_DIR,
        "01_flip_vs_stable_alignment.png",
    )

    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2:
    # Difference in mean alignment with bootstrap CI
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    diff = stats_df["mean_difference_flip_minus_stable"].values
    low = stats_df["bootstrap_ci_low"].values
    high = stats_df["bootstrap_ci_high"].values

    plt.plot(
        layers,
        diff,
        marker="o",
    )

    plt.fill_between(
        layers,
        low,
        high,
        alpha=0.2,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Layer")
    plt.ylabel(
        "Mean cosine difference (flip − stable)"
    )

    plt.title(
        "Flip–stable difference in displacement alignment"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig2_path = os.path.join(
        FIG_DIR,
        "02_alignment_difference_bootstrap_ci.png",
    )

    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3:
    # Cohen's d
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        layers,
        stats_df["cohens_d"].values,
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Layer")
    plt.ylabel("Cohen's d (flip − stable)")

    plt.title(
        "Effect size: flip vs stable displacement alignment"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig3_path = os.path.join(
        FIG_DIR,
        "03_cohens_d.png",
    )

    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 4:
    # Selected-layer distributions
    # -------------------------------------------------------------------------

    for layer in SELECTED_LAYERS:

        if layer >= n_layers:
            continue

        flip_vals = flip_alignment[:, layer]
        stable_vals = stable_alignment[:, layer]

        flip_vals = flip_vals[np.isfinite(flip_vals)]
        stable_vals = stable_vals[np.isfinite(stable_vals)]

        plt.figure(figsize=(8, 6))

        plt.hist(
            flip_vals,
            bins=25,
            alpha=0.55,
            density=True,
            label="A→C flips",
        )

        plt.hist(
            stable_vals,
            bins=25,
            alpha=0.55,
            density=True,
            label="Stable refusals",
        )

        plt.axvline(
            np.mean(flip_vals),
            linestyle="--",
            linewidth=2,
            label=f"Flip mean = {np.mean(flip_vals):.3f}",
        )

        plt.axvline(
            np.mean(stable_vals),
            linestyle="--",
            linewidth=2,
            label=f"Stable mean = {np.mean(stable_vals):.3f}",
        )

        plt.xlabel(
            "Cosine alignment with flip-derived A→C direction"
        )

        plt.ylabel("Density")

        plt.title(
            f"Flip vs stable displacement alignment — L{layer}"
        )

        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()

        path = os.path.join(
            FIG_DIR,
            f"04_alignment_distribution_L{layer}.png",
        )

        plt.savefig(path, dpi=200)
        plt.close()

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    strongest_difference = stats_df.iloc[
        np.argmax(
            np.abs(
                stats_df["mean_difference_flip_minus_stable"].values
            )
        )
    ]

    strongest_effect = stats_df.iloc[
        np.argmax(
            np.abs(
                stats_df["cohens_d"].values
            )
        )
    ]

    summary = {
        "analysis": "flip_vs_stable_A_to_C_displacement",
        "model": "Llama 3.1 8B Instruct",
        "seed": 42,

        "n_total": int(n_prompts),
        "n_flip": int(len(flip_idx)),
        "n_stable": int(len(stable_idx)),
        "n_other": int(len(other_idx)),

        "n_layers": int(n_layers),
        "hidden_dimension": int(hidden_dim),

        "cross_validation": {
            "n_splits": N_SPLITS,
            "seed": RANDOM_SEED,
            "description": (
                "A->C displacement direction estimated only from "
                "training flip prompts and evaluated on held-out "
                "flip prompts and stable refusals."
            ),
        },

        "strongest_mean_difference_layer": int(
            strongest_difference["layer"]
        ),

        "strongest_mean_difference": float(
            strongest_difference[
                "mean_difference_flip_minus_stable"
            ]
        ),

        "strongest_effect_size_layer": int(
            strongest_effect["layer"]
        ),

        "strongest_cohens_d": float(
            strongest_effect["cohens_d"]
        ),

        "layer_statistics": stats_df.to_dict(
            orient="records"
        ),
    }

    summary_path = os.path.join(
        OUT_DIR,
        "summary.json",
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Print concise results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("KEY RESULTS")
    print("=" * 88)

    display_cols = [
        "layer",
        "flip_mean_cosine",
        "stable_mean_cosine",
        "mean_difference_flip_minus_stable",
        "cohens_d",
        "mannwhitney_p",
        "mannwhitney_q_bh",
        "alignment_auc",
    ]

    print(
        stats_df[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 88)
    print("FILES WRITTEN")
    print("=" * 88)

    print(f"Results: {OUT_DIR}")
    print(f"Figures: {FIG_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()
