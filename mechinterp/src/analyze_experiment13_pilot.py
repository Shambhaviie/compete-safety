"""
Analyze Experiment 13 pilot trajectories.

NO MODEL IS LOADED.
NO GENERATION IS PERFORMED.

This script analyzes the already-generated Experiment 13 data.

Scientific comparison
----------------------

Natural competition trajectory:

    A  ->  C
    h_C - h_A

Causal ablation trajectory:

    A  ->  A_ablation
    h_A_ablation - h_A

For every layer we ask:

    How similar are these two changes?

Primary metric:
    cosine(
        h_C - h_A,
        h_A_ablation - h_A
    )

Secondary metrics:
    - norm of natural competition shift
    - norm of ablation-induced shift
    - state similarity between C and A_ablation
    - flip vs stable comparison

IMPORTANT
---------
The intervention layer itself contains the pre-intervention activation
in the saved trajectory. Therefore, interpretation of propagation begins
at intervention_layer + 1.

This analysis is descriptive for the 20-prompt pilot.
It does NOT establish statistical significance or a causal mechanism.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

ROOT = Path("/home/Shambhavi/compete_safety_v2")

ACT_DIR = ROOT / "mechinterp" / "activations"
EXP_DIR = ROOT / "mechinterp" / "results" / "experiment13"

A_FILE = ACT_DIR / "activations_prehook_A_seed42.npz"
C_FILE = ACT_DIR / "activations_prehook_C_seed42.npz"

A_META_FILE = ACT_DIR / "metadata_prehook_A_seed42.json"
C_META_FILE = ACT_DIR / "metadata_prehook_C_seed42.json"

SELECTED_FILE = EXP_DIR / "selected_prompts.json"

TRAJECTORY_FILE = EXP_DIR / "trajectory_comparisons.npz"
RESULTS_FILE = EXP_DIR / "results.json"

OUT_DIR = EXP_DIR / "pilot_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONSTANTS
# ================================================================

N_LAYERS = 32
D_MODEL = 4096

INTERVENTION_LAYERS = [7, 9, 11, 14, 18, 24]


# ================================================================
# HELPERS
# ================================================================

def cosine_rows(a, b):
    """
    Row-wise cosine similarity.

    a, b:
        [N, D]

    returns:
        [N]
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    dot = np.sum(a * b, axis=1)

    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)

    denom = na * nb

    result = np.full(
        len(a),
        np.nan,
        dtype=np.float64,
    )

    valid = denom > 1e-12

    result[valid] = (
        dot[valid] / denom[valid]
    )

    return result


def row_norms(x):
    return np.linalg.norm(
        np.asarray(x, dtype=np.float64),
        axis=1,
    )


def mean_sem(x):
    """
    Descriptive mean and SEM.

    SEM is included for visualization only.
    It is NOT being used here as a formal significance test.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    mean = np.mean(x)

    if len(x) == 1:
        return mean, np.nan

    sem = np.std(
        x,
        ddof=1,
    ) / np.sqrt(len(x))

    return mean, sem


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 72, flush=True)
print("EXPERIMENT 13 — PILOT TRAJECTORY ANALYSIS", flush=True)
print("=" * 72, flush=True)

print("\nLoading selected prompts...", flush=True)

with open(SELECTED_FILE) as f:
    selected = json.load(f)

print(
    f"Selected prompts: {len(selected)}",
    flush=True,
)

groups = np.array(
    [
        x["group"]
        for x in selected
    ]
)

flip_mask = groups == "flip"
stable_mask = groups == "stable"

print(
    f"Flip: {flip_mask.sum()}",
    flush=True,
)

print(
    f"Stable: {stable_mask.sum()}",
    flush=True,
)


# ================================================================
# LOAD NATURAL A/C ACTIVATIONS
# ================================================================

print("\nLoading natural A/C activations...", flush=True)

data_A = np.load(
    A_FILE
)

data_C = np.load(
    C_FILE
)

acts_A_all = data_A[
    "prefill_acts"
].astype(np.float32)

acts_C_all = data_C[
    "prefill_acts"
].astype(np.float32)

print(
    f"A shape: {acts_A_all.shape}",
    flush=True,
)

print(
    f"C shape: {acts_C_all.shape}",
    flush=True,
)


# ================================================================
# BUILD PAIRED A/C ARRAYS
# ================================================================

A = np.stack(
    [
        acts_A_all[
            x["A_index"]
        ]
        for x in selected
    ],
    axis=0,
)

C = np.stack(
    [
        acts_C_all[
            x["C_index"]
        ]
        for x in selected
    ],
    axis=0,
)

print(
    f"Selected A shape: {A.shape}",
    flush=True,
)

print(
    f"Selected C shape: {C.shape}",
    flush=True,
)


# ================================================================
# NATURAL COMPETITION CHANGE
# ================================================================

natural_delta = C - A

print(
    "\nNatural A -> C trajectory constructed.",
    flush=True,
)


# ================================================================
# LOAD ABLATION TRAJECTORIES
# ================================================================

print(
    "\nLoading intervention trajectories...",
    flush=True,
)

trajectory_data = np.load(
    TRAJECTORY_FILE,
    allow_pickle=True,
)

ablation_trajectories = {}

for layer in INTERVENTION_LAYERS:

    key = f"A_L{layer}_ablation"

    if key not in trajectory_data.files:
        raise RuntimeError(
            f"Missing trajectory key: {key}"
        )

    x = trajectory_data[key].astype(
        np.float32
    )

    expected = (
        len(selected),
        N_LAYERS,
        D_MODEL,
    )

    if x.shape != expected:
        raise RuntimeError(
            f"{key}: expected {expected}, "
            f"got {x.shape}"
        )

    ablation_trajectories[layer] = x

    print(
        f"  L{layer}: {x.shape}",
        flush=True,
    )


# ================================================================
# MAIN ANALYSIS
# ================================================================

all_rows = []

print(
    "\nComputing trajectory comparisons...",
    flush=True,
)

for intervention_layer in INTERVENTION_LAYERS:

    print(
        f"\nAnalyzing intervention at L{intervention_layer}",
        flush=True,
    )

    ablated = ablation_trajectories[
        intervention_layer
    ]

    # ------------------------------------------------------------
    # A -> ablation
    # ------------------------------------------------------------

    intervention_delta = (
        ablated - A
    )

    for layer in range(N_LAYERS):

        natural = natural_delta[
            :,
            layer,
            :,
        ]

        intervention = intervention_delta[
            :,
            layer,
            :,
        ]

        # --------------------------------------------------------
        # Directional similarity
        # --------------------------------------------------------

        cosine = cosine_rows(
            natural,
            intervention,
        )

        # --------------------------------------------------------
        # Magnitudes
        # --------------------------------------------------------

        natural_norm = row_norms(
            natural
        )

        intervention_norm = row_norms(
            intervention
        )

        # --------------------------------------------------------
        # Resulting-state similarity
        # --------------------------------------------------------

        state_cosine = cosine_rows(
            C[:, layer, :],
            ablated[:, layer, :],
        )

        # --------------------------------------------------------
        # Flip/stable rows
        # --------------------------------------------------------

        for i in range(
            len(selected)
        ):

            all_rows.append(
                {
                    "intervention_layer":
                        intervention_layer,

                    "layer":
                        layer,

                    "prompt_id":
                        selected[i][
                            "prompt_id"
                        ],

                    "group":
                        selected[i][
                            "group"
                        ],

                    "is_downstream":
                        layer >
                        intervention_layer,

                    "natural_shift_norm":
                        float(
                            natural_norm[i]
                        ),

                    "ablation_shift_norm":
                        float(
                            intervention_norm[i]
                        ),

                    "shift_cosine":
                        float(
                            cosine[i]
                        )
                        if np.isfinite(
                            cosine[i]
                        )
                        else None,

                    "state_cosine_C_vs_ablation":
                        float(
                            state_cosine[i]
                        )
                        if np.isfinite(
                            state_cosine[i]
                        )
                        else None,
                }
            )


df = pd.DataFrame(
    all_rows
)


# ================================================================
# SAVE RAW PER-PROMPT RESULTS
# ================================================================

csv_file = (
    OUT_DIR
    / "per_prompt_trajectory_metrics.csv"
)

df.to_csv(
    csv_file,
    index=False,
)

print(
    f"\nSaved: {csv_file}",
    flush=True,
)


# ================================================================
# DOWNSTREAM SUMMARY
# ================================================================

downstream_df = df[
    df["is_downstream"]
].copy()


summary_rows = []

for intervention_layer in (
    INTERVENTION_LAYERS
):

    for layer in range(
        intervention_layer + 1,
        N_LAYERS,
    ):

        subset = downstream_df[
            (
                downstream_df[
                    "intervention_layer"
                ]
                == intervention_layer
            )
            &
            (
                downstream_df[
                    "layer"
                ]
                == layer
            )
        ]

        for group in [
            "flip",
            "stable",
        ]:

            g = subset[
                subset["group"] == group
            ]

            cosine_mean, cosine_sem = (
                mean_sem(
                    g["shift_cosine"]
                    .values
                )
            )

            natural_mean, natural_sem = (
                mean_sem(
                    g["natural_shift_norm"]
                    .values
                )
            )

            ablation_mean, ablation_sem = (
                mean_sem(
                    g["ablation_shift_norm"]
                    .values
                )
            )

            state_mean, state_sem = (
                mean_sem(
                    g[
                        "state_cosine_C_vs_ablation"
                    ].values
                )
            )

            summary_rows.append(
                {
                    "intervention_layer":
                        intervention_layer,

                    "layer":
                        layer,

                    "group":
                        group,

                    "n":
                        len(g),

                    "shift_cosine_mean":
                        cosine_mean,

                    "shift_cosine_sem":
                        cosine_sem,

                    "natural_shift_norm_mean":
                        natural_mean,

                    "natural_shift_norm_sem":
                        natural_sem,

                    "ablation_shift_norm_mean":
                        ablation_mean,

                    "ablation_shift_norm_sem":
                        ablation_sem,

                    "state_cosine_mean":
                        state_mean,

                    "state_cosine_sem":
                        state_sem,
                }
            )


summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUT_DIR
    / "downstream_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

print(
    f"Saved: {summary_file}",
    flush=True,
)


# ================================================================
# FLIP - STABLE DIFFERENCE
# ================================================================

difference_rows = []

for intervention_layer in (
    INTERVENTION_LAYERS
):

    for layer in range(
        intervention_layer + 1,
        N_LAYERS,
    ):

        subset = summary_df[
            (
                summary_df[
                    "intervention_layer"
                ]
                == intervention_layer
            )
            &
            (
                summary_df[
                    "layer"
                ]
                == layer
            )
        ]

        flip = subset[
            subset["group"] == "flip"
        ]

        stable = subset[
            subset["group"] == "stable"
        ]

        if len(flip) != 1 or len(stable) != 1:
            continue

        flip_mean = float(
            flip[
                "shift_cosine_mean"
            ].iloc[0]
        )

        stable_mean = float(
            stable[
                "shift_cosine_mean"
            ].iloc[0]
        )

        difference_rows.append(
            {
                "intervention_layer":
                    intervention_layer,

                "layer":
                    layer,

                "flip_cosine":
                    flip_mean,

                "stable_cosine":
                    stable_mean,

                "flip_minus_stable":
                    flip_mean
                    - stable_mean,
            }
        )


difference_df = pd.DataFrame(
    difference_rows
)

difference_file = (
    OUT_DIR
    / "flip_vs_stable_cosine_difference.csv"
)

difference_df.to_csv(
    difference_file,
    index=False,
)

print(
    f"Saved: {difference_file}",
    flush=True,
)


# ================================================================
# PRINT KEY TABLE
# ================================================================

print(
    "\n"
    + "=" * 72,
    flush=True,
)

print(
    "KEY PILOT RESULT — DOWNSTREAM COSINE",
    flush=True,
)

print(
    "=" * 72,
    flush=True,
)

for intervention_layer in (
    INTERVENTION_LAYERS
):

    # Focus on final layer as a compact summary.
    final = summary_df[
        (
            summary_df[
                "intervention_layer"
            ]
            == intervention_layer
        )
        &
        (
            summary_df["layer"]
            == N_LAYERS - 1
        )
    ]

    if len(final) == 0:
        continue

    flip = final[
        final["group"] == "flip"
    ]

    stable = final[
        final["group"] == "stable"
    ]

    if len(flip) == 1 and len(stable) == 1:

        fc = float(
            flip[
                "shift_cosine_mean"
            ].iloc[0]
        )

        sc = float(
            stable[
                "shift_cosine_mean"
            ].iloc[0]
        )

        print(
            f"L{intervention_layer:>2} "
            f"-> L31 | "
            f"flip={fc:.4f} | "
            f"stable={sc:.4f} | "
            f"diff={fc-sc:+.4f}",
            flush=True,
        )


# ================================================================
# PLOTS
# ================================================================

print(
    "\nGenerating plots...",
    flush=True,
)

for intervention_layer in (
    INTERVENTION_LAYERS
):

    subset = summary_df[
        summary_df[
            "intervention_layer"
        ]
        == intervention_layer
    ]

    if len(subset) == 0:
        continue

    plt.figure(
        figsize=(9, 5)
    )

    for group in [
        "flip",
        "stable",
    ]:

        g = subset[
            subset["group"] == group
        ].sort_values("layer")

        plt.plot(
            g["layer"],
            g["shift_cosine_mean"],
            marker="o",
            label=group,
        )

        # SEM envelope
        lower = (
            g["shift_cosine_mean"]
            - g["shift_cosine_sem"]
        )

        upper = (
            g["shift_cosine_mean"]
            + g["shift_cosine_sem"]
        )

        plt.fill_between(
            g["layer"],
            lower,
            upper,
            alpha=0.15,
        )

    plt.axvline(
        intervention_layer,
        linestyle="--",
        label=(
            f"intervention L"
            f"{intervention_layer}"
        ),
    )

    plt.axhline(
        0,
        linestyle=":",
    )

    plt.xlabel(
        "Layer"
    )

    plt.ylabel(
        "Cosine similarity\n"
        "(natural A→C shift vs ablation A→Aabl shift)"
    )

    plt.title(
        f"Experiment 13 pilot — "
        f"intervention at L{intervention_layer}"
    )

    plt.legend()

    plt.tight_layout()

    outfile = (
        OUT_DIR
        / f"cosine_trajectory_L"
        f"{intervention_layer}.png"
    )

    plt.savefig(
        outfile,
        dpi=200,
    )

    plt.close()

    print(
        f"  Saved {outfile.name}",
        flush=True,
    )


# ================================================================
# FLIP-STABLE DIFFERENCE PLOT
# ================================================================

for intervention_layer in (
    INTERVENTION_LAYERS
):

    subset = difference_df[
        difference_df[
            "intervention_layer"
        ]
        == intervention_layer
    ].sort_values("layer")

    if len(subset) == 0:
        continue

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        subset["layer"],
        subset["flip_minus_stable"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle=":",
    )

    plt.axvline(
        intervention_layer,
        linestyle="--",
    )

    plt.xlabel(
        "Layer"
    )

    plt.ylabel(
        "Flip cosine − Stable cosine"
    )

    plt.title(
        f"Flip vs stable trajectory difference — "
        f"intervention at L{intervention_layer}"
    )

    plt.tight_layout()

    outfile = (
        OUT_DIR
        / f"flip_minus_stable_L"
        f"{intervention_layer}.png"
    )

    plt.savefig(
        outfile,
        dpi=200,
    )

    plt.close()


# ================================================================
# FINAL SUMMARY JSON
# ================================================================

final_summary = {
    "experiment":
        "Experiment 13 pilot trajectory analysis",

    "n_total":
        int(len(selected)),

    "n_flip":
        int(flip_mask.sum()),

    "n_stable":
        int(stable_mask.sum()),

    "intervention_layers":
        INTERVENTION_LAYERS,

    "primary_metric":
        "cosine between natural A->C shift and A->ablation shift",

    "secondary_metrics": [
        "natural shift norm",
        "ablation shift norm",
        "C-state vs ablation-state cosine",
    ],

    "important_interpretation_note":
        (
            "This 20-prompt analysis is exploratory. "
            "Positive cosine indicates directional alignment "
            "between the two state changes; it does not by itself "
            "establish a shared mechanism or causal equivalence."
        ),

    "downstream_definition":
        "layers strictly greater than intervention layer",

    "files": {
        "per_prompt":
            str(csv_file),

        "summary":
            str(summary_file),

        "flip_stable_difference":
            str(difference_file),
    },
}

summary_json = (
    OUT_DIR
    / "analysis_summary.json"
)

with open(
    summary_json,
    "w",
) as f:
    json.dump(
        final_summary,
        f,
        indent=2,
    )

print(
    f"\nSaved: {summary_json}",
    flush=True,
)

print(
    "\n"
    + "=" * 72,
    flush=True,
)

print(
    "PILOT ANALYSIS COMPLETE",
    flush=True,
)

print(
    f"Output directory: {OUT_DIR}",
    flush=True,
)

print(
    "=" * 72,
    flush=True,
)
