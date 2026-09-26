"""
EXPERIMENT 13 — FULL TRAJECTORY ANALYSIS

Reads:
    mechinterp/results/experiment13_full/

and computes descriptive trajectory and behavioural analyses.

IMPORTANT:
This script is descriptive/diagnostic.
It does not make causal claims beyond the intervention itself.

Primary comparison:

    Flip:
        A = refusal
        C = compliance

    Stable:
        A = refusal
        C = refusal

For each intervention layer, compare the downstream trajectory
following directional ablation with the natural A -> C trajectory.

Outputs:
    experiment13_full/analysis/
        behavioural_summary.csv
        trajectory_summary.csv
        downstream_summary.csv
        flip_vs_stable_effects.csv
        per_prompt_metrics.csv
        analysis_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PATHS
# ================================================================

ROOT = Path(
    "/home/Shambhavi/compete_safety_v2"
)

RESULT_DIR = (
    ROOT
    / "mechinterp/results/experiment13_full"
)

ANALYSIS_DIR = (
    RESULT_DIR / "analysis"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# PARAMETERS
# ================================================================

N_LAYERS = 32
D_MODEL = 4096

INTERVENTION_LAYERS = [
    7,
    9,
    11,
    14,
    18,
    24,
]

# Layers strictly downstream of intervention.
# Intervention layer itself is excluded.
DOWNSTREAM_START_OFFSET = 1


# ================================================================
# BASIC HELPERS
# ================================================================

def cosine(a, b):
    """
    Cosine similarity between vectors.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    return float(
        np.dot(a, b) / (na * nb)
    )


def norm(x):
    return float(np.linalg.norm(x))


def mean_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    return (
        float(np.mean(x)),
        float(np.std(x, ddof=1))
        if len(x) > 1
        else 0.0,
    )


def cohens_d(x, y):
    """
    Standardized difference:
        mean(x) - mean(y)

    Pooled SD.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = np.sqrt(
        (
            (nx - 1) * vx
            + (ny - 1) * vy
        )
        / (nx + ny - 2)
    )

    if pooled == 0:
        return np.nan

    return float(
        (np.mean(x) - np.mean(y))
        / pooled
    )


# ================================================================
# LOAD SELECTION
# ================================================================

selection_file = (
    RESULT_DIR
    / "selected_prompts.json"
)

with open(selection_file) as f:
    selection = json.load(f)

selected = selection[
    "selected_prompts"
]

print("=" * 72)
print("EXPERIMENT 13 — FULL TRAJECTORY ANALYSIS")
print("=" * 72)

print(
    f"Selected prompts: {len(selected)}"
)

print(
    "Flip:",
    sum(
        x["group"] == "flip"
        for x in selected
    )
)

print(
    "Stable:",
    sum(
        x["group"] == "stable"
        for x in selected
    )
)


# ================================================================
# LOAD NATURAL ACTIVATIONS
# ================================================================

print("\nLoading natural A/C activations...")

A = np.load(
    ROOT
    / "mechinterp/activations/"
    / "activations_prehook_A_seed42.npz"
)

C = np.load(
    ROOT
    / "mechinterp/activations/"
    / "activations_prehook_C_seed42.npz"
)

acts_A = A[
    "prefill_acts"
].astype(np.float32)

acts_C = C[
    "prefill_acts"
].astype(np.float32)

print(
    "A:",
    acts_A.shape
)

print(
    "C:",
    acts_C.shape
)


# ================================================================
# MAP NATURAL ACTIVATIONS
# ================================================================

meta_A = json.load(
    open(
        ROOT
        / "mechinterp/activations/"
        / "metadata_prehook_A_seed42.json"
    )
)

meta_C = json.load(
    open(
        ROOT
        / "mechinterp/activations/"
        / "metadata_prehook_C_seed42.json"
    )
)

pid_A = {
    m["prompt_id"]: i
    for i, m in enumerate(meta_A)
}

pid_C = {
    m["prompt_id"]: i
    for i, m in enumerate(meta_C)
}


# ================================================================
# NATURAL A -> C TRAJECTORY
# ================================================================

print(
    "\nConstructing natural A -> C trajectories..."
)

natural_A = []
natural_C = []
groups = []
prompt_ids = []

for item in selected:

    pid = item["prompt_id"]

    ia = pid_A[pid]
    ic = pid_C[pid]

    natural_A.append(
        acts_A[ia]
    )

    natural_C.append(
        acts_C[ic]
    )

    groups.append(
        item["group"]
    )

    prompt_ids.append(
        pid
    )

natural_A = np.stack(
    natural_A
)

natural_C = np.stack(
    natural_C
)

groups = np.asarray(
    groups
)

prompt_ids = np.asarray(
    prompt_ids
)

natural_shift = (
    natural_C
    - natural_A
)

print(
    "Natural trajectory:",
    natural_shift.shape
)


# ================================================================
# LOAD INTERVENTION TRAJECTORIES
# ================================================================

intervention_data = {}

for layer in INTERVENTION_LAYERS:

    path = (
        RESULT_DIR
        / f"trajectory_L{layer}_ablation.npz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing trajectory file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    )

    trajectories = data[
        "trajectories"
    ].astype(np.float32)

    file_pids = data[
        "prompt_ids"
    ]

    # ------------------------------------------------------------
    # Verify prompt ordering.
    # ------------------------------------------------------------

    if not np.array_equal(
        file_pids.astype(str),
        prompt_ids.astype(str),
    ):

        raise RuntimeError(
            f"Prompt ordering mismatch "
            f"for L{layer}."
        )

    intervention_data[layer] = {
        "trajectory":
            trajectories,

        "responses":
            data["responses"],

        "safety":
            data["safety_labels"],
    }

    print(
        f"L{layer}: {trajectories.shape}"
    )


# ================================================================
# BEHAVIOURAL SUMMARY
# ================================================================

print(
    "\nComputing behavioural summary..."
)

behaviour_rows = []

for layer in INTERVENTION_LAYERS:

    safety = (
        intervention_data[layer][
            "safety"
        ]
    )

    for group in [
        "flip",
        "stable",
    ]:

        mask = (
            groups == group
        )

        group_safety = [
            bool(x)
            for x in safety[mask]
        ]

        n = len(
            group_safety
        )

        n_safe = sum(
            group_safety
        )

        # In the original experiment:
        # safe=True means refusal.

        refusal_rate = (
            n_safe / n
            if n
            else np.nan
        )

        if group == "flip":

            # Natural C outcome = compliance.
            baseline_refusal = 0.0

        else:

            # Natural C outcome = refusal.
            baseline_refusal = 1.0

        behavioural_flip = (
            sum(
                x != baseline_refusal
                for x in group_safety
            )
        )

        behaviour_rows.append(
            {
                "layer": layer,
                "group": group,
                "n": n,
                "refusals_after_ablation":
                    n_safe,
                "compliances_after_ablation":
                    n - n_safe,
                "refusal_rate_after_ablation":
                    refusal_rate,
                "baseline_C_refusal_rate":
                    baseline_refusal,
                "behavioural_changes":
                    behavioural_flip,
            }
        )

behaviour_df = pd.DataFrame(
    behaviour_rows
)

behaviour_path = (
    ANALYSIS_DIR
    / "behavioural_summary.csv"
)

behaviour_df.to_csv(
    behaviour_path,
    index=False,
)

print(
    "Saved:",
    behaviour_path
)


# ================================================================
# TRAJECTORY METRICS
# ================================================================

print(
    "\nComputing trajectory metrics..."
)

rows = []

for layer in INTERVENTION_LAYERS:

    intervention = (
        intervention_data[layer][
            "trajectory"
        ]
    )

    intervention_idx = layer

    for i in range(
        len(selected)
    ):

        group = groups[i]

        # --------------------------------------------------------
        # Ablated trajectory relative to natural C.
        #
        # At each layer:
        #
        # intervention_state - natural_C_state
        # --------------------------------------------------------

        ablated_minus_C = (
            intervention[i]
            - natural_C[i]
        )

        # --------------------------------------------------------
        # Natural competition shift:
        #
        # C - A
        # --------------------------------------------------------

        A_to_C = natural_shift[i]

        # --------------------------------------------------------
        # Downstream layers only.
        # --------------------------------------------------------

        downstream_indices = np.arange(
            intervention_idx + 1,
            N_LAYERS,
        )

        if len(
            downstream_indices
        ) == 0:

            downstream_cos = np.nan
            downstream_norm = np.nan

        else:

            cosine_values = []

            norm_values = []

            for j in downstream_indices:

                # Direction of natural
                # competition-induced change.
                natural_vec = (
                    A_to_C[j]
                )

                # Direction of deviation caused
                # by ablation relative to C.
                ablation_vec = (
                    ablated_minus_C[j]
                )

                cosine_values.append(
                    cosine(
                        ablation_vec,
                        natural_vec,
                    )
                )

                norm_values.append(
                    norm(
                        ablation_vec
                    )
                )

            downstream_cos = (
                np.nanmean(
                    cosine_values
                )
            )

            downstream_norm = (
                np.nanmean(
                    norm_values
                )
            )

        # --------------------------------------------------------
        # Endpoint similarity at L31.
        # --------------------------------------------------------

        endpoint_ablation = (
            ablated_minus_C[
                -1
            ]
        )

        endpoint_natural = (
            A_to_C[-1]
        )

        endpoint_cos = cosine(
            endpoint_ablation,
            endpoint_natural,
        )

        endpoint_norm = norm(
            endpoint_ablation
        )

        # --------------------------------------------------------
        # Full downstream vector similarity.
        # --------------------------------------------------------

        if len(
            downstream_indices
        ) > 0:

            ablation_flat = (
                ablated_minus_C[
                    downstream_indices
                ].reshape(-1)
            )

            natural_flat = (
                A_to_C[
                    downstream_indices
                ].reshape(-1)
            )

            full_downstream_cos = (
                cosine(
                    ablation_flat,
                    natural_flat,
                )
            )

        else:

            full_downstream_cos = np.nan

        rows.append(
            {
                "prompt_id":
                    prompt_ids[i],

                "group":
                    group,

                "intervention_layer":
                    layer,

                "downstream_mean_cosine":
                    downstream_cos,

                "downstream_mean_norm":
                    downstream_norm,

                "L31_cosine":
                    endpoint_cos,

                "L31_norm":
                    endpoint_norm,

                "full_downstream_cosine":
                    full_downstream_cos,
            }
        )


metrics_df = pd.DataFrame(
    rows
)

metrics_path = (
    ANALYSIS_DIR
    / "per_prompt_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)

print(
    "Saved:",
    metrics_path
)


# ================================================================
# TRAJECTORY SUMMARY
# ================================================================

summary_rows = []

for layer in INTERVENTION_LAYERS:

    sub = metrics_df[
        metrics_df[
            "intervention_layer"
        ] == layer
    ]

    for group in [
        "flip",
        "stable",
    ]:

        g = sub[
            sub["group"] == group
        ]

        summary_rows.append(
            {
                "layer":
                    layer,

                "group":
                    group,

                "n":
                    len(g),

                "downstream_cosine_mean":
                    g[
                        "downstream_mean_cosine"
                    ].mean(),

                "downstream_cosine_std":
                    g[
                        "downstream_mean_cosine"
                    ].std(),

                "downstream_cosine_median":
                    g[
                        "downstream_mean_cosine"
                    ].median(),

                "downstream_norm_mean":
                    g[
                        "downstream_mean_norm"
                    ].mean(),

                "L31_cosine_mean":
                    g[
                        "L31_cosine"
                    ].mean(),

                "L31_cosine_std":
                    g[
                        "L31_cosine"
                    ].std(),

                "L31_cosine_median":
                    g[
                        "L31_cosine"
                    ].median(),

                "full_downstream_cosine_mean":
                    g[
                        "full_downstream_cosine"
                    ].mean(),
            }
        )

trajectory_summary_df = pd.DataFrame(
    summary_rows
)

trajectory_summary_path = (
    ANALYSIS_DIR
    / "trajectory_summary.csv"
)

trajectory_summary_df.to_csv(
    trajectory_summary_path,
    index=False,
)

print(
    "Saved:",
    trajectory_summary_path
)


# ================================================================
# FLIP VS STABLE EFFECT SIZES
# ================================================================

effect_rows = []

for layer in INTERVENTION_LAYERS:

    sub = metrics_df[
        metrics_df[
            "intervention_layer"
        ] == layer
    ]

    flip = sub[
        sub["group"] == "flip"
    ]

    stable = sub[
        sub["group"] == "stable"
    ]

    for metric in [
        "downstream_mean_cosine",
        "downstream_mean_norm",
        "L31_cosine",
        "L31_norm",
        "full_downstream_cosine",
    ]:

        x = flip[metric].values
        y = stable[metric].values

        mx, sx = mean_std(x)
        my, sy = mean_std(y)

        effect_rows.append(
            {
                "layer":
                    layer,

                "metric":
                    metric,

                "flip_n":
                    len(x),

                "stable_n":
                    len(y),

                "flip_mean":
                    mx,

                "flip_std":
                    sx,

                "stable_mean":
                    my,

                "stable_std":
                    sy,

                "flip_minus_stable":
                    mx - my,

                "cohens_d":
                    cohens_d(
                        x,
                        y,
                    ),
            }
        )

effects_df = pd.DataFrame(
    effect_rows
)

effects_path = (
    ANALYSIS_DIR
    / "flip_vs_stable_effects.csv"
)

effects_df.to_csv(
    effects_path,
    index=False,
)

print(
    "Saved:",
    effects_path
)


# ================================================================
# COMBINED LAYER SUMMARY
# ================================================================

combined_rows = []

for layer in INTERVENTION_LAYERS:

    b = behaviour_df[
        behaviour_df["layer"]
        == layer
    ]

    e = effects_df[
        (effects_df["layer"] == layer)
        &
        (
            effects_df["metric"]
            == "downstream_mean_cosine"
        )
    ].iloc[0]

    flip_b = b[
        b["group"] == "flip"
    ].iloc[0]

    stable_b = b[
        b["group"] == "stable"
    ].iloc[0]

    combined_rows.append(
        {
            "layer":
                layer,

            "flip_ablation_refusal_pct":
                100
                * flip_b[
                    "refusal_rate_after_ablation"
                ],

            "stable_ablation_refusal_pct":
                100
                * stable_b[
                    "refusal_rate_after_ablation"
                ],

            "flip_behavioural_changes":
                flip_b[
                    "behavioural_changes"
                ],

            "stable_behavioural_changes":
                stable_b[
                    "behavioural_changes"
                ],

            "flip_downstream_cosine":
                e["flip_mean"],

            "stable_downstream_cosine":
                e["stable_mean"],

            "cosine_difference":
                e[
                    "flip_minus_stable"
                ],

            "cosine_cohens_d":
                e["cohens_d"],
        }
    )

combined_df = pd.DataFrame(
    combined_rows
)

combined_path = (
    ANALYSIS_DIR
    / "layer_summary.csv"
)

combined_df.to_csv(
    combined_path,
    index=False,
)

print(
    "Saved:",
    combined_path
)


# ================================================================
# JSON SUMMARY
# ================================================================

summary = {
    "n_prompts": int(
        len(selected)
    ),

    "n_flip": int(
        sum(
            x["group"] == "flip"
            for x in selected
        )
    ),

    "n_stable": int(
        sum(
            x["group"] == "stable"
            for x in selected
        )
    ),

    "intervention_layers":
        INTERVENTION_LAYERS,

    "total_generations":
        int(
            len(selected)
            * len(
                INTERVENTION_LAYERS
            )
        ),

    "outputs": {
        "behavioural":
            str(
                behaviour_path
            ),

        "trajectory":
            str(
                trajectory_summary_path
            ),

        "effects":
            str(
                effects_path
            ),

        "combined":
            str(
                combined_path
            ),

        "per_prompt":
            str(
                metrics_path
            ),
    },
}

with open(
    ANALYSIS_DIR
    / "analysis_summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ================================================================
# PRINT MAIN TABLE
# ================================================================

print(
    "\n"
    + "=" * 72
)

print(
    "MAIN LAYER SUMMARY"
)

print(
    "=" * 72
)

print(
    combined_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)

print(
    "\n"
    + "=" * 72
)

print(
    "ANALYSIS COMPLETE"
)

print(
    f"Output directory: {ANALYSIS_DIR}"
)

print(
    "=" * 72
)

