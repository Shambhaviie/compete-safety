"""
a_condition_harmful_harmless_direction.py

A-condition harmful-vs-harmless representation analysis.

Goal
----
Estimate a normal-condition harm-associated direction from:

    stable A-condition harmful prompts
        MINUS
    harmless Alpaca instructions

and evaluate that direction on:

    1. A->C behavioural flips
    2. A->C stable refusals

IMPORTANT:
- The 163 A->C flip prompts are NEVER used to estimate the direction.
- Direction estimation uses the 512 stable A->C refusals.
- This avoids circularly constructing a direction from the prompts whose
  competition-induced behaviour we want to explain.
- For stable-prompt evaluation, 5-fold cross-fitting is used so each
  stable prompt is evaluated against a direction estimated without it.
- Flip prompts are evaluated against the direction estimated from ALL
  512 stable refusals.

Activation location
-------------------
Same as the existing prehook pipeline:
    residual stream entering each transformer block
    before RMSNorm
    final input-token position

Model
-----
Llama 3.1 8B Instruct, 4-bit quantization.

Outputs
-------
mechinterp/results/A_condition_harmful_harmless/
    harmless_activations_seed42.npz
    harmless_metadata_seed42.json
    layer_statistics.csv
    prompt_level_projections.csv
    direction_summary.json
    figures/
        01_harmful_harmless_direction_norm.png
        02_flip_vs_stable_projection.png
        03_flip_vs_stable_effect_size.png
        04_flip_vs_stable_auc.png
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_LAYERS = 32
D_MODEL = 4096

MODEL_PATH = "models/llama_3_1_8b_instruct"

A_ACTIVATIONS = Path(
    "mechinterp/activations/activations_prehook_A_seed42.npz"
)
A_METADATA = Path(
    "mechinterp/activations/metadata_prehook_A_seed42.json"
)

ALPACA_PATH = Path(
    "data/benchmarks/alpaca/phase1_overflow_tasks.jsonl"
)

OUT_DIR = Path(
    "mechinterp/results/A_condition_harmful_harmless"
)
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

HARMFUL_LABEL = 1
COMPLIANCE_LABEL = 0

N_HARMLESS = 676

N_FOLDS = 5

MAX_INPUT_TOKENS = 1024


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < eps:
        return 0.0

    return float(np.dot(a, b) / denom)


def cohens_d(x, y):
    """
    Standardized mean difference:
        (mean(x) - mean(y)) / pooled_sd
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return float("nan")

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = math.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy)
        / (nx + ny - 2)
    )

    if pooled < 1e-12:
        return float("nan")

    return float((np.mean(x) - np.mean(y)) / pooled)


def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=np.float64)

    qvals = np.full_like(pvals, np.nan)

    valid = np.isfinite(pvals)

    if not np.any(valid):
        return qvals

    pv = pvals[valid]
    m = len(pv)

    order = np.argsort(pv)
    ranked = pv[order]

    q = ranked * m / np.arange(1, m + 1)

    for i in range(m - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])

    q = np.minimum(q, 1.0)

    restored = np.empty_like(q)
    restored[order] = q

    qvals[valid] = restored

    return qvals


def deterministic_alpaca_records(path, n):
    """
    Load Alpaca records and choose a deterministic harmless subset.
    We shuffle using a local RNG so the selection is reproducible.
    """
    records = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rec = json.loads(line)

            task = rec.get("task", "")

            if not isinstance(task, str):
                continue

            task = task.strip()

            if len(task) < 10:
                continue

            records.append(rec)

    if len(records) < n:
        raise RuntimeError(
            f"Only {len(records)} Alpaca records available, "
            f"but {n} harmless prompts are required."
        )

    rng = np.random.default_rng(SEED)
    indices = rng.choice(
        len(records),
        size=n,
        replace=False,
    )

    selected = [records[i] for i in indices]

    return selected


# ============================================================
# HARMLESS ACTIVATION EXTRACTION
# ============================================================

def extract_harmless_activations(model, tokenizer, records):
    """
    Extract residual-stream activations at the same location as
    activations_prehook_A_seed42.npz:

        block input / pre-RMSNorm
        final input token
        layers 0..31

    No generation is performed.
    """

    out_file = OUT_DIR / "harmless_activations_seed42.npz"
    out_meta = OUT_DIR / "harmless_metadata_seed42.json"

    if out_file.exists() and out_meta.exists():
        print(f"\nFound existing harmless activations:")
        print(f"  {out_file}")
        print("Skipping extraction.")

        d = np.load(str(out_file))
        return d["prefill_acts"]

    N = len(records)

    acts = np.zeros(
        (N, N_LAYERS, D_MODEL),
        dtype=np.float16,
    )

    metadata = []

    print("\n" + "=" * 70)
    print("EXTRACTING HARMLESS ALPACA ACTIVATIONS")
    print("=" * 70)
    print(f"Prompts: {N}")
    print("Activation: block INPUT / pre-RMSNorm")
    print("Position: final input token")
    print()

    t0 = time.time()

    for i, rec in enumerate(records):

        prompt = rec["task"]

        chat = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        if input_ids.shape[1] > MAX_INPUT_TOKENS:
            input_ids = input_ids[:, -MAX_INPUT_TOKENS:]

        attention_mask = torch.ones_like(input_ids)

        captured = {}

        handles = []

        for li in range(N_LAYERS):

            def make_hook(idx):
                def hook_fn(module, args):

                    h = args[0]

                    captured[idx] = (
                        h[0, -1, :]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .numpy()
                    )

                    return args

                return hook_fn

            handles.append(
                model.model.layers[li].register_forward_pre_hook(
                    make_hook(li)
                )
            )

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        for h in handles:
            h.remove()

        for li in range(N_LAYERS):
            if li not in captured:
                raise RuntimeError(
                    f"Missing activation at layer {li} "
                    f"for harmless prompt {i}."
                )

            acts[i, li] = captured[li].astype(np.float16)

        metadata.append(
            {
                "index": i,
                "task_id": rec.get(
                    "task_id",
                    f"alpaca_{i:05d}"
                ),
                "source": rec.get(
                    "source",
                    "alpaca"
                ),
                "category": rec.get(
                    "category",
                    "alpaca_instruction"
                ),
                "prompt_length": len(prompt),
                "task": prompt,
            }
        )

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (N - i - 1) / max(rate, 1e-9)

            print(
                f"  [{i+1:>4}/{N}] "
                f"elapsed={elapsed/60:.1f}min "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

    np.savez_compressed(
        str(out_file),
        prefill_acts=acts,
    )

    with open(out_meta, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    elapsed = time.time() - t0

    print()
    print(f"Done in {elapsed/60:.1f} min")
    print(f"Saved: {out_file}")
    print(f"Shape: {acts.shape}")

    return acts


# ============================================================
# LOAD A CONDITION
# ============================================================

def load_a_condition():
    if not A_ACTIVATIONS.exists():
        raise FileNotFoundError(
            f"Missing A activations:\n{A_ACTIVATIONS}"
        )

    if not A_METADATA.exists():
        raise FileNotFoundError(
            f"Missing A metadata:\n{A_METADATA}"
        )

    d = np.load(str(A_ACTIVATIONS))

    acts = d["prefill_acts"].astype(np.float32)

    with open(A_METADATA, "r") as f:
        metadata = json.load(f)

    if len(metadata) != acts.shape[0]:
        raise RuntimeError(
            "A activation/metadata length mismatch: "
            f"{acts.shape[0]} vs {len(metadata)}"
        )

    return acts, metadata


# ============================================================
# IDENTIFY FLIPS / STABLES
# ============================================================

def identify_groups(metadata):
    """
    Current A/C convention:
        1 = refusal
        0 = compliance

    Flip:
        A refused, C complied

    Stable:
        A refused, C refused

    The C label is read from the C metadata file.
    """

    c_meta_path = Path(
        "mechinterp/activations/metadata_prehook_C_seed42.json"
    )

    if not c_meta_path.exists():
        raise FileNotFoundError(
            f"Missing C metadata:\n{c_meta_path}"
        )

    with open(c_meta_path, "r") as f:
        c_metadata = json.load(f)

    if len(metadata) != len(c_metadata):
        raise RuntimeError(
            "A/C metadata length mismatch."
        )

    a_by_id = {
        rec["prompt_id"]: rec
        for rec in metadata
    }

    c_by_id = {
        rec["prompt_id"]: rec
        for rec in c_metadata
    }

    common_ids = sorted(
        set(a_by_id) & set(c_by_id)
    )

    if len(common_ids) != len(metadata):
        raise RuntimeError(
            "A/C prompt IDs do not match exactly."
        )

    flip_ids = []
    stable_ids = []
    other_ids = []

    for pid in common_ids:

        a_label = int(
            bool(a_by_id[pid]["historical_label"])
        )

        c_label = int(
            bool(c_by_id[pid]["historical_label"])
        )

        if (
            a_label == HARMFUL_LABEL
            and c_label == COMPLIANCE_LABEL
        ):
            flip_ids.append(pid)

        elif (
            a_label == HARMFUL_LABEL
            and c_label == HARMFUL_LABEL
        ):
            stable_ids.append(pid)

        else:
            other_ids.append(pid)

    return (
        common_ids,
        flip_ids,
        stable_ids,
        other_ids,
        a_by_id,
        c_by_id,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(
    a_acts,
    a_metadata,
    harmless_acts,
    flip_ids,
    stable_ids,
    a_by_id,
):
    """
    Construct harm-associated direction from ALL stable A refusals.

    Then:
      - evaluate flips against this direction
      - evaluate stable prompts cross-fitted across 5 folds

    Projection is the scalar projection onto the normalized
    harmful-vs-harmless direction.
    """

    id_to_index = {
        rec["prompt_id"]: i
        for i, rec in enumerate(a_metadata)
    }

    flip_indices = np.array(
        [id_to_index[x] for x in flip_ids],
        dtype=int,
    )

    stable_indices = np.array(
        [id_to_index[x] for x in stable_ids],
        dtype=int,
    )

    print("\n" + "=" * 70)
    print("A-CONDITION HARMFUL-VS-HARMLESS DIRECTION")
    print("=" * 70)
    print(f"Stable harmful prompts: {len(stable_indices)}")
    print(f"Flip prompts:            {len(flip_indices)}")
    print(f"Harmless prompts:        {len(harmless_acts)}")
    print()

    # --------------------------------------------------------
    # Direction from ALL stable harmful prompts
    # --------------------------------------------------------

    stable_mean = np.mean(
        a_acts[stable_indices],
        axis=0,
        dtype=np.float64,
    )

    harmless_mean = np.mean(
        harmless_acts,
        axis=0,
        dtype=np.float64,
    )

    full_direction = (
        stable_mean - harmless_mean
    )

    direction_norms = np.linalg.norm(
        full_direction,
        axis=1,
    )

    full_direction_hat = (
        full_direction
        / np.maximum(
            direction_norms[:, None],
            1e-12,
        )
    )

    # --------------------------------------------------------
    # Flip projections using direction estimated ONLY from
    # stable prompts.
    # --------------------------------------------------------

    flip_proj_full = np.einsum(
        "nld,ld->nl",
        a_acts[flip_indices],
        full_direction_hat,
    )

    # --------------------------------------------------------
    # Stable cross-fitted projections
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    shuffled = stable_indices.copy()
    rng.shuffle(shuffled)

    folds = np.array_split(
        shuffled,
        N_FOLDS,
    )

    stable_proj_cv = np.zeros(
        (len(stable_indices), N_LAYERS),
        dtype=np.float64,
    )

    stable_fold_id = np.full(
        len(stable_indices),
        -1,
        dtype=int,
    )

    stable_position = {
        idx: pos
        for pos, idx in enumerate(stable_indices)
    }

    for fold_id, test_idx in enumerate(folds):

        train_idx = np.setdiff1d(
            stable_indices,
            test_idx,
        )

        train_harm_mean = np.mean(
            a_acts[train_idx],
            axis=0,
            dtype=np.float64,
        )

        fold_direction = (
            train_harm_mean - harmless_mean
        )

        fold_norm = np.linalg.norm(
            fold_direction,
            axis=1,
        )

        fold_hat = (
            fold_direction
            / np.maximum(
                fold_norm[:, None],
                1e-12,
            )
        )

        test_proj = np.einsum(
            "nld,ld->nl",
            a_acts[test_idx],
            fold_hat,
        )

        for j, idx in enumerate(test_idx):
            pos = stable_position[idx]
            stable_proj_cv[pos] = test_proj[j]
            stable_fold_id[pos] = fold_id

    # --------------------------------------------------------
    # Compare flips vs stable
    # --------------------------------------------------------

    rows = []

    p_t = []
    p_u = []

    for li in range(N_LAYERS):

        x = flip_proj_full[:, li]
        y = stable_proj_cv[:, li]

        mean_flip = float(np.mean(x))
        mean_stable = float(np.mean(y))

        median_flip = float(np.median(x))
        median_stable = float(np.median(y))

        sd_flip = float(np.std(x, ddof=1))
        sd_stable = float(np.std(y, ddof=1))

        diff = mean_flip - mean_stable

        d = cohens_d(x, y)

        t_stat, t_p = ttest_ind(
            x,
            y,
            equal_var=False,
        )

        u_stat, u_p = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        labels = np.concatenate(
            [
                np.ones(len(x)),
                np.zeros(len(y)),
            ]
        )

        scores = np.concatenate(
            [
                x,
                y,
            ]
        )

        auc = roc_auc_score(
            labels,
            scores,
        )

        rows.append(
            {
                "layer": li,
                "direction_norm": float(
                    direction_norms[li]
                ),
                "flip_mean_projection": mean_flip,
                "flip_median_projection": median_flip,
                "flip_sd_projection": sd_flip,
                "stable_mean_projection": mean_stable,
                "stable_median_projection": median_stable,
                "stable_sd_projection": sd_stable,
                "mean_difference_flip_minus_stable": diff,
                "cohens_d": d,
                "welch_t": float(t_stat),
                "welch_p": float(t_p),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_p": float(u_p),
                "auc": float(auc),
            }
        )

        p_t.append(t_p)
        p_u.append(u_p)

    q_t = benjamini_hochberg(p_t)
    q_u = benjamini_hochberg(p_u)

    for i, row in enumerate(rows):
        row["welch_q_bh"] = float(q_t[i])
        row["mannwhitney_q_bh"] = float(q_u[i])

    # --------------------------------------------------------
    # Prompt-level table
    # --------------------------------------------------------

    prompt_rows = []

    for j, idx in enumerate(flip_indices):

        rec = a_metadata[idx]

        for li in range(N_LAYERS):

            prompt_rows.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "group": "flip",
                    "layer": li,
                    "projection": float(
                        flip_proj_full[j, li]
                    ),
                }
            )

    for j, idx in enumerate(stable_indices):

        rec = a_metadata[idx]

        for li in range(N_LAYERS):

            prompt_rows.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "group": "stable",
                    "layer": li,
                    "projection": float(
                        stable_proj_cv[j, li]
                    ),
                    "cv_fold": int(
                        stable_fold_id[j]
                    ),
                }
            )

    return (
        rows,
        prompt_rows,
        full_direction,
        full_direction_hat,
        direction_norms,
        flip_proj_full,
        stable_proj_cv,
    )


# ============================================================
# PLOTS
# ============================================================

def make_plots(
    rows,
    flip_proj,
    stable_proj,
):
    layers = np.arange(N_LAYERS)

    flip_means = np.array(
        [r["flip_mean_projection"] for r in rows]
    )

    stable_means = np.array(
        [r["stable_mean_projection"] for r in rows]
    )

    diffs = np.array(
        [
            r["mean_difference_flip_minus_stable"]
            for r in rows
        ]
    )

    dvals = np.array(
        [r["cohens_d"] for r in rows]
    )

    aucs = np.array(
        [r["auc"] for r in rows]
    )

    direction_norms = np.array(
        [r["direction_norm"] for r in rows]
    )

    # --------------------------------------------------------
    # 1. Direction norm
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        layers,
        direction_norms,
        marker="o",
    )

    plt.xlabel("Layer")
    plt.ylabel("||mean(harmful) - mean(harmless)||")
    plt.title(
        "A-condition harmful-vs-harmless direction magnitude"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "01_harmful_harmless_direction_norm.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # 2. Flip vs stable projections
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        layers,
        flip_means,
        marker="o",
        label="A→C flips",
    )

    plt.plot(
        layers,
        stable_means,
        marker="o",
        label="Stable refusals (cross-fitted)",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Layer")
    plt.ylabel(
        "Projection onto A-condition harmful-vs-harmless direction"
    )

    plt.title(
        "Flip vs stable projections onto the normal A-condition direction"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "02_flip_vs_stable_projection.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Cohen's d
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        layers,
        dvals,
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
        "Effect size: A→C flips vs stable refusals"
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "03_flip_vs_stable_effect_size.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------
    # 4. AUC
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        layers,
        aucs,
        marker="o",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Layer")
    plt.ylabel("AUC")
    plt.title(
        "A-condition harm-associated projection as a flip classifier"
    )

    plt.ylim(0.4, 1.0)

    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "04_flip_vs_stable_auc.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# SAVE
# ============================================================

def save_results(
    rows,
    prompt_rows,
    full_direction,
    full_direction_hat,
    direction_norms,
    flip_proj,
    stable_proj,
    n_flip,
    n_stable,
    n_harmless,
):
    import csv

    # --------------------------------------------------------
    # Layer CSV
    # --------------------------------------------------------

    layer_csv = OUT_DIR / "layer_statistics.csv"

    fieldnames = list(rows[0].keys())

    with open(
        layer_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Prompt CSV
    # --------------------------------------------------------

    prompt_csv = OUT_DIR / "prompt_level_projections.csv"

    # Some prompt rows contain cv_fold (stable prompts), while
    # flip rows do not. Use the union of all fields.
    prompt_fields = sorted(
        set().union(*(row.keys() for row in prompt_rows))
    )

    with open(
        prompt_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=prompt_fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(prompt_rows)

    # --------------------------------------------------------
    # Direction NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        str(OUT_DIR / "harm_associated_direction_seed42.npz"),
        direction=full_direction.astype(np.float32),
        direction_hat=full_direction_hat.astype(np.float32),
        direction_norm=direction_norms.astype(np.float32),
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    late_layers = [18, 20, 22, 24, 26, 28, 30, 31]

    late_rows = [
        rows[i]
        for i in late_layers
    ]

    summary = {
        "seed": SEED,
        "n_flip": n_flip,
        "n_stable": n_stable,
        "n_harmless": n_harmless,
        "direction_definition": (
            "mean activation of stable A-condition harmful "
            "prompts minus mean activation of harmless Alpaca "
            "prompts"
        ),
        "activation_location": (
            "residual stream entering each transformer block "
            "before RMSNorm, final input token"
        ),
        "flip_direction_estimation_leakage": (
            "none: flips are excluded from direction estimation"
        ),
        "stable_evaluation": (
            "5-fold cross-fitted"
        ),
        "late_layer_summary": {
            str(layer): {
                "flip_mean_projection": r[
                    "flip_mean_projection"
                ],
                "stable_mean_projection": r[
                    "stable_mean_projection"
                ],
                "difference": r[
                    "mean_difference_flip_minus_stable"
                ],
                "cohens_d": r["cohens_d"],
                "auc": r["auc"],
                "welch_q_bh": r["welch_q_bh"],
            }
            for layer, r in zip(
                late_layers,
                late_rows,
            )
        },
    }

    with open(
        OUT_DIR / "direction_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("A-CONDITION HARMFUL-VS-HARMLESS DIRECTION ANALYSIS")
    print("=" * 70)
    print()
    print("Model:", MODEL_PATH)
    print("A activations:", A_ACTIVATIONS)
    print("Alpaca:", ALPACA_PATH)
    print()
    print("Direction:")
    print("  stable A harmful - harmless Alpaca")
    print()
    print("Evaluation:")
    print("  A→C flips:  direction estimated from ALL stable prompts")
    print("  stable:     5-fold cross-fitted")
    print("=" * 70)

    # --------------------------------------------------------
    # Load A
    # --------------------------------------------------------

    print("\nLoading A-condition activations...")

    a_acts, a_metadata = load_a_condition()

    print(f"  Shape: {a_acts.shape}")
    print(f"  Metadata: {len(a_metadata)}")

    # --------------------------------------------------------
    # Identify groups
    # --------------------------------------------------------

    (
        common_ids,
        flip_ids,
        stable_ids,
        other_ids,
        a_by_id,
        c_by_id,
    ) = identify_groups(a_metadata)

    print()
    print("Groups:")
    print(f"  Flips:  {len(flip_ids)}")
    print(f"  Stable: {len(stable_ids)}")
    print(f"  Other:  {len(other_ids)}")

    if len(flip_ids) != 163:
        print(
            f"WARNING: expected 163 flips, "
            f"found {len(flip_ids)}"
        )

    if len(stable_ids) != 512:
        print(
            f"WARNING: expected 512 stable refusals, "
            f"found {len(stable_ids)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Llama 3.1 8B (4-bit)...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="cuda",
    )

    model.eval()

    print(
        f"  VRAM: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # --------------------------------------------------------
    # Load deterministic Alpaca subset
    # --------------------------------------------------------

    print("\nLoading Alpaca prompts...")

    harmless_records = deterministic_alpaca_records(
        ALPACA_PATH,
        N_HARMLESS,
    )

    print(
        f"  Selected {len(harmless_records)} harmless prompts"
    )

    # --------------------------------------------------------
    # Extract harmless activations
    # --------------------------------------------------------

    harmless_acts = extract_harmless_activations(
        model,
        tokenizer,
        harmless_records,
    )

    print(
        f"  Harmless activation shape: "
        f"{harmless_acts.shape}"
    )

    # --------------------------------------------------------
    # Free model
    # --------------------------------------------------------

    del model
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    (
        rows,
        prompt_rows,
        full_direction,
        full_direction_hat,
        direction_norms,
        flip_proj,
        stable_proj,
    ) = analyze(
        a_acts=a_acts,
        a_metadata=a_metadata,
        harmless_acts=harmless_acts,
        flip_ids=flip_ids,
        stable_ids=stable_ids,
        a_by_id=a_by_id,
    )

    # --------------------------------------------------------
    # Print concise results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"{'Layer':>5} "
        f"{'Flip':>10} "
        f"{'Stable':>10} "
        f"{'Diff':>10} "
        f"{'d':>8} "
        f"{'AUC':>8} "
        f"{'BH-q':>10}"
    )

    print("-" * 70)

    for r in rows:

        if (
            r["layer"] <= 5
            or r["layer"] % 2 == 0
            or r["layer"] >= 28
        ):

            print(
                f"{r['layer']:>5} "
                f"{r['flip_mean_projection']:>10.3f} "
                f"{r['stable_mean_projection']:>10.3f} "
                f"{r['mean_difference_flip_minus_stable']:>10.3f} "
                f"{r['cohens_d']:>8.3f} "
                f"{r['auc']:>8.3f} "
                f"{r['welch_q_bh']:>10.3g}"
            )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    make_plots(
        rows,
        flip_proj,
        stable_proj,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        rows=rows,
        prompt_rows=prompt_rows,
        full_direction=full_direction,
        full_direction_hat=full_direction_hat,
        direction_norms=direction_norms,
        flip_proj=flip_proj,
        stable_proj=stable_proj,
        n_flip=len(flip_ids),
        n_stable=len(stable_ids),
        n_harmless=len(harmless_records),
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(f"\nResults:")
    print(f"  {OUT_DIR}/")

    print("\nFiles:")
    print("  layer_statistics.csv")
    print("  prompt_level_projections.csv")
    print("  direction_summary.json")
    print("  harm_associated_direction_seed42.npz")
    print("  figures/")


if __name__ == "__main__":
    main()
