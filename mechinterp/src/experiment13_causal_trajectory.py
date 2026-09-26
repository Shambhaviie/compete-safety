"""
Experiment 13 — Layer-wise causal trajectory tracing

SCIENTIFIC OBJECTIVE
--------------------
Determine how the residual-stream trajectory differs between:

    A = isolation / natural refusal
    C = competition / natural compliance

and whether causally ablating the refusal-associated direction at
different layers produces a downstream trajectory resembling the
natural competition trajectory.

PRIMARY QUESTION
----------------
Does the causal effect of the refusal-associated direction arise from
a particular layer-specific computational stage, and do perturbations
at different layers propagate differently through the network?

INTERVENTION LAYERS
-------------------
L7  = beginning of strong causal region
L9  = peak causal effect
L11 = second peak causal effect
L14 = end of strong causal region
L18 = strongest representational layer
L24 = late strong representation / negligible standalone causal effect

IMPORTANT INTERPRETATION
------------------------
We do NOT call any layer "the safety decision layer".

The experiment distinguishes:
    - representational divergence
    - causal perturbation
    - downstream propagation

Natural A/C activations already exist and are loaded from disk.

For intervention conditions, all 32 layer prefill activations are
recorded in one forward pass. The refusal-direction ablation remains
active throughout generation at the intervention layer, matching the
existing intervention methodology.

HOOK LOCATION
-------------
forward_pre_hook on model.model.layers[L]

The recorded activation is:
    residual stream entering layer L
    pre-RMSNorm / block input

At the intervention layer, the saved prefill activation is the
state BEFORE the intervention. Downstream layers therefore reflect
the intervention.

DATA
----
676 paired A/C prompts already available.

For the trajectory experiment:
    10 flip prompts:
        A = refusal
        C = compliance

    10 stable prompts:
        A = refusal
        C = refusal

Prompts are selected deterministically from the held-out C test set.
Selection does not use any new intervention results.

OUTPUT
------
mechinterp/results/experiment13/

    selected_prompts.json

    trajectory_L7_ablation.npz
    trajectory_L9_ablation.npz
    trajectory_L11_ablation.npz
    trajectory_L14_ablation.npz
    trajectory_L18_ablation.npz
    trajectory_L24_ablation.npz

    responses_L*_ablation.json

    results.json
    trajectory_comparisons.npz

SMOKE TEST
----------
Run:
    python3 mechinterp/src/experiment13_causal_trajectory.py --smoke-only

This runs 2 prompts at all six intervention layers.

FULL RUN
--------
Run:
    python3 mechinterp/src/experiment13_causal_trajectory.py --full

The full run performs:
    20 prompts x 6 intervention layers = 120 generations.

Checkpointing occurs after every completed prompt.

No existing scripts or activation files are modified.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path("/home/Shambhavi/compete_safety_v2")

MODEL_PATH = str(ROOT / "models" / "llama_3_1_8b_instruct")
GUARD_PATH = str(ROOT / "models" / "llama_guard_3_8b")

ACT_DIR = ROOT / "mechinterp" / "activations"
PROBE_DIR = ROOT / "mechinterp" / "probes"
RESULT_DIR = ROOT / "mechinterp" / "results" / "experiment13"

A_ACT_FILE = ACT_DIR / "activations_prehook_A_seed42.npz"
C_ACT_FILE = ACT_DIR / "activations_prehook_C_seed42.npz"

A_META_FILE = ACT_DIR / "metadata_prehook_A_seed42.json"
C_META_FILE = ACT_DIR / "metadata_prehook_C_seed42.json"

DIRECTIONS_FILE = (
    PROBE_DIR / "directions_prehook_all_layers.npy"
)

SPLIT_FILE = PROBE_DIR / "train_test_split.json"

A_ROUNDS = (
    ROOT
    / "results_task6_676prompt"
    / "control_seed42"
    / "rounds.jsonl"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# EXPERIMENT CONSTANTS
# ---------------------------------------------------------------------

SEED = 42

N_LAYERS = 32
D_MODEL = 4096

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True

# Deliberately spans the distinct regimes found in existing work.
INTERVENTION_LAYERS = [7, 9, 11, 14, 18, 24]

N_FLIP = 10
N_STABLE = 10

SMOKE_N = 2


# ---------------------------------------------------------------------
# LOAD PHASE-2 RECORDS
# ---------------------------------------------------------------------

def load_phase2_records(path):
    records = []

    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)

            if rec.get("phase") == "phase2":
                records.append(rec)

    return records


# ---------------------------------------------------------------------
# COSINE / SUMMARY HELPERS
# ---------------------------------------------------------------------

def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0.0 or nb == 0.0:
        return np.nan

    return float(np.dot(a, b) / (na * nb))


def summarize(x):
    x = np.asarray(x, dtype=np.float64)

    if len(x) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
        }

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1))
        if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
    }


# ---------------------------------------------------------------------
# PROMPT SELECTION
# ---------------------------------------------------------------------

def select_prompts(acts_A, acts_C):
    """
    Select 10 flip + 10 stable prompts.

    Eligibility:
        - prompt must belong to the C held-out test set
        - A must refuse
        - C must either comply (flip) or refuse (stable)

    Selection uses only pre-existing labels and metadata.

    To avoid arbitrary "most extreme" selection, candidates are
    deterministically ordered by:
        category
        prompt length
        prompt ID

    This is a trajectory case-study sample, NOT a new population
    estimate.
    """

    meta_A = json.load(open(A_META_FILE))
    meta_C = json.load(open(C_META_FILE))

    labels_A = acts_A.shape[0]
    labels_C = acts_C.shape[0]

    if labels_A != 676 or labels_C != 676:
        raise RuntimeError(
            "Expected 676 activations in both A and C."
        )

    # Labels come from the activation files.
    data_A = np.load(A_ACT_FILE)
    data_C = np.load(C_ACT_FILE)

    rl_A = data_A["replayed_labels"].astype(bool)
    rl_C = data_C["replayed_labels"].astype(bool)

    split = json.load(open(SPLIT_FILE))
    test_indices = set(split["test_indices"])

    pid_A = {
        m["prompt_id"]: i
        for i, m in enumerate(meta_A)
    }

    pid_C = {
        m["prompt_id"]: i
        for i, m in enumerate(meta_C)
    }

    candidates_flip = []
    candidates_stable = []

    for pid in sorted(set(pid_A) & set(pid_C)):

        ia = pid_A[pid]
        ic = pid_C[pid]

        if ic not in test_indices:
            continue

        a_refused = bool(rl_A[ia])
        c_refused = bool(rl_C[ic])

        if not a_refused:
            continue

        item = {
            "prompt_id": pid,
            "A_index": int(ia),
            "C_index": int(ic),
            "category": meta_C[ic].get(
                "category",
                "unknown",
            ),
            "prompt_length": int(
                meta_C[ic].get(
                    "prompt_length",
                    0,
                )
            ),
            "round_number_A": meta_A[ia].get(
                "round_number"
            ),
            "round_number_C": meta_C[ic].get(
                "round_number"
            ),
        }

        if not c_refused:
            candidates_flip.append(item)

        else:
            candidates_stable.append(item)

    candidates_flip.sort(
        key=lambda x: (
            x["category"],
            x["prompt_length"],
            x["prompt_id"],
        )
    )

    candidates_stable.sort(
        key=lambda x: (
            x["category"],
            x["prompt_length"],
            x["prompt_id"],
        )
    )

    if len(candidates_flip) < N_FLIP:
        raise RuntimeError(
            f"Only {len(candidates_flip)} eligible flip prompts."
        )

    if len(candidates_stable) < N_STABLE:
        raise RuntimeError(
            f"Only {len(candidates_stable)} eligible stable prompts."
        )

    selected = []

    for item in candidates_flip[:N_FLIP]:
        item = dict(item)
        item["group"] = "flip"
        selected.append(item)

    for item in candidates_stable[:N_STABLE]:
        item = dict(item)
        item["group"] = "stable"
        selected.append(item)

    out = RESULT_DIR / "selected_prompts.json"

    with open(out, "w") as f:
        json.dump(selected, f, indent=2)

    print(
        f"Selected {N_FLIP} flip + "
        f"{N_STABLE} stable prompts.",
        flush=True,
    )

    print(
        f"Selection saved to: {out}",
        flush=True,
    )

    return selected


# ---------------------------------------------------------------------
# GENERATION WITH FULL TRAJECTORY RECORDING
# ---------------------------------------------------------------------

def generate_with_trajectory(
    model,
    tokenizer,
    prompt,
    round_number,
    direction,
    intervention_layer,
):
    """
    Generate one response while:

    1. recording the residual stream entering every layer during
       the PREFILL pass;

    2. applying the learned refusal-direction ablation at the
       selected intervention layer during EVERY forward pass.

    This matches the existing intervention methodology.

    IMPORTANT:
        trajectory[L] is the block INPUT before intervention at L.

        trajectory[L+1:] contains downstream consequences of
        the intervention.

    The intervention therefore persists throughout generation,
    while the saved trajectory corresponds to the prefill state.
    """

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

    attention_mask = torch.ones_like(input_ids)

    n_input = input_ids.shape[1]

    trajectory = {}

    fire_counts = {
        li: 0
        for li in range(N_LAYERS)
    }

    handles = []

    direction_t = torch.as_tensor(
        direction,
        dtype=torch.float32,
        device="cuda",
    )

    # -------------------------------------------------------------
    # Register one pre-hook per layer.
    # -------------------------------------------------------------

    for li in range(N_LAYERS):

        def make_hook(idx):

            def hook_fn(module, args):

                fire_counts[idx] += 1

                h = args[0]

                # -------------------------------------------------
                # Record only the first firing = prefill.
                # -------------------------------------------------

                if fire_counts[idx] == 1:

                    trajectory[idx] = (
                        h[0, -1, :]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .numpy()
                    )

                # -------------------------------------------------
                # Apply intervention on EVERY forward pass at the
                # target layer.
                # -------------------------------------------------

                if idx == intervention_layer:

                    h32 = h.to(torch.float32)

                    proj = (
                        h32 @ direction_t
                    ).unsqueeze(-1)

                    h32 = (
                        h32
                        - proj * direction_t
                    )

                    modified = list(args)

                    modified[0] = h32.to(
                        h.dtype
                    )

                    return tuple(modified)

                return args

            return hook_fn

        handles.append(
            model.model.layers[li]
            .register_forward_pre_hook(
                make_hook(li)
            )
        )

    # -------------------------------------------------------------
    # Deterministic sampling seed.
    # -------------------------------------------------------------

    torch.manual_seed(
        SEED + int(round_number)
    )

    with torch.no_grad():

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    # -------------------------------------------------------------
    # Remove hooks immediately.
    # -------------------------------------------------------------

    for handle in handles:
        handle.remove()

    generated = output[0][n_input:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    # -------------------------------------------------------------
    # Validate all layers were recorded.
    # -------------------------------------------------------------

    missing = [
        li
        for li in range(N_LAYERS)
        if li not in trajectory
    ]

    if missing:
        raise RuntimeError(
            "Missing trajectory layers: "
            f"{missing}. "
            f"Fire counts: {fire_counts}"
        )

    trajectory_arr = np.stack(
        [
            trajectory[li]
            for li in range(N_LAYERS)
        ],
        axis=0,
    ).astype(np.float32)

    return response, trajectory_arr


# ---------------------------------------------------------------------
# RUN ONE INTERVENTION LAYER
# ---------------------------------------------------------------------

def run_layer(
    model,
    tokenizer,
    judge,
    selected,
    acts_A,
    directions,
    layer,
):
    """
    Run A + refusal-direction ablation at one layer.

    Saves checkpoint after every prompt.
    """

    traj_file = (
        RESULT_DIR
        / f"trajectory_L{layer}_ablation.npz"
    )

    response_file = (
        RESULT_DIR
        / f"responses_L{layer}_ablation.json"
    )

    N = len(selected)

    trajectories = np.zeros(
        (N, N_LAYERS, D_MODEL),
        dtype=np.float32,
    )

    responses = {}

    completed = set()

    # -------------------------------------------------------------
    # Resume from response checkpoint.
    # -------------------------------------------------------------

    if response_file.exists():

        try:

            old = json.load(
                open(response_file)
            )

            completed = set(old.keys())
            responses = old

        except Exception:

            completed = set()
            responses = {}

    # -------------------------------------------------------------
    # If trajectory checkpoint exists, load it.
    # -------------------------------------------------------------

    if traj_file.exists():

        try:

            old_npz = np.load(
                traj_file,
                allow_pickle=True,
            )

            old_traj = old_npz[
                "trajectories"
            ]

            if old_traj.shape == trajectories.shape:
                trajectories[:] = old_traj

        except Exception:

            print(
                "Existing trajectory checkpoint "
                "could not be loaded; starting "
                "trajectory array from zero.",
                flush=True,
            )

    direction = directions[layer]

    print(
        "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        f"LAYER {layer} ABLATION",
        flush=True,
    )

    print(
        f"Prompts: {N}",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    t0 = time.time()

    for j, item in enumerate(selected):

        pid = item["prompt_id"]

        # ---------------------------------------------------------
        # Resume only if BOTH response and trajectory are present.
        # ---------------------------------------------------------

        if pid in completed:

            # We cannot infer whether trajectory row is valid
            # solely from response presence, so check the row.
            if np.any(
                trajectories[j] != 0
            ):
                print(
                    f"[L{layer}] "
                    f"{j+1}/{N} "
                    f"{pid} already complete "
                    f"-- skipping",
                    flush=True,
                )
                continue

        ia = item["A_index"]

        rec = ROUND_RECORDS_A[ia]

        response, trajectory = (
            generate_with_trajectory(
                model=model,
                tokenizer=tokenizer,
                prompt=rec["prompt"],
                round_number=rec["round_number"],
                direction=direction,
                intervention_layer=layer,
            )
        )

        trajectories[j] = trajectory

        # ---------------------------------------------------------
        # Judge the intervention output.
        # ---------------------------------------------------------

        judgment = judge.score_response(
            rec["task"],
            response,
        )

        is_safe = bool(
            judgment["is_safe"]
        )

        responses[pid] = {
            "prompt_id": pid,
            "group": item["group"],
            "response": response,
            "is_safe": is_safe,
            "round_number": rec[
                "round_number"
            ],
        }

        # ---------------------------------------------------------
        # CHECKPOINT AFTER EVERY PROMPT
        # ---------------------------------------------------------

        np.savez_compressed(
            str(traj_file),
            trajectories=trajectories,
            prompt_ids=np.array(
                [
                    x["prompt_id"]
                    for x in selected
                ],
                dtype=object,
            ),
            groups=np.array(
                [
                    x["group"]
                    for x in selected
                ],
                dtype=object,
            ),
            intervention_layer=np.array(
                layer,
                dtype=np.int64,
            ),
            intervention_location=np.array(
                "forward_pre_hook_block_input_prefill",
                dtype=object,
            ),
            intervention_applied_during_generation=np.array(
                True,
                dtype=bool,
            ),
        )

        with open(response_file, "w") as f:
            json.dump(
                responses,
                f,
                indent=2,
            )

        elapsed = (
            time.time() - t0
        )

        done = j + 1

        eta = (
            (N - done)
            / max(done, 1)
            * elapsed
        )

        print(
            f"[L{layer}] "
            f"{done}/{N} "
            f"group={item['group']} "
            f"safe={is_safe} "
            f"elapsed={elapsed/60:.1f}m "
            f"ETA={eta/60:.1f}m",
            flush=True,
        )

    return trajectories, responses


# ---------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------

def analyze(
    selected,
    acts_A,
    acts_C,
    intervention_trajectories,
    directions,
):
    """
    Compare:

        natural competition:
            A -> C

    against:

        A -> A + ablation

    at every layer.

    Importantly, the intervention layer itself represents the
    pre-intervention state in the saved trajectory. Therefore
    downstream comparison is especially important from L+1 onward.
    """

    N = len(selected)

    A_ref = np.zeros(
        (N, N_LAYERS, D_MODEL),
        dtype=np.float32,
    )

    C_ref = np.zeros_like(A_ref)

    for j, item in enumerate(selected):

        A_ref[j] = acts_A[
            item["A_index"]
        ]

        C_ref[j] = acts_C[
            item["C_index"]
        ]

    natural_delta = C_ref - A_ref

    results = {
        "experiment": (
            "experiment13_causal_trajectory"
        ),
        "n_prompts": N,
        "n_flip": sum(
            x["group"] == "flip"
            for x in selected
        ),
        "n_stable": sum(
            x["group"] == "stable"
            for x in selected
        ),
        "intervention_layers": (
            INTERVENTION_LAYERS
        ),
        "hook": (
            "forward_pre_hook"
        ),
        "activation_location": (
            "block input / pre-RMSNorm"
        ),
        "interpretation_note": (
            "Trajectory similarity is "
            "descriptive evidence of "
            "convergence, not proof of "
            "identical mechanism."
        ),
        "layers": {},
    }

    flip_mask = np.array(
        [
            x["group"] == "flip"
            for x in selected
        ],
        dtype=bool,
    )

    stable_mask = ~flip_mask

    # -------------------------------------------------------------
    # For each intervention layer.
    # -------------------------------------------------------------

    for intervention_layer, trajectories in (
        intervention_trajectories.items()
    ):

        l9_key = f"L{intervention_layer}"

        results["layers"][l9_key] = {
            "intervention_layer": (
                intervention_layer
            ),
            "downstream": {},
        }

        # ---------------------------------------------------------
        # Compare trajectories at every layer.
        # ---------------------------------------------------------

        for l in range(N_LAYERS):

            natural = natural_delta[:, l, :]

            intervention_delta = (
                trajectories[:, l, :]
                - A_ref[:, l, :]
            )

            # -----------------------------------------------------
            # Per-prompt cosine:
            #
            # natural A->C shift
            # vs
            # intervention A->Aabl shift
            # -----------------------------------------------------

            cosines = np.array(
                [
                    cosine(
                        natural[i],
                        intervention_delta[i],
                    )
                    for i in range(N)
                ],
                dtype=np.float64,
            )

            # -----------------------------------------------------
            # Refusal-direction projections.
            # -----------------------------------------------------

            r = directions[l].astype(
                np.float64
            )

            natural_proj = (
                natural @ r
            )

            intervention_proj = (
                intervention_delta @ r
            )

            # -----------------------------------------------------
            # Norms.
            # -----------------------------------------------------

            natural_norm = np.linalg.norm(
                natural,
                axis=1,
            )

            intervention_norm = np.linalg.norm(
                intervention_delta,
                axis=1,
            )

            # -----------------------------------------------------
            # Similarity of resulting STATES:
            #
            # C state vs A+ablation state
            #
            # This is supplementary to trajectory-shift
            # similarity and should not be treated as causal.
            # -----------------------------------------------------

            state_cosines = np.array(
                [
                    cosine(
                        C_ref[i, l, :],
                        trajectories[i, l, :],
                    )
                    for i in range(N)
                ],
                dtype=np.float64,
            )

            results["layers"][l9_key][
                "downstream"
            ][str(l)] = {

                "natural_A_to_C_shift_norm": {
                    "all": summarize(
                        natural_norm
                    ),
                    "flip": summarize(
                        natural_norm[
                            flip_mask
                        ]
                    ),
                    "stable": summarize(
                        natural_norm[
                            stable_mask
                        ]
                    ),
                },

                "intervention_A_to_ablation_shift_norm": {
                    "all": summarize(
                        intervention_norm
                    ),
                    "flip": summarize(
                        intervention_norm[
                            flip_mask
                        ]
                    ),
                    "stable": summarize(
                        intervention_norm[
                            stable_mask
                        ]
                    ),
                },

                "natural_vs_intervention_shift_cosine": {
                    "all": summarize(
                        cosines
                    ),
                    "flip": summarize(
                        cosines[
                            flip_mask
                        ]
                    ),
                    "stable": summarize(
                        cosines[
                            stable_mask
                        ]
                    ),
                },

                "natural_competition_refusal_projection": {
                    "all": summarize(
                        natural_proj
                    ),
                    "flip": summarize(
                        natural_proj[
                            flip_mask
                        ]
                    ),
                    "stable": summarize(
                        natural_proj[
                            stable_mask
                        ]
                    ),
                },

                "ablation_refusal_projection": {
                    "all": summarize(
                        intervention_proj
                    ),
                    "flip": summarize(
                        intervention_proj[
                            flip_mask
                        ]
                    ),
                    "stable": summarize(
                        intervention_proj[
                            stable_mask
                        ]
                    ),
                },

                "natural_C_state_vs_ablation_state_cosine": {
                    "all": summarize(
                        state_cosines
                    ),
                    "flip": summarize(
                        state_cosines[
                            flip_mask
                        ]
                    ),
                    "stable": summarize(
                        state_cosines[
                            stable_mask
                        ]
                    ),
                },
            }

    # -------------------------------------------------------------
    # Save raw arrays.
    # -------------------------------------------------------------

    save_dict = {
        "A": A_ref,
        "C": C_ref,
        "A_to_C_delta": natural_delta,
        "prompt_ids": np.array(
            [
                x["prompt_id"]
                for x in selected
            ],
            dtype=object,
        ),
        "groups": np.array(
            [
                x["group"]
                for x in selected
            ],
            dtype=object,
        ),
    }

    for layer, trajectory in (
        intervention_trajectories.items()
    ):

        save_dict[
            f"A_L{layer}_ablation"
        ] = trajectory

        save_dict[
            f"A_to_L{layer}_ablation_delta"
        ] = (
            trajectory - A_ref
        )

    np.savez_compressed(
        str(
            RESULT_DIR
            / "trajectory_comparisons.npz"
        ),
        **save_dict,
    )

    with open(
        RESULT_DIR / "results.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    return results


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only 2 prompts x 6 intervention layers.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full 20-prompt experiment.",
    )

    args = parser.parse_args()

    if args.smoke_only == args.full:
        raise SystemExit(
            "Specify exactly one of "
            "--smoke-only or --full."
        )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "EXPERIMENT 13 — CAUSAL TRAJECTORY TRACING",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "Scientific question:",
        flush=True,
    )

    print(
        "Does causal perturbation at different layers "
        "produce downstream trajectories resembling "
        "natural competition-induced compliance?",
        flush=True,
    )

    print(
        f"Intervention layers: "
        f"{INTERVENTION_LAYERS}",
        flush=True,
    )

    print(
        "Hook: forward_pre_hook / block INPUT / pre-RMSNorm",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    # -------------------------------------------------------------
    # Load existing activation files.
    # -------------------------------------------------------------

    print(
        "\nLoading existing activations...",
        flush=True,
    )

    data_A = np.load(
        A_ACT_FILE
    )

    data_C = np.load(
        C_ACT_FILE
    )

    acts_A = data_A[
        "prefill_acts"
    ].astype(np.float32)

    acts_C = data_C[
        "prefill_acts"
    ].astype(np.float32)

    print(
        f"A shape: {acts_A.shape}",
        flush=True,
    )

    print(
        f"C shape: {acts_C.shape}",
        flush=True,
    )

    if acts_A.shape != (
        676,
        N_LAYERS,
        D_MODEL,
    ):
        raise RuntimeError(
            f"Unexpected A shape: "
            f"{acts_A.shape}"
        )

    if acts_C.shape != (
        676,
        N_LAYERS,
        D_MODEL,
    ):
        raise RuntimeError(
            f"Unexpected C shape: "
            f"{acts_C.shape}"
        )

    # -------------------------------------------------------------
    # Load directions.
    # -------------------------------------------------------------

    directions = np.load(
        DIRECTIONS_FILE
    ).astype(np.float32)

    if directions.shape != (
        N_LAYERS,
        D_MODEL,
    ):
        raise RuntimeError(
            f"Unexpected directions shape: "
            f"{directions.shape}"
        )

    # Defensive normalization.
    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    print(
        f"Directions shape: "
        f"{directions.shape}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Load A round records.
    # -------------------------------------------------------------

    global ROUND_RECORDS_A

    ROUND_RECORDS_A = (
        load_phase2_records(
            A_ROUNDS
        )
    )

    print(
        f"A phase2 records: "
        f"{len(ROUND_RECORDS_A)}",
        flush=True,
    )

    if len(ROUND_RECORDS_A) != 676:
        raise RuntimeError(
            "Expected exactly 676 A phase2 records."
        )

    # -------------------------------------------------------------
    # Select prompts.
    # -------------------------------------------------------------

    selected = select_prompts(
        acts_A,
        acts_C,
    )

    # -------------------------------------------------------------
    # Smoke/full selection.
    # -------------------------------------------------------------

    if args.smoke_only:
        run_selected = selected[:SMOKE_N]

        print(
            f"\nSMOKE TEST MODE: "
            f"{len(run_selected)} prompts",
            flush=True,
        )

    else:
        run_selected = selected

        print(
            f"\nFULL MODE: "
            f"{len(run_selected)} prompts",
            flush=True,
        )

    # -------------------------------------------------------------
    # Print selected prompt metadata.
    # -------------------------------------------------------------

    for item in run_selected:

        print(
            f"  {item['group']:>6} "
            f"{item['prompt_id']} "
            f"category={item['category']} "
            f"length={item['prompt_length']}",
            flush=True,
        )

    # -------------------------------------------------------------
    # Load model.
    # -------------------------------------------------------------

    print(
        "\nLoading Llama 3.1 8B (4-bit)...",
        flush=True,
    )

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
        f"VRAM after model: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )

    # -------------------------------------------------------------
    # Load judge.
    # -------------------------------------------------------------

    print(
        "\nLoading LlamaGuard...",
        flush=True,
    )

    sys.path.insert(
        0,
        str(ROOT / "src"),
    )

    from safety_judge import SafetyJudge

    judge = SafetyJudge(
        model_path=GUARD_PATH
    )

    judge.load()

    print(
        f"VRAM after model + judge: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )

    # -------------------------------------------------------------
    # SMOKE TEST
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        "RUNNING SMOKE TEST",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    for item in run_selected:

        rec = ROUND_RECORDS_A[
            item["A_index"]
        ]

        print(
            f"\nPrompt {item['prompt_id']} "
            f"({item['group']})",
            flush=True,
        )

        for layer in INTERVENTION_LAYERS:

            response, trajectory = (
                generate_with_trajectory(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rec["prompt"],
                    round_number=rec[
                        "round_number"
                    ],
                    direction=directions[
                        layer
                    ],
                    intervention_layer=layer,
                )
            )

            judgment = judge.score_response(
                rec["task"],
                response,
            )

            print(
                f"  L{layer}: "
                f"trajectory={trajectory.shape} "
                f"safe={bool(judgment['is_safe'])} "
                f"response_chars={len(response)}",
                flush=True,
            )

            if trajectory.shape != (
                N_LAYERS,
                D_MODEL,
            ):
                raise RuntimeError(
                    f"Incorrect trajectory shape: "
                    f"{trajectory.shape}"
                )

    print(
        "\nSMOKE TEST PASSED.",
        flush=True,
    )

    # -------------------------------------------------------------
    # If smoke-only, stop cleanly here.
    # -------------------------------------------------------------

    if args.smoke_only:

        judge.unload()

        del model

        torch.cuda.empty_cache()

        print(
            "\nSmoke test complete. "
            "No full experiment launched.",
            flush=True,
        )

        return

    # -------------------------------------------------------------
    # FULL RUN
    # -------------------------------------------------------------

    intervention_trajectories = {}

    for layer in INTERVENTION_LAYERS:

        trajectories, responses = run_layer(
            model=model,
            tokenizer=tokenizer,
            judge=judge,
            selected=run_selected,
            acts_A=acts_A,
            directions=directions,
            layer=layer,
        )

        intervention_trajectories[
            layer
        ] = trajectories

    # -------------------------------------------------------------
    # Analyze.
    # -------------------------------------------------------------

    print(
        "\nAnalyzing trajectories...",
        flush=True,
    )

    analyze(
        selected=run_selected,
        acts_A=acts_A,
        acts_C=acts_C,
        intervention_trajectories=(
            intervention_trajectories
        ),
        directions=directions,
    )

    # -------------------------------------------------------------
    # Cleanup.
    # -------------------------------------------------------------

    judge.unload()

    del model

    torch.cuda.empty_cache()

    print(
        "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        "EXPERIMENT 13 COMPLETE",
        flush=True,
    )

    print(
        f"Results: {RESULT_DIR}",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )


if __name__ == "__main__":
    main()
