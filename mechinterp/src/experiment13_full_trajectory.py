"""
EXPERIMENT 13 — FULL CAUSAL TRAJECTORY

Primary objective
-----------------
Test whether ablating the established refusal-associated direction
produces downstream residual-stream changes that are more aligned with
the naturally occurring isolation -> competition (A -> C) trajectory
for prompts that flip from refusal to compliance than for prompts that
remain refused.

Primary design
--------------
Full 676-prompt population:

    163 A -> C flip prompts
    163 A -> C stable-refusal prompts

All 163 eligible flips are used.

163 stable prompts are randomly sampled from the 513 eligible stable
prompts using a fixed random seed.

Total:
    326 prompts
    x 6 intervention layers
    = 1,956 generations

Intervention layers:
    7, 9, 11, 14, 18, 24

For every prompt/intervention-layer pair:
    1. Ablate that layer's established refusal direction.
    2. Capture the residual-stream trajectory at all 32 layers.
    3. Judge the generated response.
    4. Save trajectory + response + label.

Hook
----
forward_pre_hook only.

The intervention is applied to the block INPUT, before RMSNorm,
matching the activation location used to construct the directions.

Scientific scope
----------------
This experiment tests trajectory similarity/differential downstream
effects. It does NOT by itself prove that competition and ablation
implement the same circuit or mechanism.

The original held-out 48-flip/153-stable test population is NOT used
to select the primary sample. A later 48+48 held-out analysis can be
used as a robustness check.

Checkpointing
-------------
Every prompt is checkpointed.

Existing completed layers/prompts are resumed automatically.
"""

import gc
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


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path("/home/Shambhavi/compete_safety_v2")

MODEL_PATH = ROOT / "models" / "llama_3_1_8b_instruct"
GUARD_PATH = ROOT / "models" / "llama_guard_3_8b"

A_ACT_FILE = (
    ROOT
    / "mechinterp/activations/"
    / "activations_prehook_A_seed42.npz"
)

C_ACT_FILE = (
    ROOT
    / "mechinterp/activations/"
    / "activations_prehook_C_seed42.npz"
)

A_META_FILE = (
    ROOT
    / "mechinterp/activations/"
    / "metadata_prehook_A_seed42.json"
)

C_META_FILE = (
    ROOT
    / "mechinterp/activations/"
    / "metadata_prehook_C_seed42.json"
)

SPLIT_FILE = (
    ROOT
    / "mechinterp/probes/"
    / "train_test_split.json"
)

DIRECTIONS_FILE = (
    ROOT
    / "mechinterp/probes/"
    / "directions_prehook_all_layers.npy"
)

ROUNDS_FILE = (
    ROOT
    / "results_task6_676prompt/"
    / "competition_with_memory_seed42/"
    / "rounds.jsonl"
)

RESULT_DIR = (
    ROOT
    / "mechinterp/results/"
    / "experiment13_full"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# EXPERIMENT PARAMETERS
# =====================================================================

SEED = 42

# Fixed seed ONLY for selecting stable controls.
# This selection occurs before trajectory analysis.
SELECTION_SEED = 20260831

N_LAYERS = 32
D_MODEL = 4096

N_FLIP = 163
N_STABLE = 163

INTERVENTION_LAYERS = [
    7,
    9,
    11,
    14,
    18,
    24,
]

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True


# =====================================================================
# SAFETY JUDGE
# =====================================================================

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from safety_judge import SafetyJudge


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_model():

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
        str(MODEL_PATH)
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb,
        device_map="cuda",
    )

    model.eval()

    print(
        f"VRAM after model load: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )

    return model, tokenizer


# =====================================================================
# PROMPT MAP
# =====================================================================

def build_prompt_map():

    prompt_map = {}

    with open(ROUNDS_FILE) as f:

        for line in f:

            rec = json.loads(line)

            if rec.get("phase") != "phase2":
                continue

            pid = rec.get("prompt_id")

            if pid is None:
                continue

            prompt_map[pid] = {
                "prompt": rec["prompt"],
                "task": rec.get("task"),
                "round_number": rec.get(
                    "round_number"
                ),
            }

    print(
        f"\nLoaded {len(prompt_map)} phase-2 prompts.",
        flush=True,
    )

    return prompt_map


# =====================================================================
# PROMPT SELECTION
# =====================================================================

def select_prompts():

    print(
        "\nSelecting primary 163 + 163 sample...",
        flush=True,
    )

    data_A = np.load(
        A_ACT_FILE
    )

    data_C = np.load(
        C_ACT_FILE
    )

    rl_A = data_A[
        "replayed_labels"
    ].astype(bool)

    rl_C = data_C[
        "replayed_labels"
    ].astype(bool)

    meta_A = json.load(
        open(A_META_FILE)
    )

    meta_C = json.load(
        open(C_META_FILE)
    )

    pid_A = {
        m["prompt_id"]: i
        for i, m in enumerate(meta_A)
    }

    pid_C = {
        m["prompt_id"]: i
        for i, m in enumerate(meta_C)
    }

    flips = []
    stable = []

    # ---------------------------------------------------------------
    # Use the FULL 676-prompt population.
    # ---------------------------------------------------------------

    for pid in sorted(
        set(pid_A) & set(pid_C)
    ):

        ia = pid_A[pid]
        ic = pid_C[pid]

        a_refused = bool(
            rl_A[ia]
        )

        c_refused = bool(
            rl_C[ic]
        )

        # We only compare prompts that refuse in isolation.
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
            "round_number_A":
                meta_A[ia].get(
                    "round_number"
                ),
            "round_number_C":
                meta_C[ic].get(
                    "round_number"
                ),
        }

        if c_refused:
            stable.append(item)
        else:
            flips.append(item)

    print(
        f"Eligible flips:  {len(flips)}",
        flush=True,
    )

    print(
        f"Eligible stable: {len(stable)}",
        flush=True,
    )

    if len(flips) != 163:
        raise RuntimeError(
            f"Expected 163 eligible flips, "
            f"found {len(flips)}."
        )

    if len(stable) != 512:
        raise RuntimeError(
            f"Expected 512 eligible stable prompts, "
            f"found {len(stable)}."
        )

    # ---------------------------------------------------------------
    # Use all 163 flips.
    # ---------------------------------------------------------------

    selected_flips = flips

    # ---------------------------------------------------------------
    # Randomly select 163 stable prompts.
    #
    # This is the only selection performed on stable prompts.
    # It does NOT use activations, trajectory results, or intervention
    # outcomes.
    # ---------------------------------------------------------------

    rng = np.random.default_rng(
        SELECTION_SEED
    )

    stable_indices = rng.choice(
        len(stable),
        size=N_STABLE,
        replace=False,
    )

    stable_indices = sorted(
        stable_indices.tolist()
    )

    selected_stable = [
        stable[i]
        for i in stable_indices
    ]

    selected = []

    for item in selected_flips:

        x = dict(item)
        x["group"] = "flip"
        selected.append(x)

    for item in selected_stable:

        x = dict(item)
        x["group"] = "stable"
        selected.append(x)

    # Deterministic final ordering.
    selected.sort(
        key=lambda x: (
            x["group"],
            x["prompt_id"],
        )
    )

    selected_stable_ids = {
        x["prompt_id"]
        for x in selected_stable
    }

    excluded_stable_ids = [
        x["prompt_id"]
        for x in stable
        if x["prompt_id"]
        not in selected_stable_ids
    ]

    # ---------------------------------------------------------------
    # Save selection permanently.
    # ---------------------------------------------------------------

    out_file = (
        RESULT_DIR
        / "selected_prompts.json"
    )

    selection_record = {
        "design":
            "full_population_balanced_163_flip_163_stable",

        "selection_seed":
            SELECTION_SEED,

        "population_size":
            676,

        "n_eligible_flip":
            len(flips),

        "n_eligible_stable":
            len(stable),

        "n_selected_flip":
            len(selected_flips),

        "n_selected_stable":
            len(selected_stable),

        "n_total":
            len(selected),

        "excluded_stable_prompt_ids":
            excluded_stable_ids,

        "selected_prompts":
            selected,
    }

    with open(
        out_file,
        "w",
    ) as f:

        json.dump(
            selection_record,
            f,
            indent=2,
        )

    print(
        "\nFINAL PRIMARY DESIGN",
        flush=True,
    )

    print(
        f"  Flip:   {len(selected_flips)}",
        flush=True,
    )

    print(
        f"  Stable: {len(selected_stable)}",
        flush=True,
    )

    print(
        f"  Total:  {len(selected)}",
        flush=True,
    )

    print(
        f"  Saved:  {out_file}",
        flush=True,
    )

    return selected


def load_or_select_prompts():

    path = (
        RESULT_DIR
        / "selected_prompts.json"
    )

    if path.exists():

        print(
            "\nExisting prompt selection found.",
            flush=True,
        )

        with open(path) as f:
            record = json.load(f)

        selected = record[
            "selected_prompts"
        ]

        n_flip = sum(
            x["group"] == "flip"
            for x in selected
        )

        n_stable = sum(
            x["group"] == "stable"
            for x in selected
        )

        if (
            len(selected)
            == N_FLIP + N_STABLE
            and n_flip == N_FLIP
            and n_stable == N_STABLE
        ):

            print(
                f"Using existing selection: "
                f"{n_flip} flip + "
                f"{n_stable} stable.",
                flush=True,
            )

            return selected

        print(
            "Existing selection does not match "
            "the primary design. Recreating.",
            flush=True,
        )

    return select_prompts()


# =====================================================================
# DIRECTIONS
# =====================================================================

def load_directions():

    directions = np.load(
        DIRECTIONS_FILE
    ).astype(np.float32)

    expected_shape = (
        N_LAYERS,
        D_MODEL,
    )

    if directions.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected direction shape: "
            f"{directions.shape}; "
            f"expected {expected_shape}."
        )

    norms = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    directions = (
        directions
        / np.maximum(
            norms,
            1e-12,
        )
    )

    print(
        f"\nLoaded directions: "
        f"{directions.shape}",
        flush=True,
    )

    return directions


# =====================================================================
# GENERATION + TRAJECTORY CAPTURE
# =====================================================================

def generate_with_ablation(
    model,
    tokenizer,
    prompt,
    round_number,
    direction,
    intervention_layer,
):
    """
    Capture the natural downstream trajectory after a single
    directional ablation.

    IMPORTANT:
        Trajectory values are recorded at each layer's block INPUT
        during the prefill pass.

        The intervention itself is applied only at intervention_layer.
    """

    chat = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    input_ids = (
        tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    n_input = input_ids.shape[1]

    trajectory = np.zeros(
        (
            N_LAYERS,
            D_MODEL,
        ),
        dtype=np.float32,
    )

    fire_counts = {
        li: 0
        for li in range(N_LAYERS)
    }

    handles = []

    direction_t = torch.from_numpy(
        direction
    ).to(
        device="cuda",
        dtype=torch.float32,
    )

    # ---------------------------------------------------------------
    # Register pre-hooks at every layer.
    # ---------------------------------------------------------------

    for li in range(N_LAYERS):

        def make_hook(idx):

            def hook_fn(
                module,
                args,
            ):

                fire_counts[idx] += 1

                h = args[0]

                # ---------------------------------------------------
                # First invocation = prefill pass.
                # Capture final prompt-token residual stream.
                # ---------------------------------------------------

                if fire_counts[idx] == 1:

                    trajectory[idx] = (
                        h[
                            0,
                            -1,
                            :
                        ]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .numpy()
                    )

                # ---------------------------------------------------
                # Apply intervention only at selected layer.
                # ---------------------------------------------------

                if (
                    idx
                    == intervention_layer
                ):

                    h32 = h.to(
                        torch.float32
                    )

                    proj = (
                        h32 @ direction_t
                    ).unsqueeze(-1)

                    h32 = (
                        h32
                        - proj * direction_t
                    )

                    modified = list(args)

                    modified[0] = (
                        h32.to(
                            h.dtype
                        )
                    )

                    return tuple(
                        modified
                    )

                return args

            return hook_fn

        handles.append(
            model.model.layers[
                li
            ].register_forward_pre_hook(
                make_hook(li)
            )
        )

    # ---------------------------------------------------------------
    # Same generation convention as previous experiments.
    # ---------------------------------------------------------------

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
            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    # ---------------------------------------------------------------
    # Remove hooks.
    # ---------------------------------------------------------------

    for handle in handles:
        handle.remove()

    generated = output[
        0,
        n_input:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return response, trajectory


# =====================================================================
# CHECKPOINT HELPERS
# =====================================================================

def checkpoint_file(layer):

    return (
        RESULT_DIR
        / f"checkpoint_L{layer}.json"
    )


def trajectory_file(layer):

    return (
        RESULT_DIR
        / f"trajectory_L{layer}_ablation.npz"
    )


def load_completed(layer):

    path = checkpoint_file(
        layer
    )

    if not path.exists():
        return set()

    try:

        with open(path) as f:
            record = json.load(f)

        return set(
            record.get(
                "completed_prompt_ids",
                [],
            )
        )

    except Exception:

        return set()


# =====================================================================
# RUN ONE LAYER
# =====================================================================

def run_layer(
    model,
    tokenizer,
    judge,
    selected,
    directions,
    intervention_layer,
    prompt_map,
):

    print(
        "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        f"LAYER {intervention_layer} ABLATION",
        flush=True,
    )

    print(
        f"Prompts: {len(selected)} "
        f"(163 flip + 163 stable)",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    N = len(selected)

    out_file = trajectory_file(
        intervention_layer
    )

    ckpt_file = checkpoint_file(
        intervention_layer
    )

    completed = load_completed(
        intervention_layer
    )

    # ---------------------------------------------------------------
    # Allocate output arrays.
    # ---------------------------------------------------------------

    trajectories = np.zeros(
        (
            N,
            N_LAYERS,
            D_MODEL,
        ),
        dtype=np.float16,
    )

    responses = np.empty(
        N,
        dtype=object,
    )

    safety_labels = np.empty(
        N,
        dtype=object,
    )

    prompt_ids = np.array(
        [
            x["prompt_id"]
            for x in selected
        ],
        dtype=object,
    )

    groups = np.array(
        [
            x["group"]
            for x in selected
        ],
        dtype=object,
    )

    responses[:] = None
    safety_labels[:] = None

    # ---------------------------------------------------------------
    # Restore trajectory data if an interrupted run exists.
    # ---------------------------------------------------------------

    if out_file.exists():

        try:

            old = np.load(
                out_file,
                allow_pickle=True,
            )

            if (
                old["trajectories"].shape
                == trajectories.shape
            ):

                trajectories[:] = old[
                    "trajectories"
                ]

                old_responses = old[
                    "responses"
                ]

                old_labels = old[
                    "safety_labels"
                ]

                for i in range(N):

                    if (
                        old_responses[i]
                        is not None
                    ):

                        responses[i] = (
                            old_responses[i]
                        )

                    if (
                        old_labels[i]
                        is not None
                    ):

                        safety_labels[i] = (
                            bool(
                                old_labels[i]
                            )
                        )

                print(
                    "Restored existing "
                    "trajectory data.",
                    flush=True,
                )

        except Exception as e:

            print(
                f"Warning: could not restore "
                f"trajectory data: {e}",
                flush=True,
            )

    # ---------------------------------------------------------------
    # Run prompts.
    # ---------------------------------------------------------------

    t0 = time.time()

    for i, item in enumerate(
        selected
    ):

        pid = item[
            "prompt_id"
        ]

        if pid in completed:
            continue

        if pid not in prompt_map:

            raise RuntimeError(
                f"Prompt {pid} not found "
                f"in competition rounds."
            )

        rec = prompt_map[pid]

        response, trajectory = (
            generate_with_ablation(
                model=model,
                tokenizer=tokenizer,
                prompt=rec["prompt"],
                round_number=rec[
                    "round_number"
                ],
                direction=directions[
                    intervention_layer
                ],
                intervention_layer=
                    intervention_layer,
            )
        )

        judgment = judge.score_response(
            rec["task"],
            response,
        )

        safe = bool(
            judgment["is_safe"]
        )

        trajectories[i] = (
            trajectory.astype(
                np.float16
            )
        )

        responses[i] = response
        safety_labels[i] = safe

        completed.add(pid)

        # -----------------------------------------------------------
        # Checkpoint after EVERY prompt.
        # -----------------------------------------------------------

        np.savez_compressed(
            out_file,
            trajectories=trajectories,
            responses=responses,
            safety_labels=safety_labels,
            prompt_ids=prompt_ids,
            groups=groups,
            intervention_layer=
                intervention_layer,
        )

        with open(
            ckpt_file,
            "w",
        ) as f:

            json.dump(
                {
                    "intervention_layer":
                        intervention_layer,

                    "n_total":
                        N,

                    "n_completed":
                        len(completed),

                    "completed_prompt_ids":
                        sorted(
                            completed
                        ),

                    "complete":
                        False,
                },
                f,
                indent=2,
            )

        elapsed = (
            time.time() - t0
        )

        done = len(completed)

        remaining = N - done

        eta = (
            remaining
            / max(done, 1)
            * elapsed
        )

        print(
            f"[L{intervention_layer}] "
            f"{done:>3}/{N} "
            f"group={item['group']} "
            f"safe={safe} "
            f"elapsed={elapsed/60:.1f}m "
            f"ETA={eta/60:.1f}m",
            flush=True,
        )

    # ---------------------------------------------------------------
    # Mark layer complete.
    # ---------------------------------------------------------------

    with open(
        ckpt_file,
        "w",
    ) as f:

        json.dump(
            {
                "intervention_layer":
                    intervention_layer,

                "n_total":
                    N,

                "n_completed":
                    N,

                "completed_prompt_ids":
                    prompt_ids.tolist(),

                "complete":
                    True,
            },
            f,
            indent=2,
        )

    print(
        f"\nL{intervention_layer} COMPLETE.",
        flush=True,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "EXPERIMENT 13 — FULL CAUSAL TRAJECTORY",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "Primary design: "
        "163 flip + 163 stable",
        flush=True,
    )

    print(
        f"Intervention layers: "
        f"{INTERVENTION_LAYERS}",
        flush=True,
    )

    print(
        f"Total generations: "
        f"{(N_FLIP + N_STABLE) * len(INTERVENTION_LAYERS)}",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Selection.
    # ---------------------------------------------------------------

    selected = (
        load_or_select_prompts()
    )

    if len(selected) != (
        N_FLIP + N_STABLE
    ):

        raise RuntimeError(
            f"Expected "
            f"{N_FLIP + N_STABLE} prompts, "
            f"got {len(selected)}."
        )

    # ---------------------------------------------------------------
    # Prompt map + directions.
    # ---------------------------------------------------------------

    prompt_map = (
        build_prompt_map()
    )

    directions = (
        load_directions()
    )

    # ---------------------------------------------------------------
    # Load models.
    # ---------------------------------------------------------------

    model, tokenizer = (
        load_model()
    )

    print(
        "\nLoading LlamaGuard...",
        flush=True,
    )

    judge = SafetyJudge(
        model_path=str(
            GUARD_PATH
        )
    )

    judge.load()

    print(
        f"VRAM after both models: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Run layers.
    # ---------------------------------------------------------------

    for layer in (
        INTERVENTION_LAYERS
    ):

        ckpt = checkpoint_file(
            layer
        )

        if ckpt.exists():

            try:

                with open(ckpt) as f:
                    record = json.load(f)

                if record.get(
                    "complete",
                    False,
                ):

                    print(
                        f"\nL{layer} already "
                        f"complete — skipping.",
                        flush=True,
                    )

                    continue

            except Exception:
                pass

        run_layer(
            model=model,
            tokenizer=tokenizer,
            judge=judge,
            selected=selected,
            directions=directions,
            intervention_layer=layer,
            prompt_map=prompt_map,
        )

    # ---------------------------------------------------------------
    # Cleanup.
    # ---------------------------------------------------------------

    print(
        "\nCleaning up...",
        flush=True,
    )

    judge.unload()

    del judge
    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

    print(
        f"VRAM allocated after unload: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )

    print(
        "\n"
        + "=" * 72,
        flush=True,
    )

    print(
        "EXPERIMENT 13 FULL RUN COMPLETE",
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
