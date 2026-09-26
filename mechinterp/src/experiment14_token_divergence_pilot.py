"""
EXPERIMENT 14A — TOKEN-LEVEL NATURAL-GENERATION DIVERGENCE PILOT

Question:
    When refusal-direction ablation changes behaviour during natural
    generation, does the internal trajectory diverge before or after
    the natural and ablated responses begin producing different tokens?

Design:
    20 flip + 20 stable prompts = 40 prompts
    Intervention layers: 9, 11, 18, 24

For each prompt/layer:
    1. Run natural Condition C generation.
    2. Run the same generation with refusal-direction ablation.
    3. Record generated token IDs.
    4. Record residual-stream state entering L31 at every generated
       token position for both runs.
    5. Find the first generated token position at which the sequences
       differ.
    6. Compare the internal states before and at that divergence.

IMPORTANT:
    - Natural generation is preserved.
    - No teacher forcing.
    - Intervention uses forward_pre_hook only.
    - Same prompt and generation seed.
    - Same sampling parameters.
    - This script does NOT modify Experiment 13.
    - This is a pilot only.
"""

import json
import sys
import time
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT = Path("/home/Shambhavi/compete_safety_v2")

MODEL_PATH = "models/llama_3_1_8b_instruct"
GUARD_PATH = "models/llama_guard_3_8b"

A_ACT_FILE = ROOT / "mechinterp/activations/activations_prehook_A_seed42.npz"
C_ACT_FILE = ROOT / "mechinterp/activations/activations_prehook_C_seed42.npz"

A_META_FILE = ROOT / "mechinterp/activations/metadata_prehook_A_seed42.json"
C_META_FILE = ROOT / "mechinterp/activations/metadata_prehook_C_seed42.json"

SPLIT_FILE = ROOT / "mechinterp/probes/train_test_split.json"

C_ROUNDS_FILE = (
    ROOT
    / "results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl"
)

RESULT_DIR = ROOT / "mechinterp/results/experiment14_pilot"

SEED = 42
D_MODEL = 4096
N_LAYERS = 32

# Pilot size.
N_FLIP = 20
N_STABLE = 20

# Deliberately restrict to the four most informative layers.
INTERVENTION_LAYERS = [9, 11, 18, 24]

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
DO_SAMPLE = True

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPERS
# ================================================================

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def load_direction(layer):
    path = (
        ROOT
        / "mechinterp/probes/directions_prehook_all_layers.npy"
    )

    directions = np.load(path).astype(np.float32)

    if directions.shape != (32, 4096):
        raise RuntimeError(
            f"Unexpected direction shape: {directions.shape}"
        )

    direction = torch.from_numpy(
        directions[layer]
    ).to("cuda", dtype=torch.float32)

    direction = direction / torch.linalg.vector_norm(direction)

    return direction


def load_prompts():
    """
    Select 20 flip + 20 stable prompts from the already-established
    163 flip / 512 stable C test-set population.

    Selection is deterministic.

    Eligibility:
        A refuses
        C complies -> flip
        A refuses
        C refuses  -> stable
        C prompt must belong to the existing held-out test split.
    """

    data_A = np.load(A_ACT_FILE)
    data_C = np.load(C_ACT_FILE)

    rl_A = data_A["replayed_labels"].astype(bool)
    rl_C = data_C["replayed_labels"].astype(bool)

    meta_A = json.load(open(A_META_FILE))
    meta_C = json.load(open(C_META_FILE))

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

    flips = []
    stable = []

    for pid in sorted(set(pid_A) & set(pid_C)):

        ia = pid_A[pid]
        ic = pid_C[pid]

        if ic not in test_indices:
            continue

        # Require A refusal.
        if not bool(rl_A[ia]):
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
            "round_number": meta_C[ic].get(
                "round_number"
            ),
        }

        if not bool(rl_C[ic]):
            item["group"] = "flip"
            flips.append(item)

        else:
            item["group"] = "stable"
            stable.append(item)

    flips.sort(
        key=lambda x: (
            x["category"],
            x["prompt_length"],
            x["prompt_id"],
        )
    )

    stable.sort(
        key=lambda x: (
            x["category"],
            x["prompt_length"],
            x["prompt_id"],
        )
    )

    print(f"Eligible flips:  {len(flips)}", flush=True)
    print(f"Eligible stable: {len(stable)}", flush=True)

    if len(flips) < N_FLIP:
        raise RuntimeError(
            f"Only {len(flips)} eligible flips."
        )

    if len(stable) < N_STABLE:
        raise RuntimeError(
            f"Only {len(stable)} eligible stable prompts."
        )

    selected = (
        flips[:N_FLIP]
        + stable[:N_STABLE]
    )

    out = RESULT_DIR / "selected_prompts.json"

    with open(out, "w") as f:
        json.dump(selected, f, indent=2)

    print(
        f"Selected {N_FLIP} flip + "
        f"{N_STABLE} stable prompts.",
        flush=True,
    )

    return selected


# ================================================================
# NATURAL GENERATION WITH TOKEN-LEVEL L31 RECORDING
# ================================================================

def generate_natural(
    model,
    tokenizer,
    prompt,
    round_number,
):
    """
    Natural Condition C generation.

    Records:
        token_ids
        L31 residual stream at each generated token position

    We use a pre-hook on L31.

    During generation:
        - first call is prefill
        - subsequent calls correspond to generated tokens
          with KV cache

    The pre-hook captures the residual stream entering L31 for
    each generated token.
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

    states = []
    fire_count = 0

    def hook_fn(module, args):

        nonlocal fire_count

        h = args[0]

        fire_count += 1

        # Prefill is the first call and contains the entire prompt.
        # We do not store it here.
        if fire_count >= 2:

            # Decode step should have seq_len = 1.
            if h.shape[1] == 1:
                states.append(
                    h[0, -1, :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

        return args

    handle = model.model.layers[31].register_forward_pre_hook(
        hook_fn
    )

    torch.manual_seed(
        SEED + round_number
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

    handle.remove()

    generated_ids = (
        output[0][n_input:]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    states = np.asarray(
        states,
        dtype=np.float16,
    )

    return {
        "token_ids": generated_ids,
        "states_L31": states,
        "response": response,
    }


# ================================================================
# ABLATED GENERATION
# ================================================================

def generate_ablated(
    model,
    tokenizer,
    prompt,
    round_number,
    intervention_layer,
    direction,
):
    """
    Natural generation with refusal-direction ablation at one layer.

    Also records L31 residual-stream state at every generated
    token position.

    IMPORTANT:
        Intervention is a forward_pre_hook on the chosen layer.
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

    states = []
    fire_count = 0

    # ------------------------------------------------------------
    # Intervention hook
    # ------------------------------------------------------------

    def intervention_hook(module, args):

        h = args[0].to(torch.float32)

        proj = (
            h @ direction
        ).unsqueeze(-1)

        h = (
            h
            - proj * direction
        )

        modified = list(args)
        modified[0] = h.to(
            args[0].dtype
        )

        return tuple(modified)

    intervention_handle = (
        model.model.layers[
            intervention_layer
        ].register_forward_pre_hook(
            intervention_hook
        )
    )

    # ------------------------------------------------------------
    # L31 recording hook
    # ------------------------------------------------------------

    def recording_hook(module, args):

        nonlocal fire_count

        h = args[0]

        fire_count += 1

        if fire_count >= 2:

            if h.shape[1] == 1:

                states.append(
                    h[0, -1, :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

        return args

    recording_handle = (
        model.model.layers[31]
        .register_forward_pre_hook(
            recording_hook
        )
    )

    try:

        torch.manual_seed(
            SEED + round_number
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

    finally:

        intervention_handle.remove()
        recording_handle.remove()

    generated_ids = (
        output[0][n_input:]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    states = np.asarray(
        states,
        dtype=np.float16,
    )

    return {
        "token_ids": generated_ids,
        "states_L31": states,
        "response": response,
    }


# ================================================================
# TOKEN DIVERGENCE ANALYSIS
# ================================================================

def first_token_difference(
    natural_ids,
    ablated_ids,
):
    """
    Return first generated-token position where sequences differ.

    Position is zero-indexed relative to generated tokens.

    None means all overlapping tokens are identical.
    """

    n = min(
        len(natural_ids),
        len(ablated_ids),
    )

    for i in range(n):

        if natural_ids[i] != ablated_ids[i]:
            return i

    if len(natural_ids) != len(ablated_ids):
        return n

    return None


def cosine_similarity(a, b):
    """
    Cosine similarity between two vectors.
    """

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(a, b) / denom
    )


def analyze_pair(
    natural,
    ablated,
):
    """
    Compare L31 trajectories.

    We deliberately distinguish:

        pre-divergence:
            positions before the first token difference

        divergence token:
            position of first token difference

        post-divergence:
            positions after it

    This lets us ask whether hidden-state divergence exists
    before token sequences diverge.
    """

    natural_ids = natural["token_ids"]
    ablated_ids = ablated["token_ids"]

    natural_states = natural["states_L31"].astype(
        np.float32
    )

    ablated_states = ablated["states_L31"].astype(
        np.float32
    )

    first_diff = first_token_difference(
        natural_ids,
        ablated_ids,
    )

    n_states = min(
        len(natural_states),
        len(ablated_states),
    )

    if n_states == 0:
        return {
            "first_token_difference": first_diff,
            "n_common_state_positions": 0,
            "pre_divergence_mean_cosine": None,
            "divergence_token_cosine": None,
            "post_divergence_mean_cosine": None,
        }

    cosines = np.array(
        [
            cosine_similarity(
                natural_states[i],
                ablated_states[i],
            )
            for i in range(n_states)
        ],
        dtype=np.float32,
    )

    if first_diff is None:
        pre = cosines
        divergence = np.array(
            [],
            dtype=np.float32,
        )
        post = np.array(
            [],
            dtype=np.float32,
        )

    else:
        pre = cosines[
            :min(first_diff, n_states)
        ]

        if first_diff < n_states:
            divergence = cosines[
                first_diff:first_diff + 1
            ]
            post = cosines[
                first_diff + 1:
            ]
        else:
            divergence = np.array(
                [],
                dtype=np.float32,
            )
            post = np.array(
                [],
                dtype=np.float32,
            )

    return {
        "first_token_difference": (
            int(first_diff)
            if first_diff is not None
            else None
        ),
        "n_common_state_positions": int(
            n_states
        ),
        "pre_divergence_mean_cosine": (
            float(np.mean(pre))
            if len(pre)
            else None
        ),
        "pre_divergence_min_cosine": (
            float(np.min(pre))
            if len(pre)
            else None
        ),
        "divergence_token_cosine": (
            float(np.mean(divergence))
            if len(divergence)
            else None
        ),
        "post_divergence_mean_cosine": (
            float(np.mean(post))
            if len(post)
            else None
        ),
    }


# ================================================================
# RUN PILOT
# ================================================================

def main():

    print("=" * 78)
    print("EXPERIMENT 14A — TOKEN-LEVEL NATURAL-GENERATION PILOT")
    print("=" * 78)

    selected = load_prompts()

    print(
        f"Total prompts: {len(selected)}",
        flush=True,
    )

    print(
        f"Intervention layers: {INTERVENTION_LAYERS}",
        flush=True,
    )

    print(
        f"Total intervention generations: "
        f"{len(selected) * len(INTERVENTION_LAYERS)}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

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
        f"VRAM after model load: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    results = []

    total = (
        len(selected)
        * len(INTERVENTION_LAYERS)
    )

    completed = 0

    t0 = time.time()

    for p_idx, item in enumerate(selected):

        prompt_id = item["prompt_id"]
        group = item["group"]
        prompt = None

        # Get exact prompt text from rounds.
        if prompt is None:
            pass

        # --------------------------------------------------------
        # Locate prompt by prompt_id.
        # --------------------------------------------------------

        # This is loaded lazily once below.
        if p_idx == 0:

            records = [
                json.loads(line)
                for line in open(
                    C_ROUNDS_FILE
                )
                if json.loads(line)["phase"]
                == "phase2"
            ]

            prompt_map = {
                r.get("prompt_id"): r
                for r in records
            }

        rec = prompt_map[prompt_id]

        prompt = rec["prompt"]

        round_number = int(
            rec["round_number"]
        )

        print(
            "\n" + "-" * 78,
            flush=True,
        )

        print(
            f"PROMPT {p_idx + 1}/{len(selected)} "
            f"{prompt_id} ({group})",
            flush=True,
        )

        # --------------------------------------------------------
        # Natural baseline
        # --------------------------------------------------------

        natural = generate_natural(
            model,
            tokenizer,
            prompt,
            round_number,
        )

        print(
            f"  Natural: "
            f"tokens={len(natural['token_ids'])} "
            f"chars={len(natural['response'])}",
            flush=True,
        )

        # --------------------------------------------------------
        # Each intervention
        # --------------------------------------------------------

        for layer in INTERVENTION_LAYERS:

            direction = load_direction(
                layer
            )

            ablated = generate_ablated(
                model,
                tokenizer,
                prompt,
                round_number,
                layer,
                direction,
            )

            metrics = analyze_pair(
                natural,
                ablated,
            )

            natural_refused = None
            ablated_refused = None

            # We deliberately do not invoke LlamaGuard here.
            # Experiment 13 already established the behavioural
            # labels for these exact intervention runs.
            #
            # We only need the generated response and token
            # divergence for this pilot.

            result = {
                "prompt_id": prompt_id,
                "group": group,
                "round_number": round_number,
                "layer": layer,

                "natural_response": natural[
                    "response"
                ],

                "ablated_response": ablated[
                    "response"
                ],

                "natural_token_count": int(
                    len(natural["token_ids"])
                ),

                "ablated_token_count": int(
                    len(ablated["token_ids"])
                ),

                "natural_token_ids": natural[
                    "token_ids"
                ].tolist(),

                "ablated_token_ids": ablated[
                    "token_ids"
                ].tolist(),

                **metrics,
            }

            results.append(result)

            completed += 1

            elapsed = (
                time.time() - t0
            )

            eta = (
                (total - completed)
                / max(completed, 1)
                * elapsed
            )

            print(
                f"  L{layer}: "
                f"first_diff="
                f"{metrics['first_token_difference']} "
                f"pre_cos="
                f"{metrics['pre_divergence_mean_cosine']} "
                f"post_cos="
                f"{metrics['post_divergence_mean_cosine']} "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )

            del direction
            cleanup()

        # --------------------------------------------------------
        # Checkpoint after every prompt
        # --------------------------------------------------------

        checkpoint = (
            RESULT_DIR
            / "checkpoint.json"
        )

        with open(
            checkpoint,
            "w",
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
            )

        print(
            f"  Checkpoint saved "
            f"({len(results)} results).",
            flush=True,
        )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    final_file = (
        RESULT_DIR
        / "results.json"
    )

    with open(
        final_file,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 78,
        flush=True,
    )

    print(
        "EXPERIMENT 14A COMPLETE",
        flush=True,
    )

    print(
        f"Results: {final_file}",
        flush=True,
    )

    print(
        f"Total comparisons: {len(results)}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    del model
    cleanup()

    print(
        f"VRAM after cleanup: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
