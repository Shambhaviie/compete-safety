"""
Stage 1: full-layer causal screen for A <-> C activation patching.

Scientific question
-------------------
For A->C flip prompts:

    A = no competition -> refusal
    C = competition + memory -> compliance

Does replacing the internal state at a particular layer with the
corresponding state from the other condition causally change safety
behavior?

Stage 1 design
--------------
- 20 reproducibly selected A->C flip prompts
- all 32 layers
- both directions:
      A_to_C : C context + A activation
      C_to_A : A context + C activation
- final input-token position
- block input / pre-RMSNorm
- exact generation settings from the working experiment
- LlamaGuard judged after all focal-model generations

This is a SCREEN, not the final statistical test.
The strongest layers will subsequently be tested on all 163 flips.
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

# Project safety judge
sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "src")
)

from safety_judge import SafetyJudge


# ============================================================================
# CONFIG -- MATCH WORKING EXPERIMENT
# ============================================================================

SEED = 42

N_LAYERS = 32
D_MODEL = 4096

MODEL_PATH = "models/llama_3_1_8b_instruct"
GUARD_PATH = "models/llama_guard_3_8b"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True

# Reproducible screen selection.
SCREEN_SEED = 20260906
N_SCREEN = 20

ACT_DIR = Path("mechinterp/activations")

A_ACT_FILE = ACT_DIR / "activations_prehook_A_seed42.npz"
C_ACT_FILE = ACT_DIR / "activations_prehook_C_seed42.npz"

A_META_FILE = ACT_DIR / "metadata_prehook_A_seed42.json"
C_META_FILE = ACT_DIR / "metadata_prehook_C_seed42.json"

A_RUN_DIR = Path(
    "results_task6_676prompt/control_seed42"
)

C_RUN_DIR = Path(
    "results_task6_676prompt/competition_with_memory_seed42"
)

OUT_DIR = Path(
    "mechinterp/results/activation_patch_stage1"
)


# ============================================================================
# BASIC IO
# ============================================================================

def load_jsonl(path):
    records = []

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_prompt_id(row):
    for key in ["prompt_id", "id", "round_number"]:
        if key in row:
            return str(row[key])

    raise KeyError(
        f"Could not find prompt_id. Keys: {row.keys()}"
    )


def get_round_number(row):
    for key in ["round_number", "round", "round_idx"]:
        if key in row:
            return int(row[key])

    raise KeyError(
        f"Could not find round number. Keys: {row.keys()}"
    )


def get_prompt(row):
    for key in ["prompt", "user_prompt", "question"]:
        if key in row:
            return row[key]

    if "messages" in row:
        for message in reversed(row["messages"]):
            if message.get("role") == "user":
                return message.get("content", "")

    raise KeyError(
        f"Could not find prompt text. Keys: {row.keys()}"
    )


# ============================================================================
# PHASE-2 RECORD LOADING
# ============================================================================

def load_phase2_records(run_dir):
    """
    Exactly match the working activation_patch_A_C.py implementation.
    """

    path = run_dir / "rounds.jsonl"

    records = [
        rec
        for rec in load_jsonl(path)
        if rec.get("phase") == "phase2"
    ]

    return records


# ============================================================================
# GENERATION SEED
# ============================================================================

def set_generation_seed(round_number):
    """
    Exact seed convention used by the existing experiment.
    """

    seed = SEED + int(round_number)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# INPUT
# ============================================================================

def make_input(tokenizer, prompt):

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

    return input_ids, attention_mask


# ============================================================================
# MODEL
# ============================================================================

def load_model():

    print("\nLoading Llama 3.1 8B (4-bit NF4)...")

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

    return model, tokenizer


# ============================================================================
# PATCHED GENERATION
# ============================================================================

def generate_patched(
    model,
    tokenizer,
    prompt,
    round_number,
    source_vector,
    layer,
):

    input_ids, attention_mask = make_input(
        tokenizer,
        prompt,
    )

    set_generation_seed(round_number)

    hook_state = {
        "patched": False
    }

    def patch_hook(module, args):

        hidden_states = args[0]

        # Only patch the first full-sequence prefill.
        # Generation-time calls have sequence length 1.
        if (
            hidden_states.shape[1] > 1
            and not hook_state["patched"]
        ):

            hidden_states = hidden_states.clone()

            source = source_vector.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            hidden_states[:, -1, :] = source

            hook_state["patched"] = True

            return (hidden_states,) + args[1:]

        return args

    handle = model.model.layers[
        layer
    ].register_forward_pre_hook(
        patch_hook
    )

    try:

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

        handle.remove()

    generated = output[
        0,
        input_ids.shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return response, hook_state["patched"]


# ============================================================================
# SELECT SCREEN PROMPTS
# ============================================================================

def select_screen_prompts(
    flip_indices,
    meta_A,
):

    rng = np.random.default_rng(
        SCREEN_SEED
    )

    selected = rng.choice(
        flip_indices,
        size=N_SCREEN,
        replace=False,
    )

    selected = np.sort(selected)

    return selected


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("STAGE 1: FULL-LAYER A <-> C CAUSAL SCREEN")
    print("=" * 80)
    print()
    print(f"Total A->C flips:       163")
    print(f"Screen prompts:         {N_SCREEN}")
    print(f"Layers:                 {N_LAYERS}")
    print(f"Directions:             A_to_C + C_to_A")
    print(f"Screen selection seed:  {SCREEN_SEED}")
    print()
    print(
        "Patch site: block INPUT / pre-RMSNorm / "
        "final input-token position"
    )
    print(
        "Generation: sampling / T=0.7 / max_new_tokens=512"
    )
    print()

    # ========================================================================
    # LOAD ACTIVATIONS
    # ========================================================================

    print("Loading activation datasets...")

    assert A_ACT_FILE.exists(), (
        f"Missing: {A_ACT_FILE}"
    )

    assert C_ACT_FILE.exists(), (
        f"Missing: {C_ACT_FILE}"
    )

    assert A_META_FILE.exists(), (
        f"Missing: {A_META_FILE}"
    )

    assert C_META_FILE.exists(), (
        f"Missing: {C_META_FILE}"
    )

    A = np.load(str(A_ACT_FILE))
    C = np.load(str(C_ACT_FILE))

    A_prefill = A["prefill_acts"]
    C_prefill = C["prefill_acts"]

    # Use stored replay labels exactly as in working script.
    A_labels = A["replayed_labels"].astype(bool)
    C_labels = C["replayed_labels"].astype(bool)

    with open(A_META_FILE, "r") as f:
        A_meta = json.load(f)

    with open(C_META_FILE, "r") as f:
        C_meta = json.load(f)

    print(
        f"  A prefill shape: {A_prefill.shape}"
    )

    print(
        f"  C prefill shape: {C_prefill.shape}"
    )

    assert A_prefill.shape == (
        676,
        N_LAYERS,
        D_MODEL,
    )

    assert C_prefill.shape == (
        676,
        N_LAYERS,
        D_MODEL,
    )

    assert len(A_meta) == 676
    assert len(C_meta) == 676

    # Verify A/C activation alignment.
    for i, (a, c) in enumerate(
        zip(A_meta, C_meta)
    ):

        assert a["index"] == c["index"], (
            f"Index mismatch at {i}"
        )

        assert a["prompt_id"] == c["prompt_id"], (
            f"Prompt ID mismatch at {i}"
        )

    flip_indices = np.where(
        A_labels & (~C_labels)
    )[0]

    print(
        f"  A refusals:    {A_labels.sum()}"
    )

    print(
        f"  C refusals:    {C_labels.sum()}"
    )

    print(
        f"  A -> C flips:  {len(flip_indices)}"
    )

    assert len(flip_indices) == 163

    # ========================================================================
    # LOAD PHASE-2 RECORDS
    # ========================================================================

    print("\nLoading Phase-2 records...")

    A_records = load_phase2_records(
        A_RUN_DIR
    )

    C_records = load_phase2_records(
        C_RUN_DIR
    )

    assert len(A_records) == 676, (
        f"Expected 676 A records, found {len(A_records)}"
    )

    assert len(C_records) == 676, (
        f"Expected 676 C records, found {len(C_records)}"
    )

    # Exact metadata/record alignment.
    for i in range(676):

        assert (
            A_meta[i]["prompt_id"]
            == A_records[i].get("prompt_id")
        ), f"A prompt mismatch at {i}"

        assert (
            C_meta[i]["prompt_id"]
            == C_records[i].get("prompt_id")
        ), f"C prompt mismatch at {i}"

        assert (
            A_meta[i]["round_number"]
            == A_records[i].get("round_number")
        ), f"A round mismatch at {i}"

        assert (
            C_meta[i]["round_number"]
            == C_records[i].get("round_number")
        ), f"C round mismatch at {i}"

    print(
        "  Metadata/round alignment: PASSED"
    )

    # ========================================================================
    # SELECT 20 SCREEN PROMPTS
    # ========================================================================

    screen_indices = select_screen_prompts(
        flip_indices,
        A_meta,
    )

    print()
    print("=" * 80)
    print("SCREEN PROMPTS")
    print("=" * 80)

    for i, idx in enumerate(
        screen_indices,
        start=1,
    ):

        print(
            f"{i:2d}. index={idx:3d} "
            f"prompt_id={A_meta[idx]['prompt_id']} "
            f"category={A_meta[idx].get('category', 'NA')}"
        )

    # Save the exact screen membership.
    screen_file = (
        OUT_DIR /
        "screen_prompt_selection.json"
    )

    with open(screen_file, "w") as f:

        json.dump(
            {
                "screen_seed": SCREEN_SEED,
                "n_screen": N_SCREEN,
                "flip_pool_size": len(flip_indices),
                "indices": [
                    int(x)
                    for x in screen_indices
                ],
                "prompt_ids": [
                    A_meta[x]["prompt_id"]
                    for x in screen_indices
                ],
                "categories": [
                    A_meta[x].get(
                        "category",
                        None
                    )
                    for x in screen_indices
                ],
            },
            f,
            indent=2,
        )

    # ========================================================================
    # LOAD FOCAL MODEL
    # ========================================================================

    model, tokenizer = load_model()

    # ========================================================================
    # RUN ALL LAYERS / BOTH DIRECTIONS
    # ========================================================================

    directions = [
        (
            "A_to_C",
            A_prefill,
            C_records,
        ),
        (
            "C_to_A",
            C_prefill,
            A_records,
        ),
    ]

    generation_results = []

    total_generations = (
        len(screen_indices)
        * N_LAYERS
        * len(directions)
    )

    generation_counter = 0

    generation_start = time.perf_counter()

    print()
    print("=" * 80)
    print(
        f"PATCHED GENERATION: "
        f"{total_generations} TOTAL"
    )
    print("=" * 80)

    for direction_name, source_acts, target_records in directions:

        print()
        print("=" * 80)
        print(f"DIRECTION: {direction_name}")
        print("=" * 80)

        for layer in range(N_LAYERS):

            layer_start = time.perf_counter()

            layer_safe = 0
            layer_hook_failures = 0

            print()
            print(
                f"Layer {layer:2d} | "
                f"{direction_name} | "
                f"{len(screen_indices)} prompts"
            )

            for idx in screen_indices:

                prompt_id = A_meta[idx]["prompt_id"]

                target_record = target_records[idx]

                assert (
                    str(target_record["prompt_id"])
                    == str(prompt_id)
                )

                prompt = get_prompt(
                    target_record
                )

                round_number = get_round_number(
                    target_record
                )

                # ------------------------------------------------------------
                # Direction:
                #
                # A_to_C:
                #   target = C context
                #   source = A activation
                #
                # C_to_A:
                #   target = A context
                #   source = C activation
                # ------------------------------------------------------------

                source_vector = torch.from_numpy(
                    source_acts[
                        idx,
                        layer,
                    ]
                )

                response, hook_fired = generate_patched(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    round_number=round_number,
                    source_vector=source_vector,
                    layer=layer,
                )

                generation_counter += 1

                if not hook_fired:
                    layer_hook_failures += 1

                generation_results.append(
                    {
                        "direction": direction_name,
                        "layer": int(layer),
                        "index": int(idx),
                        "prompt_id": str(prompt_id),
                        "round_number": int(round_number),
                        "response": response,
                        "hook_fired": bool(hook_fired),
                    }
                )

                # Progress every 5 prompts.
                if (
                    len(generation_results) == 1
                    or generation_counter % 5 == 0
                ):

                    elapsed = (
                        time.perf_counter()
                        - generation_start
                    )

                    avg = (
                        elapsed
                        / generation_counter
                    )

                    remaining = (
                        total_generations
                        - generation_counter
                    )

                    eta = (
                        avg * remaining
                    )

                    print(
                        f"  overall "
                        f"{generation_counter:4d}/"
                        f"{total_generations} | "
                        f"avg={avg:.2f}s | "
                        f"ETA={eta/60:.1f} min"
                    )

            layer_elapsed = (
                time.perf_counter()
                - layer_start
            )

            print(
                f"  Layer {layer} complete: "
                f"{layer_elapsed/60:.2f} min | "
                f"hook failures="
                f"{layer_hook_failures}"
            )

    generation_total = (
        time.perf_counter()
        - generation_start
    )

    # ========================================================================
    # SAVE RAW GENERATIONS
    # ========================================================================

    generation_file = (
        OUT_DIR /
        "stage1_generations.json"
    )

    with open(
        generation_file,
        "w",
    ) as f:

        json.dump(
            generation_results,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PATCHED GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Total generations: {len(generation_results)}"
    )

    print(
        f"Total time: "
        f"{generation_total/60:.2f} min"
    )

    print(
        f"Average: "
        f"{generation_total/len(generation_results):.2f} "
        f"sec/generation"
    )

    # ========================================================================
    # UNLOAD FOCAL MODEL
    # ========================================================================

    print()
    print("Unloading focal model...")

    del model
    del tokenizer
    del A
    del C

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(
        f"VRAM after focal model unload: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # ========================================================================
    # LOAD SAFETY JUDGE
    # ========================================================================

    print()
    print("=" * 80)
    print("LOADING LLAMAGUARD")
    print("=" * 80)

    judge = SafetyJudge(
        model_path=GUARD_PATH,
        device="cuda",
    )

    judge.load()

    # ========================================================================
    # JUDGE ALL PATCHED RESPONSES
    # ========================================================================

    print()
    print("=" * 80)
    print(
        f"LLAMAGUARD JUDGING: "
        f"{len(generation_results)} RESPONSES"
    )
    print("=" * 80)

    # We need prompt text for judging.
    A_record_by_index = {
        i: A_records[i]
        for i in range(676)
    }

    C_record_by_index = {
        i: C_records[i]
        for i in range(676)
    }

    judged_results = []

    judge_start = time.perf_counter()

    for n, item in enumerate(
        generation_results,
        start=1,
    ):

        idx = item["index"]

        if item["direction"] == "A_to_C":
            target_record = C_record_by_index[idx]
        else:
            target_record = A_record_by_index[idx]

        prompt = get_prompt(
            target_record
        )

        result = judge.score_response(
            prompt,
            item["response"],
        )

        judged_results.append(
            {
                **item,
                "patched_is_safe": bool(
                    result["is_safe"]
                ),
                "violated_categories": (
                    result["violated_categories"]
                ),
                "judge_raw_output": (
                    result["raw_output"]
                ),
            }
        )

        if (
            n == 1
            or n % 20 == 0
        ):

            elapsed = (
                time.perf_counter()
                - judge_start
            )

            avg = elapsed / n

            remaining = (
                len(generation_results)
                - n
            )

            eta = avg * remaining

            print(
                f"  {n:4d}/"
                f"{len(generation_results)} | "
                f"avg={avg:.2f}s | "
                f"ETA={eta/60:.1f} min"
            )

    judge_total = (
        time.perf_counter()
        - judge_start
    )

    # ========================================================================
    # SAVE JUDGED RESULTS
    # ========================================================================

    judged_file = (
        OUT_DIR /
        "stage1_judged_results.json"
    )

    with open(
        judged_file,
        "w",
    ) as f:

        json.dump(
            judged_results,
            f,
            indent=2,
        )

    # ========================================================================
    # SUMMARIZE LAYER EFFECTS
    # ========================================================================

    summary = []

    for direction in [
        "A_to_C",
        "C_to_A",
    ]:

        direction_rows = [
            x
            for x in judged_results
            if x["direction"] == direction
        ]

        for layer in range(N_LAYERS):

            rows = [
                x
                for x in direction_rows
                if x["layer"] == layer
            ]

            n = len(rows)

            refusals = sum(
                x["patched_is_safe"]
                for x in rows
            )

            refusal_rate = (
                refusals / n
                if n
                else None
            )

            # Natural baseline for these selected A->C flips:
            #
            # A = 100% refusal
            # C = 0% refusal
            #
            # Therefore:
            # A_to_C delta = patched - 0
            # C_to_A delta = patched - 100
            #
            if direction == "A_to_C":
                natural_rate = 0.0
            else:
                natural_rate = 1.0

            delta_pp = (
                (refusal_rate - natural_rate)
                * 100
                if refusal_rate is not None
                else None
            )

            summary.append(
                {
                    "direction": direction,
                    "layer": layer,
                    "n": n,
                    "patched_refusals": refusals,
                    "patched_refusal_rate": (
                        refusal_rate
                    ),
                    "natural_refusal_rate": (
                        natural_rate
                    ),
                    "delta_pp": delta_pp,
                }
            )

    summary_file = (
        OUT_DIR /
        "stage1_layer_summary.json"
    )

    with open(
        summary_file,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================================
    # FINAL TABLE
    # ========================================================================

    print()
    print("=" * 80)
    print("STAGE 1 FINAL SUMMARY")
    print("=" * 80)

    for direction in [
        "A_to_C",
        "C_to_A",
    ]:

        print()
        print(direction)

        print(
            f"{'Layer':<8}"
            f"{'Patched Refusal%':<20}"
            f"{'Delta pp':<12}"
            f"{'N':<8}"
        )

        print("-" * 55)

        rows = [
            x
            for x in summary
            if x["direction"] == direction
        ]

        for x in rows:

            print(
                f"{x['layer']:<8}"
                f"{100*x['patched_refusal_rate']:<20.1f}"
                f"{x['delta_pp']:<12.1f}"
                f"{x['n']:<8}"
            )

    print()
    print("=" * 80)
    print("TIMING")
    print("=" * 80)

    print(
        f"Generation: "
        f"{generation_total/60:.2f} min"
    )

    print(
        f"LlamaGuard: "
        f"{judge_total/60:.2f} min"
    )

    print(
        f"Total: "
        f"{(generation_total + judge_total)/60:.2f} min"
    )

    print()
    print("Saved:")
    print(f"  {screen_file}")
    print(f"  {generation_file}")
    print(f"  {judged_file}")
    print(f"  {summary_file}")

    # ========================================================================
    # CLEANUP
    # ========================================================================

    judge.unload()

    print()
    print("DONE.")


if __name__ == "__main__":
    main()

