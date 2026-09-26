"""
Benchmark one full activation-patching layer.

This reproduces the production activation-patching setup:

- Llama 3.1 8B Instruct
- 4-bit NF4
- sampling temperature 0.7
- max_new_tokens 512
- seed = 42 + round_number
- same SafetyJudge used by the project

Only one layer/direction is benchmarked:
    L9, A -> C, all 163 A->C flip prompts.

The focal model and LlamaGuard are loaded sequentially, not simultaneously.
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

# Project import
sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "src")
)

from safety_judge import SafetyJudge


# ============================================================
# CONFIG -- copied from working activation_patch_A_C.py
# ============================================================

SEED = 42
LAYER = 9

MODEL_PATH = "models/llama_3_1_8b_instruct"
GUARD_PATH = "models/llama_guard_3_8b"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True

ACT_DIR = Path("mechinterp/activations")

A_ACT_FILE = ACT_DIR / "activations_prehook_A_seed42.npz"
C_ACT_FILE = ACT_DIR / "activations_prehook_C_seed42.npz"

A_META_FILE = ACT_DIR / "metadata_prehook_A_seed42.json"
C_META_FILE = ACT_DIR / "metadata_prehook_C_seed42.json"

C_RUN_DIR = Path(
    "results_task6_676prompt/competition_with_memory_seed42"
)

OUT_DIR = Path(
    "mechinterp/results/activation_patch_benchmark"
)


# ============================================================
# DATA
# ============================================================

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
        f"Could not find prompt ID. Keys: {row.keys()}"
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
        for m in reversed(row["messages"]):
            if m.get("role") == "user":
                return m.get("content", "")

    raise KeyError(
        f"Could not find prompt text. Keys: {row.keys()}"
    )


# ============================================================
# MODEL
# ============================================================

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
        f"VRAM: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    return model, tokenizer


# ============================================================
# INPUT
# ============================================================

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


# ============================================================
# SEED
# ============================================================

def set_generation_seed(round_number):

    seed = SEED + int(round_number)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PATCHED GENERATION
# ============================================================

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
        prompt
    )

    set_generation_seed(round_number)

    hook_state = {
        "patched": False
    }

    def patch_hook(module, args):

        hidden_states = args[0]

        # First call = full prompt prefill.
        # Later calls during generation have sequence length 1.
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


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("ACTIVATION PATCH TIMING BENCHMARK")
    print("=" * 80)
    print()
    print("Layer:      9")
    print("Direction:  A -> C")
    print("Prompts:    163")
    print("Generation: sampling, T=0.7, max_new_tokens=512")
    print()

    # --------------------------------------------------------
    # LOAD ACTIVATIONS
    # --------------------------------------------------------

    print("Loading activations...")

    A = np.load(A_ACT_FILE)
    C = np.load(C_ACT_FILE)

    A_acts = A["prefill_acts"]

    with open(A_META_FILE, "r") as f:
        meta_A = json.load(f)

    with open(C_META_FILE, "r") as f:
        meta_C = json.load(f)

    # Use the exact labels stored with the activation datasets,
    # matching the working activation_patch_A_C.py implementation.
    A_labels = A["replayed_labels"].astype(bool)
    C_labels = C["replayed_labels"].astype(bool)

    prompt_ids = [
        str(x["prompt_id"])
        for x in meta_A
    ]

    flip_indices = np.where(
        (A_labels == 1) &
        (C_labels == 0)
    )[0]

    assert len(flip_indices) == 163

    print(
        f"Verified A->C flips: "
        f"{len(flip_indices)}"
    )

    # --------------------------------------------------------
    # LOAD C RECORDS
    # --------------------------------------------------------

    # Match the working experiment exactly:
    # only Phase-2 records are used, in their original order.
    C_records = [
        rec
        for rec in load_jsonl(C_RUN_DIR / "rounds.jsonl")
        if rec.get("phase") == "phase2"
    ]

    assert len(C_records) == 676, (
        f"Expected 676 Phase-2 C records, found {len(C_records)}"
    )

    # Metadata and records are aligned by index.
    for i, (meta, rec) in enumerate(
        zip(meta_C, C_records)
    ):
        assert meta["prompt_id"] == rec.get("prompt_id"), (
            f"C prompt_id mismatch at index {i}: "
            f"{meta['prompt_id']} vs {rec.get('prompt_id')}"
        )

        assert meta["round_number"] == rec.get("round_number"), (
            f"C round mismatch at index {i}: "
            f"{meta['round_number']} vs {rec.get('round_number')}"
        )

    # --------------------------------------------------------
    # LOAD FOCAL MODEL ONLY
    # --------------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 1: PATCHED GENERATION")
    print("=" * 80)

    generation_results = []

    start_total = time.perf_counter()

    for n, idx in enumerate(
        flip_indices,
        start=1
    ):

        prompt_id = prompt_ids[idx]

        # Use the metadata index directly. This preserves the
        # exact activation <-> Phase-2 record alignment.
        record = C_records[idx]

        assert str(record.get("prompt_id")) == prompt_id

        prompt = get_prompt(record)

        round_number = get_round_number(record)

        source_vector = torch.from_numpy(
            A_acts[idx, LAYER]
        )

        t0 = time.perf_counter()

        response, hook_fired = generate_patched(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            round_number=round_number,
            source_vector=source_vector,
            layer=LAYER,
        )

        elapsed = (
            time.perf_counter() - t0
        )

        generation_results.append(
            {
                "prompt_id": prompt_id,
                "round_number": round_number,
                "response": response,
                "generation_seconds": elapsed,
                "hook_fired": hook_fired,
            }
        )

        if n == 1 or n % 10 == 0:

            elapsed_total = (
                time.perf_counter()
                - start_total
            )

            avg = elapsed_total / n

            remaining = avg * (
                len(flip_indices) - n
            )

            print(
                f"{n:3d}/163 | "
                f"last={elapsed:.2f}s | "
                f"avg={avg:.2f}s | "
                f"ETA={remaining/60:.1f} min"
            )

    generation_total = (
        time.perf_counter()
        - start_total
    )

    hooks_failed = sum(
        not x["hook_fired"]
        for x in generation_results
    )

    print()
    print(
        f"Generation complete: "
        f"{generation_total/60:.2f} min"
    )

    print(
        f"Average: "
        f"{generation_total/163:.2f} sec/prompt"
    )

    print(
        f"Hook failures: {hooks_failed}"
    )

    # Save generations before unloading model.
    generation_file = (
        OUT_DIR /
        "L9_A_to_C_generations.json"
    )

    with open(
        generation_file,
        "w"
    ) as f:

        json.dump(
            generation_results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # UNLOAD FOCAL MODEL
    # --------------------------------------------------------

    print()
    print("Unloading focal model...")

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(
        f"VRAM after focal-model unload: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # --------------------------------------------------------
    # LOAD SAFETY JUDGE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 2: LLAMAGUARD TIMING")
    print("=" * 80)

    judge = SafetyJudge(
        model_path=GUARD_PATH,
        device="cuda",
    )

    judge.load()

    judge_start = time.perf_counter()

    safe_count = 0

    judged_results = []

    for n, item in enumerate(
        generation_results,
        start=1
    ):

        # Recover the corresponding C record through the
        # activation-aligned index.
        idx = next(
            i for i, x in enumerate(prompt_ids)
            if x == item["prompt_id"]
        )

        record = C_records[idx]

        prompt = get_prompt(record)

        t0 = time.perf_counter()

        result = judge.score_response(
            prompt,
            item["response"],
        )

        elapsed = (
            time.perf_counter() - t0
        )

        safe = bool(result["is_safe"])

        safe_count += int(safe)

        judged_results.append(
            {
                **item,
                "patched_is_safe": safe,
                "judge_seconds": elapsed,
                "judge_raw_output": result[
                    "raw_output"
                ],
            }
        )

        if n == 1 or n % 10 == 0:

            elapsed_total = (
                time.perf_counter()
                - judge_start
            )

            avg = elapsed_total / n

            remaining = avg * (
                len(generation_results) - n
            )

            print(
                f"{n:3d}/163 | "
                f"last={elapsed:.2f}s | "
                f"avg={avg:.2f}s | "
                f"ETA={remaining/60:.1f} min"
            )

    judge_total = (
        time.perf_counter()
        - judge_start
    )

    # --------------------------------------------------------
    # SAVE FINAL BENCHMARK
    # --------------------------------------------------------

    judged_file = (
        OUT_DIR /
        "L9_A_to_C_benchmark.json"
    )

    with open(
        judged_file,
        "w"
    ) as f:

        json.dump(
            {
                "layer": LAYER,
                "direction": "A_to_C",
                "n": 163,
                "patched_refusals": safe_count,
                "patched_refusal_rate": (
                    safe_count / 163
                ),
                "generation_seconds": (
                    generation_total
                ),
                "judge_seconds": (
                    judge_total
                ),
                "total_seconds": (
                    generation_total
                    + judge_total
                ),
                "generation_avg_seconds": (
                    generation_total / 163
                ),
                "judge_avg_seconds": (
                    judge_total / 163
                ),
                "total_avg_seconds": (
                    (
                        generation_total
                        + judge_total
                    ) / 163
                ),
                "results": judged_results,
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL TIMING
    # --------------------------------------------------------

    total = (
        generation_total
        + judge_total
    )

    projected = total * 32 * 2

    print()
    print("=" * 80)
    print("BENCHMARK RESULT")
    print("=" * 80)

    print(
        f"Patched refusal: "
        f"{safe_count}/163 "
        f"({100*safe_count/163:.1f}%)"
    )

    print()

    print(
        f"Generation: "
        f"{generation_total/60:.2f} min "
        f"({generation_total/163:.2f} sec/prompt)"
    )

    print(
        f"LlamaGuard: "
        f"{judge_total/60:.2f} min "
        f"({judge_total/163:.2f} sec/prompt)"
    )

    print(
        f"Total: "
        f"{total/60:.2f} min"
    )

    print()

    print(
        "Projected naive 32-layer x 2-direction run:"
    )

    print(
        f"  {projected/3600:.2f} hours"
    )

    print()

    print(
        f"Saved: {judged_file}"
    )

    judge.unload()


if __name__ == "__main__":
    main()

