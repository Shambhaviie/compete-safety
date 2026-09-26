import os
import json
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_PATH = "models/Llama-3.1-8B-Instruct"
GUARD_PATH = "models/llama_guard_3_8b"

ACT_A = "mechinterp/activations/activations_prehook_A_seed42.npz"
ACT_C = "mechinterp/activations/activations_prehook_C_seed42.npz"

META_A = "mechinterp/activations/activations_prehook_A_seed42_metadata.json"
META_C = "mechinterp/activations/activations_prehook_C_seed42_metadata.json"

A_RECORDS = "results_task6_676prompt/control_seed42/rounds.jsonl"
C_RECORDS = "results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl"

OUTPUT_DIR = Path("mechinterp/results/activation_patch_A_C_batched")


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_prompt_id(row):
    for key in ["prompt_id", "id", "round_number"]:
        if key in row:
            return str(row[key])
    raise KeyError(f"Could not find prompt id in record: {row.keys()}")


def get_round_number(row):
    for key in ["round_number", "round", "round_idx"]:
        if key in row:
            return int(row[key])
    raise KeyError(f"Could not find round number in record: {row.keys()}")


def get_user_prompt(row):
    for key in ["prompt", "user_prompt", "question"]:
        if key in row:
            return row[key]

    # fallback for common chat formats
    if "messages" in row:
        msgs = row["messages"]
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m.get("content", "")

    raise KeyError(f"Could not find prompt text in record: {row.keys()}")


# ============================================================
# LLAMAGUARD
# ============================================================

def load_guard():
    print("Loading LlamaGuard...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(GUARD_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        GUARD_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    print("LlamaGuard loaded.")
    return tok, model


def guard_is_safe(tokenizer, model, prompt, response):
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": response,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    decoded = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip().lower()

    return decoded.startswith("safe")


# ============================================================
# GENERATION
# ============================================================

def generate_once(
    model,
    tokenizer,
    prompt,
    seed,
    patch_layer=None,
    patch_vector=None,
):
    set_seed(seed)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hook_handle = None

    if patch_layer is not None:

        assert patch_vector is not None

        def patch_hook(module, args):
            hidden = args[0]

            # Only patch the PREFILL forward pass.
            #
            # During generation, sequence length becomes 1.
            # The first call has the full prompt.
            if hidden.shape[1] > 1:
                hidden = hidden.clone()

                source = patch_vector.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )

                hidden[:, -1, :] = source

                args = (hidden,) + args[1:]

            return args

        hook_handle = model.model.layers[patch_layer].register_forward_pre_hook(
            patch_hook
        )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if hook_handle is not None:
        hook_handle.remove()

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(range(32)),
    )

    parser.add_argument(
        "--direction",
        choices=["A_to_C", "C_to_A", "both"],
        default="both",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPTIMIZED A <-> C ACTIVATION PATCHING")
    print("=" * 80)
    print(f"Layers: {args.layers}")
    print(f"Direction: {args.direction}")
    print(f"Batch size: {args.batch_size}")
    print()

    # --------------------------------------------------------
    # LOAD ACTIVATIONS
    # --------------------------------------------------------

    print("Loading activation datasets...")

    A = np.load(ACT_A)
    C = np.load(ACT_C)

    A_acts = A["prefill_acts"]
    C_acts = C["prefill_acts"]

    print("A:", A_acts.shape)
    print("C:", C_acts.shape)

    assert A_acts.shape == (676, 32, 4096)
    assert C_acts.shape == (676, 32, 4096)

    with open(META_A) as f:
        meta_A = json.load(f)

    with open(META_C) as f:
        meta_C = json.load(f)

    ids_A = [str(x["prompt_id"]) for x in meta_A["prompts"]]
    ids_C = [str(x["prompt_id"]) for x in meta_C["prompts"]]

    assert ids_A == ids_C

    # --------------------------------------------------------
    # LOAD RECORDS
    # --------------------------------------------------------

    A_records = load_jsonl(A_RECORDS)
    C_records = load_jsonl(C_RECORDS)

    assert len(A_records) == 676
    assert len(C_records) == 676

    A_by_id = {
        get_prompt_id(x): x
        for x in A_records
    }

    C_by_id = {
        get_prompt_id(x): x
        for x in C_records
    }

    assert set(A_by_id) == set(C_by_id)

    # --------------------------------------------------------
    # IDENTIFY FLIPS
    # --------------------------------------------------------

    A_labels = np.array(meta_A["labels"])
    C_labels = np.array(meta_C["labels"])

    # 1 = refusal
    # 0 = compliance

    flip_indices = np.where(
        (A_labels == 1) &
        (C_labels == 0)
    )[0]

    print()
    print(f"A refusals: {A_labels.sum()}")
    print(f"C refusals: {C_labels.sum()}")
    print(f"A -> C flips: {len(flip_indices)}")

    assert len(flip_indices) == 163

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading Llama 3.1 8B...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model.eval()

    print(
        f"VRAM after model load: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # --------------------------------------------------------
    # LOAD GUARD
    # --------------------------------------------------------

    guard_tok, guard_model = load_guard()

    # --------------------------------------------------------
    # SELECT DIRECTIONS
    # --------------------------------------------------------

    directions = []

    if args.direction in ["A_to_C", "both"]:
        directions.append(("A_to_C", A_acts, C_records))

    if args.direction in ["C_to_A", "both"]:
        directions.append(("C_to_A", C_acts, A_records))

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    all_results = []

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    for direction_name, source_acts, target_records in directions:

        print()
        print("=" * 80)
        print(f"DIRECTION: {direction_name}")
        print("=" * 80)

        # We keep the exact same 163 prompts.
        indices = flip_indices.copy()

        if args.limit is not None:
            indices = indices[:args.limit]

        results_path = (
            OUTPUT_DIR /
            f"{direction_name}_per_prompt.jsonl"
        )

        summary_path = (
            OUTPUT_DIR /
            f"{direction_name}_summary.json"
        )

        # Start fresh.
        if results_path.exists():
            results_path.unlink()

        for layer in args.layers:

            print()
            print("-" * 80)
            print(
                f"{direction_name} | Layer {layer} | "
                f"{len(indices)} prompts"
            )
            print("-" * 80)

            patched_safe = 0
            total = 0

            # ------------------------------------------------
            # IMPORTANT:
            # This implementation intentionally runs prompts
            # sequentially because generate() is stateful and
            # the intervention must occur at a precise layer.
            #
            # The optimization is therefore memory / loading
            # efficiency and avoiding redundant model reloads.
            # ------------------------------------------------

            for j, idx in enumerate(indices):

                prompt_id = ids_A[idx]

                if direction_name == "A_to_C":
                    source_vector = torch.from_numpy(
                        source_acts[idx, layer]
                    )
                    target_record = C_by_id[prompt_id]
                else:
                    source_vector = torch.from_numpy(
                        source_acts[idx, layer]
                    )
                    target_record = A_by_id[prompt_id]

                prompt = get_user_prompt(target_record)

                round_number = get_round_number(
                    target_record
                )

                # Match the project's deterministic seed convention.
                generation_seed = SEED + round_number

                response = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    seed=generation_seed,
                    patch_layer=layer,
                    patch_vector=source_vector,
                )

                safe = guard_is_safe(
                    guard_tok,
                    guard_model,
                    prompt,
                    response,
                )

                if safe:
                    patched_safe += 1

                total += 1

                row = {
                    "direction": direction_name,
                    "layer": layer,
                    "prompt_id": prompt_id,
                    "round_number": round_number,
                    "patched_is_safe": bool(safe),
                }

                with open(results_path, "a") as f:
                    f.write(
                        json.dumps(row) + "\n"
                    )

                if (j + 1) % 10 == 0 or j == 0:
                    print(
                        f"  {j+1}/{len(indices)} | "
                        f"patched refusal="
                        f"{patched_safe}/{total} "
                        f"({100*patched_safe/total:.1f}%)"
                    )

            summary = {
                "direction": direction_name,
                "layer": layer,
                "n": total,
                "patched_refusal_rate": (
                    patched_safe / total
                    if total else None
                ),
                "patched_refusals": patched_safe,
            }

            all_results.append(summary)

            print(
                f"  FINAL LAYER {layer}: "
                f"{patched_safe}/{total} "
                f"refusal "
                f"({100*patched_safe/total:.1f}%)"
            )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_path = OUTPUT_DIR / "full_summary.json"

    with open(summary_path, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "layers": args.layers,
                "directions": args.direction,
                "n_flip_prompts": len(flip_indices),
                "results": all_results,
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for direction in ["A_to_C", "C_to_A"]:

        rows = [
            x for x in all_results
            if x["direction"] == direction
        ]

        if not rows:
            continue

        print()
        print(direction)
        print("-" * 60)
        print(
            f"{'Layer':<8}"
            f"{'Patched Refusal%':<20}"
            f"{'N':<8}"
        )

        for x in rows:
            print(
                f"{x['layer']:<8}"
                f"{100*x['patched_refusal_rate']:<20.1f}"
                f"{x['n']:<8}"
            )

    print()
    print(f"Saved: {summary_path}")

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del guard_model
    del model

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"VRAM after cleanup: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()

