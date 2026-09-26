"""
activation_patch_A_C.py

A <-> C causal activation-patching experiment.

Scientific question
-------------------
For prompts where:
    A = no competition -> refusal
    C = competition + memory -> compliance

does replacing the competition-condition residual state with the
corresponding no-competition state restore refusal?

Primary intervention:
    C[h_L, final_input_position] <- A[h_L, final_input_position]

Reverse intervention:
    A[h_L, final_input_position] <- C[h_L, final_input_position]

Activation location
-------------------
Forward pre-hook on model.model.layers[L].

This is the residual stream entering transformer block L,
before RMSNorm / attention / MLP, matching the location used in
the existing pre-hook activation extraction.

Initial layers:
    9, 11, 14, 18

Use --all-layers later for the full 0..31 sweep.

Important
---------
The saved A/C activations are the prefill activations at the final
input-token position from the already validated activation extraction.
We therefore patch only that position during the PREFILL pass.

We do NOT patch subsequent generated-token passes.

Natural A/C behavioural labels come from the validated replay labels
stored with the activation dataset. Patched generations are newly
generated and independently judged by LlamaGuard.

Output
------
mechinterp/results/activation_patch_A_C/
    summary.json
    per_prompt_results.jsonl
    smoke_test.jsonl  (only if --limit is used)
"""

import argparse
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

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from safety_judge import SafetyJudge


# ============================================================================
# CONFIG
# ============================================================================

SEED = 42

N_LAYERS = 32
D_MODEL = 4096

MODEL_PATH = "models/llama_3_1_8b_instruct"
GUARD_PATH = "models/llama_guard_3_8b"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True

ACT_DIR = Path("mechinterp/activations")
OUT_DIR = Path("mechinterp/results/activation_patch_A_to_C_full_163")

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

DEFAULT_LAYERS = [9, 11, 14, 18]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_phase2_records(run_dir):
    """
    Load the Phase-2 records in exactly the same order used by
    activation extraction.
    """
    path = run_dir / "rounds.jsonl"

    records = [
        rec
        for rec in load_jsonl(path)
        if rec.get("phase") == "phase2"
    ]

    return records


def load_activation_data():
    print("\nLoading activation datasets...")

    assert A_ACT_FILE.exists(), f"Missing: {A_ACT_FILE}"
    assert C_ACT_FILE.exists(), f"Missing: {C_ACT_FILE}"
    assert A_META_FILE.exists(), f"Missing: {A_META_FILE}"
    assert C_META_FILE.exists(), f"Missing: {C_META_FILE}"

    A = np.load(str(A_ACT_FILE))
    C = np.load(str(C_ACT_FILE))

    with open(A_META_FILE, "r") as f:
        A_meta = json.load(f)

    with open(C_META_FILE, "r") as f:
        C_meta = json.load(f)

    print(f"  A prefill shape: {A['prefill_acts'].shape}")
    print(f"  C prefill shape: {C['prefill_acts'].shape}")

    assert A["prefill_acts"].shape == C["prefill_acts"].shape
    assert A["prefill_acts"].shape[1:] == (N_LAYERS, D_MODEL)

    assert len(A_meta) == len(C_meta)
    assert len(A_meta) == A["prefill_acts"].shape[0]

    # Verify prompt alignment.
    for i, (a, c) in enumerate(zip(A_meta, C_meta)):
        assert a["index"] == c["index"], (
            f"Index mismatch at {i}: {a['index']} vs {c['index']}"
        )

        assert a["prompt_id"] == c["prompt_id"], (
            f"Prompt ID mismatch at {i}: "
            f"{a['prompt_id']} vs {c['prompt_id']}"
        )

    # Primary labels are replayed labels:
    # True = safe/refusal
    # False = unsafe/compliance
    A_labels = A["replayed_labels"].astype(bool)
    C_labels = C["replayed_labels"].astype(bool)

    flips = np.where(A_labels & (~C_labels))[0]

    print(f"  Total prompts: {len(A_labels)}")
    print(f"  A refusals:    {A_labels.sum()}")
    print(f"  C refusals:    {C_labels.sum()}")
    print(f"  A -> C flips:  {len(flips)}")

    assert len(flips) == 163, (
        f"Expected 163 A->C flips, found {len(flips)}. "
        "Do not continue until this is understood."
    )

    return A, C, A_meta, C_meta, flips


def validate_round_alignment(A_meta, C_meta, A_records, C_records):
    """
    Check that the saved activation metadata corresponds to the original
    experimental records.
    """

    assert len(A_meta) == len(A_records)
    assert len(C_meta) == len(C_records)

    for i in range(len(A_meta)):
        am = A_meta[i]
        cm = C_meta[i]
        ar = A_records[i]
        cr = C_records[i]

        assert am["prompt_id"] == ar.get("prompt_id", ""), (
            f"A prompt_id mismatch at index {i}"
        )

        assert cm["prompt_id"] == cr.get("prompt_id", ""), (
            f"C prompt_id mismatch at index {i}"
        )

        assert am["round_number"] == ar.get("round_number"), (
            f"A round mismatch at index {i}"
        )

        assert cm["round_number"] == cr.get("round_number"), (
            f"C round mismatch at index {i}"
        )

    print("  Metadata/round alignment: PASSED")


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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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
# GENERATION
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


def set_generation_seed(round_number):
    """
    Match the existing experiment/replay convention.
    """
    seed = SEED + int(round_number)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_natural(model, tokenizer, prompt, round_number):
    """
    Natural generation with no intervention.

    Used only for smoke testing the generation pathway.
    Full runs use the already validated natural A/C labels.
    """

    input_ids, attention_mask = make_input(
        tokenizer, prompt
    )

    set_generation_seed(round_number)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, input_ids.shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return response


def generate_with_patch(
    model,
    tokenizer,
    prompt,
    round_number,
    source_activation,
    layer,
):
    """
    Generate with a single activation patch.

    source_activation:
        [4096] vector from the corresponding source condition.

    layer:
        transformer block whose INPUT residual stream is patched.

    IMPORTANT:
        We patch ONLY the final input-token position during PREFILL.

        The hook fires again during autoregressive decoding with
        seq_len=1. Those later calls are deliberately untouched.
    """

    input_ids, attention_mask = make_input(
        tokenizer, prompt
    )

    n_input = input_ids.shape[1]

    source = torch.as_tensor(
        source_activation,
        dtype=torch.float32,
        device="cuda",
    )

    # Convert once to model hidden-state dtype.
    # Llama 3.1 is running with bfloat16 compute.
    source = source.to(dtype=torch.bfloat16)

    fire_count = {"n": 0}
    patch_check = {
        "prefill_seen": False,
        "patched": False,
        "max_abs_change": None,
    }

    def patch_hook(module, args):
        """
        Forward PRE-hook.

        args[0] is the residual stream entering the transformer block.
        """

        hidden_states = args[0]

        fire_count["n"] += 1

        # First invocation for this layer is the prefill pass.
        if fire_count["n"] == 1:
            patch_check["prefill_seen"] = True

            # Expected shape:
            # [batch=1, sequence_length, hidden_size]
            assert hidden_states.ndim == 3
            assert hidden_states.shape[0] == 1
            assert hidden_states.shape[1] == n_input
            assert hidden_states.shape[2] == D_MODEL

            # Make a copy of the original final-position state
            # for a sanity check.
            original = hidden_states[
                0, -1, :
            ].detach()

            # Clone because we must not modify the incoming tensor
            # in-place.
            patched = hidden_states.clone()

            patched[
                0, -1, :
            ] = source.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            difference = (
                patched[0, -1, :] - original
            ).abs().max().item()

            patch_check["max_abs_change"] = float(
                difference
            )

            patch_check["patched"] = True

            return (patched,) + tuple(args[1:])

        # All decode-time calls are left untouched.
        return args

    handle = model.model.layers[
        layer
    ].register_forward_pre_hook(patch_hook)

    try:
        set_generation_seed(round_number)

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

    assert patch_check["prefill_seen"], (
        f"Layer {layer}: prefill hook never fired."
    )

    assert patch_check["patched"], (
        f"Layer {layer}: activation was not patched."
    )

    generated = output[
        0,
        input_ids.shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return response, patch_check


# ============================================================================
# JUDGING
# ============================================================================

def judge_response(judge, task, response):
    result = judge.score_response(
        task,
        response,
    )

    return bool(result["is_safe"])


# ============================================================================
# STATISTICS
# ============================================================================

def paired_counts(natural_labels, patched_labels):
    """
    Boolean convention:
        True  = refusal/safe
        False = compliance/unsafe

    Returns:
        refusal -> compliance
        compliance -> refusal
        unchanged refusal
        unchanged compliance
    """

    natural_labels = np.asarray(
        natural_labels,
        dtype=bool,
    )

    patched_labels = np.asarray(
        patched_labels,
        dtype=bool,
    )

    refusal_to_compliance = int(
        np.sum(
            natural_labels &
            (~patched_labels)
        )
    )

    compliance_to_refusal = int(
        np.sum(
            (~natural_labels) &
            patched_labels
        )
    )

    refusal_to_refusal = int(
        np.sum(
            natural_labels &
            patched_labels
        )
    )

    compliance_to_compliance = int(
        np.sum(
            (~natural_labels) &
            (~patched_labels)
        )
    )

    return {
        "refusal_to_compliance": refusal_to_compliance,
        "compliance_to_refusal": compliance_to_refusal,
        "refusal_to_refusal": refusal_to_refusal,
        "compliance_to_compliance": compliance_to_compliance,
    }


# ============================================================================
# ONE DIRECTION
# ============================================================================

def run_direction(
    model,
    tokenizer,
    judge,
    direction,
    A,
    C,
    A_meta,
    C_meta,
    A_records,
    C_records,
    flip_indices,
    layers,
    limit=None,
    output_path=None,
):
    """
    Run one direction:

        A_to_C:
            target = C
            source = A

        C_to_A:
            target = A
            source = C
    """

    if direction == "A_to_C":
        source_acts = A["prefill_acts"]
        target_acts = C["prefill_acts"]
        target_meta = C_meta
        target_records = C_records
        source_label = "A"
        target_label = "C"

    elif direction == "C_to_A":
        source_acts = C["prefill_acts"]
        target_acts = A["prefill_acts"]
        target_meta = A_meta
        target_records = A_records
        source_label = "C"
        target_label = "A"

    else:
        raise ValueError(direction)

    indices = list(flip_indices)

    if limit is not None:
        indices = indices[:limit]

    print("\n" + "=" * 80)
    print(f"DIRECTION: {direction}")
    print("=" * 80)
    print(f"Prompts: {len(indices)}")
    print(f"Layers:  {layers}")

    results = []

    total = len(indices) * len(layers)

    completed = 0
    t0 = time.time()

    for layer in layers:

        print("\n" + "-" * 80)
        print(
            f"{direction} | Layer {layer} | "
            f"{len(indices)} prompts"
        )
        print("-" * 80)

        for j, idx in enumerate(indices):

            meta = target_meta[idx]
            rec = target_records[idx]

            prompt_id = meta["prompt_id"]
            round_number = meta["round_number"]

            # Verify source/target identity for this prompt.
            if direction == "A_to_C":
                assert A_meta[idx]["prompt_id"] == prompt_id
                assert C_meta[idx]["prompt_id"] == prompt_id
            else:
                assert C_meta[idx]["prompt_id"] == prompt_id
                assert A_meta[idx]["prompt_id"] == prompt_id

            source_activation = source_acts[
                idx,
                layer,
                :,
            ]

            # Generate patched response.
            response, patch_info = generate_with_patch(
                model=model,
                tokenizer=tokenizer,
                prompt=rec["prompt"],
                round_number=round_number,
                source_activation=source_activation,
                layer=layer,
            )

            patched_safe = judge_response(
                judge,
                rec["task"],
                response,
            )

            # Natural label comes from validated activation replay.
            natural_safe = bool(
                target_acts is not None
            )

            # We do NOT derive natural_safe from activations.
            # Use the stored replayed label instead.
            if direction == "A_to_C":
                natural_safe = bool(
                    C["replayed_labels"][idx]
                )
            else:
                natural_safe = bool(
                    A["replayed_labels"][idx]
                )

            result = {
                "direction": direction,
                "layer": int(layer),
                "index": int(idx),
                "prompt_id": prompt_id,
                "round_number": int(round_number),
                "category": meta.get("category", ""),
                "source_condition": source_label,
                "target_condition": target_label,
                "natural_safe": natural_safe,
                "patched_safe": patched_safe,
                "natural_refusal": natural_safe,
                "patched_refusal": patched_safe,
                "natural_compliance": not natural_safe,
                "patched_compliance": not patched_safe,
                "patch_applied": bool(
                    patch_info["patched"]
                ),
                "prefill_seen": bool(
                    patch_info["prefill_seen"]
                ),
                "patch_max_abs_change": patch_info[
                    "max_abs_change"
                ],
                # Do not save harmful prompt/response text.
            }

            results.append(result)

            completed += 1

            if (
                (j + 1) % 5 == 0
                or j == 0
                or j + 1 == len(indices)
            ):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                remaining = total - completed
                eta = remaining / max(rate, 1e-6)

                print(
                    f"  Layer {layer}: "
                    f"{j+1}/{len(indices)} | "
                    f"patched_safe={patched_safe} | "
                    f"patch_delta={patch_info['max_abs_change']:.3f} | "
                    f"ETA={eta/60:.1f} min",
                    flush=True,
                )

    return results


# ============================================================================
# SUMMARY
# ============================================================================

def summarize(results):
    summary = {
        "n_results": len(results),
        "by_layer": {},
    }

    layers = sorted(
        set(r["layer"] for r in results)
    )

    for layer in layers:

        rows = [
            r for r in results
            if r["layer"] == layer
        ]

        natural = np.array(
            [r["natural_safe"] for r in rows],
            dtype=bool,
        )

        patched = np.array(
            [r["patched_safe"] for r in rows],
            dtype=bool,
        )

        counts = paired_counts(
            natural,
            patched,
        )

        summary["by_layer"][str(layer)] = {
            "n": len(rows),
            "natural_refusal_rate": float(
                natural.mean()
            ),
            "patched_refusal_rate": float(
                patched.mean()
            ),
            "delta_pp": float(
                100.0 * (
                    patched.mean() -
                    natural.mean()
                )
            ),
            **counts,
            "mean_patch_max_abs_change": float(
                np.mean([
                    r["patch_max_abs_change"]
                    for r in rows
                ])
            ),
        }

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=DEFAULT_LAYERS,
        help="Layers to patch. Default: 9 11 14 18",
    )

    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Patch all layers 0..31",
    )

    parser.add_argument(
        "--direction",
        choices=["A_to_C", "C_to_A", "both"],
        default="both",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N A->C flip prompts. "
             "Use --limit 5 for smoke testing.",
    )

    parser.add_argument(
        "--output-prefix",
        default="pilot",
    )

    args = parser.parse_args()

    if args.all_layers:
        layers = list(range(N_LAYERS))
    else:
        layers = args.layers

    assert all(
        0 <= L < N_LAYERS
        for L in layers
    ), f"Invalid layer list: {layers}"

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("A <-> C ACTIVATION PATCHING")
    print("=" * 80)
    print(f"Seed:        {SEED}")
    print(f"Layers:      {layers}")
    print(f"Direction:   {args.direction}")
    print(f"Limit:       {args.limit}")
    print(
        "Patch site:  block INPUT / pre-RMSNorm / "
        "final input-token position"
    )
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    A, C, A_meta, C_meta, flip_indices = (
        load_activation_data()
    )

    A_records = load_phase2_records(A_RUN_DIR)
    C_records = load_phase2_records(C_RUN_DIR)

    print(
        f"\n  A records: {len(A_records)}"
    )
    print(
        f"  C records: {len(C_records)}"
    )

    validate_round_alignment(
        A_meta,
        C_meta,
        A_records,
        C_records,
    )

    # ------------------------------------------------------------------
    # Load model + judge
    # ------------------------------------------------------------------

    model, tokenizer = load_model()

    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(
        model_path=GUARD_PATH
    )
    judge.load()

    print(
        f"  VRAM after LlamaGuard: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    all_results = {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    directions = (
        ["A_to_C", "C_to_A"]
        if args.direction == "both"
        else [args.direction]
    )

    for direction in directions:

        results = run_direction(
            model=model,
            tokenizer=tokenizer,
            judge=judge,
            direction=direction,
            A=A,
            C=C,
            A_meta=A_meta,
            C_meta=C_meta,
            A_records=A_records,
            C_records=C_records,
            flip_indices=flip_indices,
            layers=layers,
            limit=args.limit,
        )

        all_results[direction] = results

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    output_jsonl = (
        OUT_DIR /
        f"{args.output_prefix}_per_prompt.jsonl"
    )

    with open(output_jsonl, "w") as f:
        for direction, rows in all_results.items():
            for row in rows:
                f.write(
                    json.dumps(row) + "\n"
                )

    combined = []
    for rows in all_results.values():
        combined.extend(rows)

    summary = {
        "experiment": "A_C_activation_patching",
        "seed": SEED,
        "layers": layers,
        "limit": args.limit,
        "n_flip_prompts_total": int(
            len(flip_indices)
        ),
        "n_prompts_run": (
            min(len(flip_indices), args.limit)
            if args.limit is not None
            else len(flip_indices)
        ),
        "patch_location": (
            "transformer block input / pre-RMSNorm"
        ),
        "patch_position": (
            "final input-token position"
        ),
        "patch_only_prefill": True,
        "directions": directions,
        "results": {
            direction: summarize(rows)
            for direction, rows
            in all_results.items()
        },
    }

    summary_path = (
        OUT_DIR /
        f"{args.output_prefix}_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Print concise summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for direction, rows in all_results.items():

        print(f"\n{direction}")

        s = summarize(rows)

        print(
            f"{'Layer':<8}"
            f"{'Natural%':<12}"
            f"{'Patched%':<12}"
            f"{'Delta pp':<12}"
            f"{'R->C':<8}"
            f"{'C->R':<8}"
        )

        print("-" * 60)

        for layer, vals in s["by_layer"].items():

            print(
                f"{layer:<8}"
                f"{100*vals['natural_refusal_rate']:<12.1f}"
                f"{100*vals['patched_refusal_rate']:<12.1f}"
                f"{vals['delta_pp']:<12.1f}"
                f"{vals['refusal_to_compliance']:<8}"
                f"{vals['compliance_to_refusal']:<8}"
            )

    print("\nSaved:")
    print(f"  {output_jsonl}")
    print(f"  {summary_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    judge.unload()

    del judge
    del model

    gc.collect()
    torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()
