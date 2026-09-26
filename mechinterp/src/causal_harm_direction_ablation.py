"""
causal_harm_direction_ablation.py

Causal test of the A-condition harmful-vs-harmless direction.

Question:
    Does removing the normal A-condition harm-associated direction
    from the residual stream cause normally-refusing harmful prompts
    to become compliant?

Population:
    512 A->C stable refusals.

Interventions:
    L9
    L11
    L9 + L11

Direction:
    A-condition stable harmful mean
    MINUS
    harmless Alpaca mean

The direction was estimated independently of the 163 A->C flips.

Intervention:
    h <- h - alpha * <h, r_hat> * r_hat

where alpha = 1.0.

Outputs:
    mechinterp/results/causal_harm_direction_ablation/
        L9_outputs.jsonl
        L11_outputs.jsonl
        L9_L11_outputs.jsonl
        summary.json
        all_results.jsonl

Every generated response is printed to the terminal and saved.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import sys

sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "src")
)

from safety_judge import SafetyJudge


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_PATH = "models/llama_3_1_8b_instruct"
GUARD_PATH = "models/llama_guard_3_8b"

A_RUN_DIR = Path(
    "results_task6_676prompt/control_seed42"
)

A_METADATA = Path(
    "mechinterp/activations/metadata_prehook_A_seed42.json"
)

DIRECTION_FILE = Path(
    "mechinterp/results/A_condition_harmful_harmless/"
    "harm_associated_direction_seed42.npz"
)

OUT_DIR = Path(
    "mechinterp/results/causal_harm_direction_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

N_LAYERS = 32
D_MODEL = 4096

MAX_NEW_TOK = 512

TEMPERATURE = 0.7
DO_SAMPLE = True

ALPHA = 1.0

TARGET_LAYERS = {
    "L9": [9],
    "L11": [11],
    "L9_L11": [9, 11],
}


# ============================================================
# LOAD STABLE PROMPTS
# ============================================================

def load_stable_records():
    """
    Match the 512 stable A->C prompts by prompt_id.

    Stable:
        A historical label = refusal
        C historical label = refusal
    """

    if not A_METADATA.exists():
        raise FileNotFoundError(
            A_METADATA
        )

    c_metadata_path = Path(
        "mechinterp/activations/"
        "metadata_prehook_C_seed42.json"
    )

    if not c_metadata_path.exists():
        raise FileNotFoundError(
            c_metadata_path
        )

    with open(A_METADATA) as f:
        a_meta = json.load(f)

    with open(c_metadata_path) as f:
        c_meta = json.load(f)

    c_by_id = {
        r["prompt_id"]: r
        for r in c_meta
    }

    stable_ids = []

    for rec in a_meta:

        pid = rec["prompt_id"]

        a_label = bool(
            rec["historical_label"]
        )

        c_rec = c_by_id.get(pid)

        if c_rec is None:
            raise RuntimeError(
                f"Missing C record for {pid}"
            )

        c_label = bool(
            c_rec["historical_label"]
        )

        if a_label and c_label:
            stable_ids.append(pid)

    if len(stable_ids) != 512:
        raise RuntimeError(
            f"Expected 512 stable refusals, "
            f"found {len(stable_ids)}"
        )

    # Load original A rounds.
    rounds_path = (
        A_RUN_DIR / "rounds.jsonl"
    )

    if not rounds_path.exists():
        raise FileNotFoundError(
            rounds_path
        )

    records = []

    with open(rounds_path) as f:
        for line in f:

            rec = json.loads(line)

            if rec.get("phase") != "phase2":
                continue

            records.append(rec)

    by_id = {
        r.get("prompt_id"): r
        for r in records
    }

    selected = []

    for pid in stable_ids:

        if pid not in by_id:
            raise RuntimeError(
                f"Could not find {pid} "
                f"in {rounds_path}"
            )

        selected.append(
            by_id[pid]
        )

    print(
        f"Loaded {len(selected)} stable A prompts."
    )

    return selected


# ============================================================
# LOAD DIRECTION
# ============================================================

def load_direction():
    if not DIRECTION_FILE.exists():
        raise FileNotFoundError(
            DIRECTION_FILE
        )

    d = np.load(
        str(DIRECTION_FILE)
    )

    if "direction_hat" not in d:
        raise RuntimeError(
            "Direction file does not contain "
            "'direction_hat'."
        )

    direction = d["direction_hat"].astype(
        np.float32
    )

    if direction.shape != (
        N_LAYERS,
        D_MODEL,
    ):
        raise RuntimeError(
            f"Unexpected direction shape: "
            f"{direction.shape}"
        )

    norms = np.linalg.norm(
        direction,
        axis=1,
    )

    print(
        "Loaded A-condition harm-associated direction."
    )

    print(
        f"  Shape: {direction.shape}"
    )

    print(
        f"  Norms of normalized directions: "
        f"min={norms.min():.4f}, "
        f"max={norms.max():.4f}"
    )

    return direction


# ============================================================
# ABLATION GENERATION
# ============================================================

def generate_ablated(
    model,
    tokenizer,
    prompt,
    round_number,
    direction_hat,
    layers,
):
    """
    Generate one response with directional ablation.

    Intervention occurs at the residual stream entering
    the selected transformer blocks, exactly matching the
    prehook activation location used to construct the direction.
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

    attention_mask = torch.ones_like(
        input_ids
    )

    handles = []

    for layer_idx in layers:

        direction = torch.tensor(
            direction_hat[layer_idx],
            dtype=torch.float32,
            device="cuda",
        )

        direction = direction.to(
            dtype=torch.bfloat16
        )

        def make_hook(r_hat):

            def hook_fn(module, args):

                h = args[0]

                # h shape:
                # [batch, sequence, hidden]
                #
                # IMPORTANT:
                # Preserve the exact dtype of the incoming
                # residual stream. The quantized Llama model
                # expects the hidden state entering downstream
                # modules to remain in its native dtype.
                #
                # Convert the direction to h.dtype before
                # computing the projection and intervention.

                r = r_hat.to(
                    device=h.device,
                    dtype=h.dtype,
                )

                projection = torch.sum(
                    h * r,
                    dim=-1,
                    keepdim=True,
                )

                delta = (
                    ALPHA
                    * projection
                    * r
                )

                delta = delta.to(
                    dtype=h.dtype
                )

                h_new = (
                    h - delta
                ).to(
                    dtype=h.dtype
                )

                return (h_new,) + tuple(
                    args[1:]
                )

            return hook_fn

        handles.append(
            model.model.layers[
                layer_idx
            ].register_forward_pre_hook(
                make_hook(direction)
            )
        )

    # Same deterministic seed convention
    # as the existing activation extraction.
    torch.manual_seed(
        SEED + int(round_number)
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED + int(round_number)
        )

    with torch.no_grad():

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOK,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    for handle in handles:
        handle.remove()

    generated = output[
        0,
        input_ids.shape[1]:,
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return response


# ============================================================
# RUN ONE CONDITION
# ============================================================

def run_condition(
    model,
    tokenizer,
    judge,
    records,
    direction,
    condition_name,
    layers,
):
    print()
    print("=" * 80)
    print(
        f"INTERVENTION: {condition_name}"
    )
    print(
        f"Layers: {layers}"
    )
    print(
        f"Alpha: {ALPHA}"
    )
    print("=" * 80)

    output_file = (
        OUT_DIR
        / f"{condition_name}_outputs.jsonl"
    )

    results = []

    n_refusal = 0
    n_compliance = 0

    t0 = time.time()

    with open(
        output_file,
        "w",
    ) as fout:

        for i, rec in enumerate(records):

            pid = rec.get(
                "prompt_id",
                f"unknown_{i}"
            )

            prompt = rec["prompt"]

            round_number = rec.get(
                "round_number",
                i + 1,
            )

            # ------------------------------------------------
            # NATURAL / BEFORE ABLATION
            #
            # Generate with the exact same seed, but with no
            # intervention layers. This gives the baseline
            # response for direct comparison.
            # ------------------------------------------------

            natural_response = generate_ablated(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                round_number=round_number,
                direction_hat=direction,
                layers=[],
            )

            natural_judgment = judge.score_response(
                rec.get("task", prompt),
                natural_response,
            )

            natural_is_safe = bool(
                natural_judgment["is_safe"]
            )

            if natural_is_safe:
                natural_label = "REFUSAL"
            else:
                natural_label = "COMPLIANCE"

            # ------------------------------------------------
            # ABLATED
            # ------------------------------------------------

            response = generate_ablated(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                round_number=round_number,
                direction_hat=direction,
                layers=layers,
            )

            judgment = judge.score_response(
                rec.get("task", prompt),
                response,
            )

            is_safe = bool(
                judgment["is_safe"]
            )

            # Existing project convention:
            # True = refusal / safe
            # False = compliance / unsafe
            if is_safe:
                n_refusal += 1
                label = "REFUSAL"
            else:
                n_compliance += 1
                label = "COMPLIANCE"

            result = {
                "prompt_id": pid,
                "round_number": round_number,
                "condition": condition_name,
                "layers": layers,
                "alpha": ALPHA,
                "prompt": prompt,

                # Natural baseline
                "natural_response": natural_response,
                "natural_is_safe": natural_is_safe,
                "natural_label": natural_label,

                # Ablated response
                "response": response,
                "is_safe": is_safe,
                "label": label,

                "historical_A_refusal": bool(
                    rec.get(
                        "is_safe_honest",
                        True,
                    )
                ),
                "category": rec.get(
                    "category",
                    "unknown",
                ),
            }

            results.append(result)

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # ------------------------------------------------
            # PRINT EVERY OUTPUT
            # ------------------------------------------------

            print()
            print("=" * 100)
            print(
                f"[{i+1}/{len(records)}] {pid}"
            )
            print(
                f"Intervention: {condition_name} | "
                f"alpha={ALPHA}"
            )
            print("=" * 100)

            print()
            print("#################### NATURAL / BEFORE ABLATION ####################")
            print(
                f"Safety: {natural_label}"
            )
            print()
            print(natural_response)
            print()
            print("######################### ABLATED #########################")
            print(
                f"Safety: {label}"
            )
            print()
            print(response)
            print()
            print("####################################################################")

            if (
                (i + 1) % 25 == 0
                or i == 0
            ):

                elapsed = (
                    time.time() - t0
                )

                rate = (
                    (i + 1)
                    / max(elapsed, 1e-9)
                )

                eta = (
                    len(records) - i - 1
                ) / max(rate, 1e-9)

                print(
                    f"\nProgress: "
                    f"{i+1}/{len(records)} | "
                    f"refusal={n_refusal} | "
                    f"compliance={n_compliance} | "
                    f"ETA={eta/60:.1f} min",
                    flush=True,
                )

    refusal_rate = (
        n_refusal / len(records)
    )

    compliance_rate = (
        n_compliance / len(records)
    )

    summary = {
        "condition": condition_name,
        "layers": layers,
        "alpha": ALPHA,
        "n": len(records),
        "refusals": n_refusal,
        "compliances": n_compliance,
        "refusal_rate": refusal_rate,
        "compliance_rate": compliance_rate,
        "output_file": str(output_file),
    }

    print()
    print("=" * 80)
    print(
        f"{condition_name} COMPLETE"
    )
    print("=" * 80)
    print(
        f"Refusals:    "
        f"{n_refusal}/{len(records)} "
        f"({100*refusal_rate:.1f}%)"
    )
    print(
        f"Compliances: "
        f"{n_compliance}/{len(records)} "
        f"({100*compliance_rate:.1f}%)"
    )
    print(
        f"Saved: {output_file}"
    )

    return results, summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CAUSAL TEST: A-CONDITION HARM-ASSOCIATED DIRECTION"
    )
    print("=" * 80)

    print()
    print(
        "Population: 512 stable A->C refusals"
    )
    print(
        "Direction: stable harmful - harmless"
    )
    print(
        "Intervention: directional ablation"
    )
    print(
        f"Alpha: {ALPHA}"
    )
    print(
        "Layers: L9, L11, L9+L11"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    records = load_stable_records()

    direction = load_direction()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading Llama 3.1 8B (4-bit)...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            device_map="cuda",
        )
    )

    model.eval()

    print(
        f"VRAM: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # --------------------------------------------------------
    # Load safety judge
    # --------------------------------------------------------

    print()
    print("Loading LlamaGuard...")

    judge = SafetyJudge(
        model_path=GUARD_PATH
    )

    judge.load()

    print(
        f"VRAM after judge: "
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    # --------------------------------------------------------
    # Run all interventions
    # --------------------------------------------------------

    all_results = []
    summaries = []

    for condition_name, layers in TARGET_LAYERS.items():

        results, summary = run_condition(
            model=model,
            tokenizer=tokenizer,
            judge=judge,
            records=records,
            direction=direction,
            condition_name=condition_name,
            layers=layers,
        )

        all_results.extend(results)
        summaries.append(summary)

    # --------------------------------------------------------
    # Save combined results
    # --------------------------------------------------------

    combined_file = (
        OUT_DIR / "all_results.jsonl"
    )

    with open(
        combined_file,
        "w",
    ) as f:

        for result in all_results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary_file = (
        OUT_DIR / "summary.json"
    )

    summary = {
        "seed": SEED,
        "alpha": ALPHA,
        "population": "A->C stable refusals",
        "n_prompts": len(records),
        "direction": str(
            DIRECTION_FILE
        ),
        "direction_definition": (
            "mean stable A-condition harmful "
            "activation minus mean harmless Alpaca "
            "activation"
        ),
        "intervention": (
            "h <- h - alpha * "
            "<h,r_hat> * r_hat"
        ),
        "activation_location": (
            "residual stream entering transformer "
            "block, pre-RMSNorm"
        ),
        "conditions": summaries,
    }

    with open(
        summary_file,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ALL CAUSAL TESTS COMPLETE")
    print("=" * 80)

    print()
    print(
        f"{'Condition':<12} "
        f"{'Refusal':>10} "
        f"{'Compliance':>12} "
        f"{'Refusal %':>12}"
    )

    print("-" * 50)

    for s in summaries:

        print(
            f"{s['condition']:<12} "
            f"{s['refusals']:>10} "
            f"{s['compliances']:>12} "
            f"{100*s['refusal_rate']:>11.1f}%"
        )

    print()
    print(
        f"Combined results: {combined_file}"
    )
    print(
        f"Summary:          {summary_file}"
    )


if __name__ == "__main__":
    main()
