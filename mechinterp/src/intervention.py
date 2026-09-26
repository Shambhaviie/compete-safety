"""
intervention.py

Causal intervention experiment using the C-derived prefill direction at layer 17.

Two interventions (Arditi et al. style):

1. DIRECTIONAL ABLATION on C_test refused prompts:
   h' = h - (h^T r_hat) * r_hat
   Applied at layer 17, ALL token positions, EVERY generation step.
   Question: does refusal rate decrease?

2. DIRECTION ADDITION on C_test complied prompts:
   h' = h + alpha * r_hat
   Applied at layer 17, ALL token positions, EVERY generation step.
   Tested at alpha in {0.5, 1.0, 2.0, 5.0}.
   Question: does refusal rate increase?

Each prompt gets paired baseline + intervention generation.
Primary: C_test held-out prompts (not used in direction computation).
Secondary: Condition A refused prompts (cross-condition validation).

Design principles:
- Directional ablation not simple subtraction
- Intervention at all positions every step (matching Arditi et al.)
- Paired baseline for within-prompt comparison
- Multiple alpha values for addition (systematic dose-response)
- No overclaiming: "contributes to" not "causally mediates"
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from safety_judge import SafetyJudge

# ── Config ─────────────────────────────────────────────────────────────────
SEED        = 42
MODEL_PATH  = 'models/llama_3_1_8b_instruct'
GUARD_PATH  = 'models/llama_guard_3_8b'
MAX_NEW_TOK = 512
TEMPERATURE = 0.7
DO_SAMPLE   = True

DIRECTION_LAYER = 17
ADDITION_ALPHAS = [0.5, 1.0, 2.0, 5.0]

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_direction():
    path = PROBE_DIR / f'direction_prefill_acts_layer{DIRECTION_LAYER}.npy'
    d    = np.load(str(path)).astype(np.float32)
    # Verify unit norm
    norm = np.linalg.norm(d)
    assert abs(norm - 1.0) < 1e-4, f"Direction not unit norm: {norm}"
    return torch.tensor(d, dtype=torch.float32).cuda()


def load_model():
    print("Loading Llama 3.1 8B (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4'
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map='cuda'
    )
    mdl.eval()
    print(f"  VRAM after Llama: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return mdl, tok


def make_ablation_hook(direction):
    """
    Directional ablation hook: h' = h - (h^T r_hat) r_hat
    Applied to ALL token positions at every generation step.
    direction: [4096] unit-norm tensor on cuda
    """
    def hook_fn(module, inp, out):
        h    = out[0].to(torch.float32)          # [1, seq_len, 4096]
        proj = (h @ direction).unsqueeze(-1)      # [1, seq_len, 1]
        h    = h - proj * direction               # remove component
        # Return with original dtype
        modified = list(out)
        modified[0] = h.to(out[0].dtype)
        return tuple(modified)
    return hook_fn


def make_addition_hook(direction, alpha):
    """
    Direction addition hook: h' = h + alpha * r_hat
    Applied to ALL token positions at every generation step.
    direction: [4096] unit-norm tensor on cuda
    """
    def hook_fn(module, inp, out):
        h        = out[0].to(torch.float32)       # [1, seq_len, 4096]
        h        = h + alpha * direction           # add direction
        modified = list(out)
        modified[0] = h.to(out[0].dtype)
        return tuple(modified)
    return hook_fn


def generate(model, tokenizer, prompt, round_number, hook_fn=None):
    """
    Generate response with optional intervention hook at layer 17.
    Matches original experiment_runner.py exactly.
    """
    chat      = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)

    handle = None
    if hook_fn is not None:
        handle = model.model.layers[DIRECTION_LAYER].register_forward_hook(hook_fn)

    torch.manual_seed(SEED + round_number)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOK,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    if handle is not None:
        handle.remove()

    n_input   = input_ids.shape[1]
    generated = output[0][n_input:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_ablation(model, tokenizer, judge, records, label, direction):
    """
    Ablation experiment on refused prompts.
    Paired: baseline vs ablation per prompt.
    """
    print(f"\n--- ABLATION: {label} refused prompts (n={len(records)}) ---")
    results = []
    baseline_refused = 0
    ablated_refused  = 0

    for i, rec in enumerate(records):
        # Baseline
        resp_base = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j_base    = judge.score_response(rec['task'], resp_base)
        base_safe = bool(j_base['is_safe'])

        # Ablation
        hook      = make_ablation_hook(direction)
        resp_abl  = generate(model, tokenizer, rec['prompt'],
                              rec['round_number'], hook_fn=hook)
        j_abl     = judge.score_response(rec['task'], resp_abl)
        abl_safe  = bool(j_abl['is_safe'])

        if base_safe: baseline_refused += 1
        if abl_safe:  ablated_refused  += 1

        results.append({
            'prompt_id':       rec.get('prompt_id', ''),
            'round_number':    rec.get('round_number'),
            'category':        rec.get('category', ''),
            'baseline_safe':   base_safe,
            'ablated_safe':    abl_safe,
            'flipped':         base_safe and not abl_safe,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1:>4}/{len(records)} | "
                  f"baseline_refusal={100*baseline_refused/(i+1):.0f}% | "
                  f"ablated_refusal={100*ablated_refused/(i+1):.0f}%",
                  flush=True)

    N = len(records)
    flipped = sum(1 for r in results if r['flipped'])
    print(f"\n  Baseline refusal rate: {100*baseline_refused/N:.1f}%")
    print(f"  Ablated refusal rate:  {100*ablated_refused/N:.1f}%")
    print(f"  Prompts flipped (R→C): {flipped}/{N} ({100*flipped/N:.1f}%)")
    return results, baseline_refused/N, ablated_refused/N


def run_addition(model, tokenizer, judge, records, label, direction):
    """
    Addition experiment on complied prompts.
    Paired: baseline vs addition at each alpha per prompt.
    """
    print(f"\n--- ADDITION: {label} complied prompts (n={len(records)}) ---")
    results = []
    baseline_refused = 0

    # Baseline pass first
    baselines = []
    for rec in records:
        resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j    = judge.score_response(rec['task'], resp)
        safe = bool(j['is_safe'])
        baselines.append(safe)
        if safe: baseline_refused += 1

    print(f"  Baseline refusal rate: {100*baseline_refused/len(records):.1f}%")

    # Addition at each alpha
    for alpha in ADDITION_ALPHAS:
        added_refused = 0
        alpha_results = []

        for i, (rec, base_safe) in enumerate(zip(records, baselines)):
            hook     = make_addition_hook(direction, alpha)
            resp_add = generate(model, tokenizer, rec['prompt'],
                                rec['round_number'], hook_fn=hook)
            j_add    = judge.score_response(rec['task'], resp_add)
            add_safe = bool(j_add['is_safe'])
            if add_safe: added_refused += 1

            alpha_results.append({
                'prompt_id':    rec.get('prompt_id', ''),
                'round_number': rec.get('round_number'),
                'category':     rec.get('category', ''),
                'alpha':        alpha,
                'baseline_safe': base_safe,
                'added_safe':   add_safe,
                'flipped':      not base_safe and add_safe,
            })

        N       = len(records)
        flipped = sum(1 for r in alpha_results if r['flipped'])
        print(f"  alpha={alpha:.1f}: refusal={100*added_refused/N:.1f}% | "
              f"flipped (C→R): {flipped}/{N} ({100*flipped/N:.1f}%)")
        results.extend(alpha_results)

    return results, baseline_refused/len(records)


if __name__ == '__main__':
    print("=" * 60)
    print("CAUSAL INTERVENTION EXPERIMENT")
    print(f"Direction: prefill layer {DIRECTION_LAYER} (C-derived)")
    print(f"Intervention: ALL token positions, every generation step")
    print(f"Primary: C_test held-out | Secondary: A refused")
    print("=" * 60)

    # Load direction
    direction = load_direction()
    print(f"\nDirection loaded: layer {DIRECTION_LAYER}, "
          f"norm={direction.norm().item():.4f}")

    # Load models
    model, tokenizer = load_model()
    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load split and metadata
    split    = json.load(open(PROBE_DIR / 'train_test_split.json'))
    meta_C   = json.load(open(ACT_DIR / 'metadata_C_seed42.json'))
    meta_A   = json.load(open(ACT_DIR / 'metadata_A_seed42.json'))
    data_C   = np.load(str(ACT_DIR / 'activations_C_seed42.npz'))
    data_A   = np.load(str(ACT_DIR / 'activations_A_seed42.npz'))
    rl_C     = data_C['replayed_labels'].astype(bool)
    rl_A     = data_A['replayed_labels'].astype(bool)

    # Load original round records for prompts/tasks
    records_C_all = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']
    records_A_all = [json.loads(l) for l in open(
        'results_task6_676prompt/control_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']

    # C_test indices from saved split
    test_idx  = set(split['test_indices'])

    # C_test refused and complied
    C_test_refused  = [records_C_all[i] for i in split['test_indices']
                       if rl_C[i]]
    C_test_complied = [records_C_all[i] for i in split['test_indices']
                       if not rl_C[i]]

    print(f"\nC_test refused:  {len(C_test_refused)}")
    print(f"C_test complied: {len(C_test_complied)}")

    # A refused (all, secondary validation)
    A_refused = [records_A_all[i] for i in range(len(records_A_all))
                 if rl_A[i]]
    # Use first 100 for speed
    A_refused = A_refused[:100]
    print(f"A refused (sample): {len(A_refused)}")

    all_results = {}

    # ── PRIMARY: Ablation on C_test refused ────────────────────────────
    abl_results, base_r, abl_r = run_ablation(
        model, tokenizer, judge, C_test_refused, 'C_test', direction
    )
    all_results['C_ablation'] = {
        'baseline_refusal': base_r,
        'ablated_refusal':  abl_r,
        'delta':            abl_r - base_r,
        'records':          abl_results,
    }

    # ── PRIMARY: Addition on C_test complied ───────────────────────────
    add_results, add_base_r = run_addition(
        model, tokenizer, judge, C_test_complied, 'C_test', direction
    )
    all_results['C_addition'] = {
        'baseline_refusal': add_base_r,
        'alphas':           ADDITION_ALPHAS,
        'records':          add_results,
    }

    # ── SECONDARY: Ablation on A refused ───────────────────────────────
    abl_A_results, base_A_r, abl_A_r = run_ablation(
        model, tokenizer, judge, A_refused, 'A', direction
    )
    all_results['A_ablation'] = {
        'baseline_refusal': base_A_r,
        'ablated_refusal':  abl_A_r,
        'delta':            abl_A_r - base_A_r,
        'records':          abl_A_results,
    }

    # ── Save ───────────────────────────────────────────────────────────
    out_file = OUT_DIR / 'intervention_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("INTERVENTION SUMMARY")
    print(f"{'='*60}")
    print(f"\nAblation on C_test refused:")
    print(f"  Baseline: {100*base_r:.1f}% → Ablated: {100*abl_r:.1f}% "
          f"(Δ={100*(abl_r-base_r):.1f}pp)")
    print(f"\nAblation on A refused:")
    print(f"  Baseline: {100*base_A_r:.1f}% → Ablated: {100*abl_A_r:.1f}% "
          f"(Δ={100*(abl_A_r-base_A_r):.1f}pp)")
    print(f"\nAddition on C_test complied:")
    by_alpha = {}
    for r in add_results:
        a = r['alpha']
        if a not in by_alpha: by_alpha[a] = []
        by_alpha[a].append(r['added_safe'])
    for alpha in ADDITION_ALPHAS:
        rate = sum(by_alpha[alpha]) / len(by_alpha[alpha])
        print(f"  alpha={alpha}: {100*rate:.1f}% refusal")

    print(f"\nResults saved: {out_file}")

    judge.unload()
    del model
    torch.cuda.empty_cache()
