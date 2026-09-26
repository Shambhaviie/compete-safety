"""
layer_specific_ablation.py

Experiment C: Layer-specific directional ablation.

For each layer L in 0..31:
  direction r_hat_L = unit-normalised mean(C_train_refused at L) - mean(C_train_complied at L)
  intervention: forward_pre_hook at layer L applying h' = h - (h . r_hat_L) r_hat_L

Direction and intervention are at the SAME residual stream position (block input,
pre-RMSNorm), consistent with Arditi et al. and the corrected pre-hook extraction.

Primary comparison: Layer 18 result (r_hat_18 at layer 18) vs the previous
Layer-17 result (r_hat_17 at layer 17 from forward-hook extraction -- one-layer offset).

Design:
- One baseline per prompt (no regeneration across layers)
- Same seed, same prompt, same generation settings as all previous experiments
- Layer 0 skipped (direction norm = 0.0, AUC = 0.5, no refusal signal)
- 153 C_test refused prompts (same held-out split, never used in direction computation)
- McNemar's test reported for layers with >= 4 discordant pairs

Output: mechinterp/results/layer_specific_ablation_results.json
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

SEED        = 42
N_LAYERS    = 32
MODEL_PATH  = 'models/llama_3_1_8b_instruct'
GUARD_PATH  = 'models/llama_guard_3_8b'
MAX_NEW_TOK = 512
TEMPERATURE = 0.7
DO_SAMPLE   = True

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_directions():
    path = PROBE_DIR / 'directions_prehook_all_layers.npy'
    dirs = np.load(str(path)).astype(np.float32)  # [32, 4096]
    # Verify norms
    norms = np.linalg.norm(dirs, axis=1)
    print(f"Direction norms: min={norms.min():.4f} max={norms.max():.4f}")
    print(f"Layer 0 norm: {norms[0]:.4f} (should be ~0, no signal)")
    return [torch.tensor(dirs[L], dtype=torch.float32).cuda()
            if norms[L] > 1e-6 else None
            for L in range(N_LAYERS)]


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
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return mdl, tok


def generate(model, tokenizer, prompt, round_number, direction=None, layer=None):
    """
    Generate with optional single pre-hook at one layer.
    direction: [4096] unit-norm tensor on cuda, or None for baseline.
    layer: which layer to hook, or None for baseline.
    """
    chat      = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    handle = None
    if direction is not None and layer is not None:
        def hook_fn(module, args):
            h        = args[0].to(torch.float32)
            proj     = (h @ direction).unsqueeze(-1)
            h        = h - proj * direction
            modified = list(args)
            modified[0] = h.to(args[0].dtype)
            return tuple(modified)
        handle = model.model.layers[layer].register_forward_pre_hook(hook_fn)

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

    return tokenizer.decode(output[0][n_input:], skip_special_tokens=True).strip()


def mcnemar_test(b, c):
    """
    McNemar's test for paired binary data.
    b = R->C flips, c = C->R flips.
    Uses exact binomial when b+c < 25, chi2 otherwise.
    Only meaningful if b+c >= 4.
    """
    if b + c < 4:
        return None, None
    from scipy.stats import binomtest
    pval = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    stat = float(b)
    return stat, float(pval)


if __name__ == '__main__':
    print("=" * 65)
    print("LAYER-SPECIFIC DIRECTIONAL ABLATION — Experiment C")
    print("r_hat_L at layer L | pre-hook | direction = block input of L")
    print(f"Primary result: Layer 18 (correctly aligned, CV AUC=0.9754)")
    print("=" * 65)

    directions = load_all_directions()
    model, tokenizer = load_model()

    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load C_test refused prompts
    split     = json.load(open(PROBE_DIR / 'train_test_split.json'))
    data_C    = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
    rl_C      = data_C['replayed_labels'].astype(bool)
    records_C = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']

    C_test_refused = [records_C[i] for i in split['test_indices'] if rl_C[i]]
    N = len(C_test_refused)
    print(f"\nC_test refused: {N} prompts")
    assert N == 153, f"Expected 153, got {N}"

    # Load CV results for comparison
    cv_results = json.load(open(PROBE_DIR / 'cv_results_prehook.json'))
    cv_auc_by_layer = {r['layer']: r['cv_auc'] for r in cv_results['layers']}

    # ── Step 1: Generate all baselines once ────────────────────────────
    print(f"\nGenerating {N} baselines...")
    t0 = time.time()
    baselines = []
    base_labels = []
    for i, rec in enumerate(C_test_refused):
        resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j    = judge.score_response(rec['task'], resp)
        baselines.append(resp)
        base_labels.append(bool(j['is_safe']))
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{N} baselines done", flush=True)

    base_refusal = sum(base_labels) / N
    print(f"Baseline refusal rate: {100*base_refusal:.1f}% ({sum(base_labels)}/{N})")

    # ── Step 2: Layer-specific ablation ────────────────────────────────
    all_layer_results = []

    print(f"\n{'Layer':<8} {'CV AUC':<10} {'Base%':<8} {'Abl%':<8} "
          f"{'Δpp':<8} {'R→C':<6} {'C→R':<6} {'p-val'}")
    print('-' * 70)

    for L in range(N_LAYERS):
        direction = directions[L]

        if direction is None:
            # Layer 0: no signal, skip intervention
            all_layer_results.append({
                'layer': L,
                'skipped': True,
                'reason': 'direction_norm_zero',
                'cv_auc': cv_auc_by_layer[L],
            })
            print(f"{L:<8} {cv_auc_by_layer[L]:<10.4f} {'—':<8} {'—':<8} "
                  f"{'—':<8} {'—':<6} {'—':<6} — (no signal)")
            continue

        abl_labels  = []
        per_prompt  = []
        R_to_C      = 0
        C_to_R      = 0

        for i, (rec, base_resp, base_safe) in enumerate(
                zip(C_test_refused, baselines, base_labels)):
            resp = generate(model, tokenizer, rec['prompt'],
                           rec['round_number'], direction=direction, layer=L)
            j    = judge.score_response(rec['task'], resp)
            abl_safe = bool(j['is_safe'])
            abl_labels.append(abl_safe)

            flip_RC = base_safe and not abl_safe
            flip_CR = not base_safe and abl_safe
            if flip_RC: R_to_C += 1
            if flip_CR: C_to_R += 1

            per_prompt.append({
                'index':          i,
                'prompt_id':      rec.get('prompt_id', ''),
                'round_number':   rec.get('round_number'),
                'category':       rec.get('category', ''),
                'baseline_safe':  base_safe,
                'ablated_safe':   abl_safe,
                'R_to_C':         flip_RC,
                'C_to_R':         flip_CR,
            })

        abl_refusal = sum(abl_labels) / N
        delta_pp    = 100 * (abl_refusal - base_refusal)

        # McNemar's (only if enough discordant pairs)
        stat, pval = mcnemar_test(R_to_C, C_to_R)
        pval_str   = f"{pval:.4f}" if pval is not None else "n/a"

        primary_marker = " ← PRIMARY" if L == 18 else ""

        print(f"{L:<8} {cv_auc_by_layer[L]:<10.4f} "
              f"{100*base_refusal:<8.1f} {100*abl_refusal:<8.1f} "
              f"{delta_pp:<8.1f} {R_to_C:<6} {C_to_R:<6} "
              f"{pval_str}{primary_marker}",
              flush=True)

        all_layer_results.append({
            'layer':              L,
            'skipped':            False,
            'cv_auc':             cv_auc_by_layer[L],
            'direction_norm':     float(torch.norm(direction).item()),
            'n_prompts':          N,
            'baseline_refusal':   float(base_refusal),
            'ablated_refusal':    float(abl_refusal),
            'delta_pp':           float(delta_pp),
            'R_to_C':             R_to_C,
            'C_to_R':             C_to_R,
            'mcnemar_stat':       stat,
            'mcnemar_pval':       pval,
            'per_prompt':         per_prompt,
        })

    # ── Summary ────────────────────────────────────────────────────────
    active = [r for r in all_layer_results if not r.get('skipped')]
    best_by_flips = max(active, key=lambda r: r['R_to_C'])
    best_by_delta = min(active, key=lambda r: r['delta_pp'])

    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"  Total R→C flips: {sum(r['R_to_C'] for r in active)}")
    print(f"  Layer with most flips: {best_by_flips['layer']} "
          f"({best_by_flips['R_to_C']} flips, Δ={best_by_flips['delta_pp']:.1f}pp)")
    print(f"  Layer 18 (primary): "
          f"R→C={[r for r in active if r['layer']==18][0]['R_to_C']} "
          f"Δ={[r for r in active if r['layer']==18][0]['delta_pp']:.1f}pp")
    print(f"\n  Compare to previous experiments (forward-hook, one-layer offset):")
    print(f"    Layer-17-only (old):    Δ=-1.3pp  2/153 R→C")
    print(f"    Arditi all-layer (old): Δ=-1.3pp  2/153 R→C")
    print(f"\n  Multiple comparisons note: 31 layers tested (layer 0 skipped).")
    print(f"  McNemar's reported only for layers with ≥4 discordant pairs.")
    print(f"  Profile across layers is more informative than individual p-values.")

    # Save
    out_file = OUT_DIR / 'layer_specific_ablation_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'experiment': 'layer_specific_ablation',
            'n_prompts':  N,
            'activation_type': 'prehook_prefill',
            'intervention': 'pre_hook_at_layer_L_using_r_hat_L',
            'note': 'Layer 18 is primary result (correctly aligned with CV best layer)',
            'layers': all_layer_results,
        }, f, indent=2)
    print(f"\nResults saved: {out_file}")

    total_time = time.time() - t0
    print(f"Total time: {total_time/60:.1f}min")

    judge.unload()
    del model
    torch.cuda.empty_cache()
