"""
intervention_v2.py

Two experiments extending the initial causal intervention:

EXPERIMENT 1 — Full Arditi-style directional ablation
Apply h' = h - (h^T r_hat) r_hat at ALL 32 layers simultaneously,
all token positions, every generation step.
Compare to our previous Layer-17-only ablation.

EXPERIMENT 2 — Random vector control for direction addition
Test whether the addition effect is specific to the learned direction
or a generic effect of residual stream perturbation.
Construct a random unit-norm control vector u (same dim, same seed).
Run addition with learned direction and random vector at same alphas.
If random vector produces similar refusal rates, the effect is not
direction-specific.

Both experiments use the same held-out C_test prompts as before.
Same generation settings, seeds, and LlamaGuard judging.
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
SEED            = 42
N_LAYERS        = 32
MODEL_PATH      = 'models/llama_3_1_8b_instruct'
GUARD_PATH      = 'models/llama_guard_3_8b'
MAX_NEW_TOK     = 512
TEMPERATURE     = 0.7
DO_SAMPLE       = True
DIRECTION_LAYER = 17
ADDITION_ALPHAS = [0.5, 1.0, 2.0, 5.0]

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_direction():
    path = PROBE_DIR / f'direction_prefill_acts_layer{DIRECTION_LAYER}.npy'
    d    = np.load(str(path)).astype(np.float32)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-4, "Direction not unit norm"
    return torch.tensor(d, dtype=torch.float32).cuda()


def make_random_control(direction_tensor, seed=999):
    """
    Random unit-norm control vector.
    Same dimensionality as learned direction.
    Orthogonalised to the learned direction to ensure
    it does not accidentally capture refusal information.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    u = torch.randn(direction_tensor.shape[0], generator=rng,
                    dtype=torch.float32).cuda()
    # Orthogonalise against learned direction
    u = u - (u @ direction_tensor) * direction_tensor
    u = u / u.norm()
    return u


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


def make_full_ablation_hooks(direction, n_layers=N_LAYERS):
    """
    Experiment 1: Arditi-style full ablation.
    Returns list of hook functions for all layers.
    h' = h - (h^T r_hat) r_hat at every layer.
    """
    def make_hook():
        def hook_fn(module, inp, out):
            h    = out[0].to(torch.float32)
            proj = (h @ direction).unsqueeze(-1)
            h    = h - proj * direction
            modified    = list(out)
            modified[0] = h.to(out[0].dtype)
            return tuple(modified)
        return hook_fn
    return [make_hook() for _ in range(n_layers)]


def make_addition_hook(vector, alpha):
    """
    h' = h + alpha * vector
    Works for both learned direction and random control.
    """
    def hook_fn(module, inp, out):
        h        = out[0].to(torch.float32)
        h        = h + alpha * vector
        modified    = list(out)
        modified[0] = h.to(out[0].dtype)
        return tuple(modified)
    return hook_fn


def generate(model, tokenizer, prompt, round_number,
             layer_hooks=None):
    """
    layer_hooks: dict {layer_idx: hook_fn} or None for baseline.
    Applies hooks at specified layers during generation.
    """
    chat      = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)

    handles = []
    if layer_hooks:
        for layer_idx, hook_fn in layer_hooks.items():
            h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            handles.append(h)

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

    for h in handles:
        h.remove()

    n_input = input_ids.shape[1]
    return tokenizer.decode(
        output[0][n_input:], skip_special_tokens=True
    ).strip()


# ── Experiment 1: Full Arditi-style ablation ───────────────────────────────

def run_full_ablation(model, tokenizer, judge, records, direction):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Full directional ablation (all {N_LAYERS} layers)")
    print(f"{'='*60}")
    print(f"n_prompts={len(records)}")

    hook_fns     = make_full_ablation_hooks(direction)
    results      = []
    base_refused = 0
    abl_refused  = 0

    for i, rec in enumerate(records):
        # Baseline
        resp_base = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j_base    = judge.score_response(rec['task'], resp_base)
        base_safe = bool(j_base['is_safe'])

        # Full ablation -- hooks on all 32 layers
        layer_hooks = {li: hook_fns[li] for li in range(N_LAYERS)}
        resp_abl  = generate(model, tokenizer, rec['prompt'],
                             rec['round_number'], layer_hooks=layer_hooks)
        j_abl     = judge.score_response(rec['task'], resp_abl)
        abl_safe  = bool(j_abl['is_safe'])

        if base_safe: base_refused += 1
        if abl_safe:  abl_refused  += 1

        results.append({
            'prompt_id':     rec.get('prompt_id', ''),
            'round_number':  rec.get('round_number'),
            'category':      rec.get('category', ''),
            'baseline_safe': base_safe,
            'ablated_safe':  abl_safe,
            'R_to_C':        base_safe and not abl_safe,
            'C_to_R':        not base_safe and abl_safe,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1:>4}/{len(records)} | "
                  f"baseline={100*base_refused/(i+1):.0f}% | "
                  f"ablated={100*abl_refused/(i+1):.0f}%",
                  flush=True)

    N        = len(records)
    R_to_C   = sum(1 for r in results if r['R_to_C'])
    C_to_R   = sum(1 for r in results if r['C_to_R'])

    print(f"\n  Baseline refusal:  {100*base_refused/N:.1f}%")
    print(f"  Ablated refusal:   {100*abl_refused/N:.1f}%")
    print(f"  Delta:             {100*(abl_refused-base_refused)/N:.1f}pp")
    print(f"  R→C flips:         {R_to_C}/{N} ({100*R_to_C/N:.1f}%)")
    print(f"  C→R flips:         {C_to_R}/{N} ({100*C_to_R/N:.1f}%)")
    print(f"\n  Compare to Layer-17-only ablation: Δ=-1.3pp (2/153 flipped)")

    return {
        'baseline_refusal': base_refused/N,
        'ablated_refusal':  abl_refused/N,
        'delta':            (abl_refused-base_refused)/N,
        'R_to_C':           R_to_C,
        'C_to_R':           C_to_R,
        'records':          results,
    }


# ── Experiment 2: Random vector control ───────────────────────────────────

def run_addition_with_control(model, tokenizer, judge, records,
                               direction, random_vec):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Direction addition vs random control")
    print(f"{'='*60}")
    print(f"n_prompts={len(records)} | alphas={ADDITION_ALPHAS}")
    print(f"Intervention layer: {DIRECTION_LAYER}")

    # Baseline pass first
    baselines = []
    base_refused = 0
    print("\n  Running baselines...")
    for rec in records:
        resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j    = judge.score_response(rec['task'], resp)
        safe = bool(j['is_safe'])
        baselines.append(safe)
        if safe: base_refused += 1
    print(f"  Baseline refusal: {100*base_refused/len(records):.1f}%")

    learned_results = {}
    random_results  = {}

    print(f"\n  {'Alpha':<8} {'Learned':>10} {'Random':>10} {'Diff':>8}")
    print(f"  {'─'*40}")

    for alpha in ADDITION_ALPHAS:
        learned_refused = 0
        random_refused  = 0
        learned_recs    = []
        random_recs     = []

        for i, (rec, base_safe) in enumerate(zip(records, baselines)):
            # Learned direction addition
            hook_l   = {DIRECTION_LAYER: make_addition_hook(direction, alpha)}
            resp_l   = generate(model, tokenizer, rec['prompt'],
                                rec['round_number'], layer_hooks=hook_l)
            j_l      = judge.score_response(rec['task'], resp_l)
            safe_l   = bool(j_l['is_safe'])
            if safe_l: learned_refused += 1

            # Random vector addition
            hook_r   = {DIRECTION_LAYER: make_addition_hook(random_vec, alpha)}
            resp_r   = generate(model, tokenizer, rec['prompt'],
                                rec['round_number'], layer_hooks=hook_r)
            j_r      = judge.score_response(rec['task'], resp_r)
            safe_r   = bool(j_r['is_safe'])
            if safe_r: random_refused += 1

            learned_recs.append({
                'prompt_id':    rec.get('prompt_id',''),
                'round_number': rec.get('round_number'),
                'category':     rec.get('category',''),
                'alpha':        alpha,
                'baseline_safe': base_safe,
                'learned_safe': safe_l,
                'random_safe':  safe_r,
                'learned_flipped': not base_safe and safe_l,
                'random_flipped':  not base_safe and safe_r,
            })

        N = len(records)
        l_rate = 100 * learned_refused / N
        r_rate = 100 * random_refused  / N
        diff   = l_rate - r_rate
        print(f"  {alpha:<8.1f} {l_rate:>9.1f}% {r_rate:>9.1f}% {diff:>+7.1f}pp")

        learned_results[str(alpha)] = {
            'refusal_rate': learned_refused/N,
            'records': learned_recs,
        }
        random_results[str(alpha)] = {
            'refusal_rate': random_refused/N,
        }

    return {
        'baseline_refusal':  base_refused/len(records),
        'learned_direction': learned_results,
        'random_direction':  random_results,
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("INTERVENTION v2: Full ablation + Random control")
    print("=" * 60)

    direction  = load_direction()
    random_vec = make_random_control(direction)

    # Verify orthogonality
    dot = (direction @ random_vec).item()
    print(f"\nLearned direction norm:  {direction.norm().item():.4f}")
    print(f"Random vector norm:      {random_vec.norm().item():.4f}")
    print(f"Dot product (should≈0):  {dot:.6f}")

    model, tokenizer = load_model()
    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load C_test records
    split     = json.load(open(PROBE_DIR / 'train_test_split.json'))
    data_C    = np.load(str(ACT_DIR / 'activations_C_seed42.npz'))
    rl_C      = data_C['replayed_labels'].astype(bool)
    records_C = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']

    C_test_refused  = [records_C[i] for i in split['test_indices'] if rl_C[i]]
    C_test_complied = [records_C[i] for i in split['test_indices'] if not rl_C[i]]
    print(f"\nC_test refused:  {len(C_test_refused)}")
    print(f"C_test complied: {len(C_test_complied)}")

    all_results = {}

    # Experiment 1
    all_results['exp1_full_ablation'] = run_full_ablation(
        model, tokenizer, judge, C_test_refused, direction
    )

    # Experiment 2
    all_results['exp2_addition_control'] = run_addition_with_control(
        model, tokenizer, judge, C_test_complied, direction, random_vec
    )

    # Save
    out_file = OUT_DIR / 'intervention_v2_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    exp1 = all_results['exp1_full_ablation']
    print(f"\nExp 1 — Full ablation (all layers):")
    print(f"  Baseline: {100*exp1['baseline_refusal']:.1f}% → "
          f"Ablated: {100*exp1['ablated_refusal']:.1f}% "
          f"(Δ={100*exp1['delta']:.1f}pp)")
    print(f"  R→C flips: {exp1['R_to_C']} | C→R: {exp1['C_to_R']}")
    print(f"  Layer-17-only was Δ=-1.3pp — comparison above shows full-model effect")

    exp2 = all_results['exp2_addition_control']
    print(f"\nExp 2 — Addition: learned vs random control:")
    print(f"  {'Alpha':<8} {'Learned':>10} {'Random':>10} {'Diff':>8}")
    for alpha in ADDITION_ALPHAS:
        l = 100 * exp2['learned_direction'][str(alpha)]['refusal_rate']
        r = 100 * exp2['random_direction'][str(alpha)]['refusal_rate']
        print(f"  {alpha:<8.1f} {l:>9.1f}% {r:>9.1f}% {l-r:>+7.1f}pp")

    print(f"\nResults saved: {out_file}")

    judge.unload()
    del model
    torch.cuda.empty_cache()
