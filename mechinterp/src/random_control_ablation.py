"""
random_control_ablation.py

Random-direction control for the layer-specific ablation experiment.

Tests whether the large causal effects at layers 9 and 11 are specific
to the learned refusal-associated direction or a generic effect of
perturbing the residual stream at those layers.

For each target layer L in {9, 11}:
  Learned direction:  r_hat_L (from directions_prehook_all_layers.npy)
  Random control:     u_L = random unit vector, orthogonalised against r_hat_L
  
  Apply h' = h - (h . v) v  where v is learned or random
  via pre-hook at layer L, all token positions, all generation steps.

Paired design:
  Same 153 C_test refused prompts
  Same seeds
  Baseline (no hook) -- regenerated fresh for this experiment
  Learned ablation
  Random ablation

Reports:
  For each layer: baseline / learned / random refusal rates and flip counts
  Difference: learned - random (direction-specific effect)
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

SEED           = 42
N_LAYERS       = 32
MODEL_PATH     = 'models/llama_3_1_8b_instruct'
GUARD_PATH     = 'models/llama_guard_3_8b'
MAX_NEW_TOK    = 512
TEMPERATURE    = 0.7
DO_SAMPLE      = True
TARGET_LAYERS  = [9, 11, 10, 8, 14, 7, 13, 3, 12, 17, 4, 5, 6, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1, 2]   # peak causal layers from layer_specific_ablation
RANDOM_SEED    = 999       # fixed seed for random direction construction

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_random_control(learned_direction, layer, seed=RANDOM_SEED):
    """
    Random unit vector orthogonalised against the learned direction.
    Orthogonalisation ensures the random vector does not accidentally
    capture refusal-associated information from the learned direction.
    """
    rng = torch.Generator()
    rng.manual_seed(seed + layer)   # different random vector per layer
    u = torch.randn(learned_direction.shape[0], generator=rng,
                    dtype=torch.float32).cuda()
    u = u - (u @ learned_direction) * learned_direction
    u = u / u.norm()
    dot = (u @ learned_direction).item()
    assert abs(dot) < 1e-5, f"Orthogonalisation failed: dot={dot}"
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


def generate(model, tokenizer, prompt, round_number, direction=None, layer=None):
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


if __name__ == '__main__':
    print("=" * 65)
    print("RANDOM-DIRECTION CONTROL ABLATION")
    print(f"Target layers: {TARGET_LAYERS}")
    print("Learned direction vs orthogonal random vector")
    print("=" * 65)

    # Load learned directions
    all_dirs = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)
    learned  = {L: torch.tensor(all_dirs[L], dtype=torch.float32).cuda()
                for L in TARGET_LAYERS}
    random   = {L: make_random_control(learned[L], L) for L in TARGET_LAYERS}

    for L in TARGET_LAYERS:
        dot  = (learned[L] @ random[L]).item()
        print(f"Layer {L}: learned norm={learned[L].norm():.4f} | "
              f"random norm={random[L].norm():.4f} | "
              f"dot product={dot:.6f} (should be ~0)")

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
    assert N == 153

    all_results = {}
    t0 = time.time()

    for L in TARGET_LAYERS:
        print(f"\n{'='*55}")
        print(f"LAYER {L}")
        print(f"{'='*55}")

        base_labels    = []
        learned_labels = []
        random_labels  = []
        per_prompt     = []

        for i, rec in enumerate(C_test_refused):
            # Baseline
            resp_base = generate(model, tokenizer, rec['prompt'], rec['round_number'])
            j_base    = judge.score_response(rec['task'], resp_base)
            base_safe = bool(j_base['is_safe'])

            # Learned direction ablation
            resp_l = generate(model, tokenizer, rec['prompt'], rec['round_number'],
                              direction=learned[L], layer=L)
            j_l    = judge.score_response(rec['task'], resp_l)
            l_safe = bool(j_l['is_safe'])

            # Random direction ablation
            resp_r = generate(model, tokenizer, rec['prompt'], rec['round_number'],
                              direction=random[L], layer=L)
            j_r    = judge.score_response(rec['task'], resp_r)
            r_safe = bool(j_r['is_safe'])

            base_labels.append(base_safe)
            learned_labels.append(l_safe)
            random_labels.append(r_safe)

            per_prompt.append({
                'prompt_id':      rec.get('prompt_id', ''),
                'round_number':   rec.get('round_number'),
                'category':       rec.get('category', ''),
                'task':           rec.get('task', '')[:150],
                'baseline_safe':  base_safe,
                'learned_safe':   l_safe,
                'random_safe':    r_safe,
                'learned_R_to_C': base_safe and not l_safe,
                'random_R_to_C':  base_safe and not r_safe,
            })

            if (i + 1) % 25 == 0 or i == 0:
                l_ref = sum(learned_labels) / (i+1)
                r_ref = sum(random_labels)  / (i+1)
                print(f"  {i+1:>4}/{N} | "
                      f"learned_refusal={100*l_ref:.0f}% | "
                      f"random_refusal={100*r_ref:.0f}%",
                      flush=True)

        base_ref    = sum(base_labels)    / N
        learned_ref = sum(learned_labels) / N
        random_ref  = sum(random_labels)  / N
        learned_RC  = sum(1 for p in per_prompt if p['learned_R_to_C'])
        random_RC   = sum(1 for p in per_prompt if p['random_R_to_C'])

        print(f"\n  Layer {L} results:")
        print(f"  {'':20} {'Refusal%':>10} {'R→C flips':>12} {'Δpp vs baseline':>16}")
        print(f"  {'Baseline':20} {100*base_ref:>10.1f} {'—':>12} {'—':>16}")
        print(f"  {'Learned direction':20} {100*learned_ref:>10.1f} {learned_RC:>12} {100*(learned_ref-base_ref):>+16.1f}")
        print(f"  {'Random control':20} {100*random_ref:>10.1f} {random_RC:>12} {100*(random_ref-base_ref):>+16.1f}")
        print(f"  {'Specific effect':20} {'':>10} {learned_RC-random_RC:>12} {100*(learned_ref-random_ref):>+16.1f}pp")

        all_results[str(L)] = {
            'layer':             L,
            'n_prompts':         N,
            'baseline_refusal':  float(base_ref),
            'learned_refusal':   float(learned_ref),
            'random_refusal':    float(random_ref),
            'learned_R_to_C':    learned_RC,
            'random_R_to_C':     random_RC,
            'specific_effect_pp': float(100*(learned_ref-random_ref)),
            'specific_flips':    learned_RC - random_RC,
            'per_prompt':        per_prompt,
        }

    # Save
    out_file = OUT_DIR / 'random_control_ablation_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'Layer':<8} {'Learned R→C':<14} {'Random R→C':<14} {'Specific flips':<16} {'Direction-specific?'}")
    print('-'*70)
    for L in TARGET_LAYERS:
        r = all_results[str(L)]
        specific = r['specific_flips']
        verdict  = 'YES' if specific > 5 else ('WEAK' if specific > 0 else 'NO')
        print(f"{L:<8} {r['learned_R_to_C']:<14} {r['random_R_to_C']:<14} {specific:<16} {verdict}")

    print(f"\nResults saved: {out_file}")
    print(f"Total time: {(time.time()-t0)/60:.1f}min")

    judge.unload()
    del model
    torch.cuda.empty_cache()
