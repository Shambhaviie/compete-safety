"""
pairwise_ablation.py

NEW EXPERIMENT 6: Pairwise ablation at layers 9 and 11.

Tests whether ablating two layers simultaneously produces
more, fewer, or the same flips as either layer alone.

Conditions:
  - Layer 9 alone  (already known: 60 flips)
  - Layer 11 alone (already known: 57 flips)
  - Layer 9 + 11 simultaneously
  - Layer 9 + 14 simultaneously
  - Layer 9 + 11 + 14 simultaneously

Measures causal interaction:
  interaction = combined_flips - (L9_flips + L11_flips)
  negative = subadditive (interference)
  zero     = independent
  positive = superadditive (synergy)
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

SEED       = 42
MODEL_PATH = 'models/llama_3_1_8b_instruct'
GUARD_PATH = 'models/llama_guard_3_8b'
MAX_NEW_TOK = 512
TEMPERATURE = 0.7
DO_SAMPLE   = True

# Combinations to test
COMBINATIONS = [
    [9],
    [11],
    [14],
    [9, 11],
    [9, 14],
    [9, 11, 14],
]

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    print('Loading Llama 3.1 8B (4-bit)...')
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
    print(f'  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
    return mdl, tok


def generate(model, tokenizer, prompt, round_number, layer_dirs=None):
    """
    Generate with optional simultaneous ablation at multiple layers.
    layer_dirs: dict {layer_idx: direction_tensor}
    """
    chat      = [{'role': 'user', 'content': prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors='pt'
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    handles = []
    if layer_dirs:
        for L, d in layer_dirs.items():
            def make_hook(direction):
                def hook_fn(module, args):
                    h    = args[0].to(torch.float32)
                    proj = (h @ direction).unsqueeze(-1)
                    h    = h - proj * direction
                    mod  = list(args)
                    mod[0] = h.to(args[0].dtype)
                    return tuple(mod)
                return hook_fn
            handles.append(
                model.model.layers[L].register_forward_pre_hook(make_hook(d))
            )

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
    return tokenizer.decode(output[0][n_input:], skip_special_tokens=True).strip()


if __name__ == '__main__':
    print('='*65)
    print('PAIRWISE ABLATION — CAUSAL INTERACTION BETWEEN LAYERS')
    print(f'Combinations: {COMBINATIONS}')
    print('='*65)

    all_dirs = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)
    needed_layers = sorted(set(L for combo in COMBINATIONS for L in combo))
    directions = {L: torch.tensor(all_dirs[L], dtype=torch.float32).cuda()
                  for L in needed_layers}
    print(f'Directions loaded for layers: {needed_layers}')

    model, tokenizer = load_model()
    print('\nLoading LlamaGuard...')
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f'  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)

    # Load C_test refused prompts
    split  = json.load(open(PROBE_DIR / 'train_test_split.json'))
    data_C = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
    rl_C   = data_C['replayed_labels'].astype(bool)
    records_C = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']
    C_test_refused = [records_C[i] for i in split['test_indices'] if rl_C[i]]
    N = len(C_test_refused)
    print(f'\nC_test refused: {N} prompts')

    # Generate baselines once
    print('\nRunning baselines...')
    baselines = []
    for rec in C_test_refused:
        resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j    = judge.score_response(rec['task'], resp)
        baselines.append(bool(j['is_safe']))
    base_rate = sum(baselines) / N
    print(f'Baseline refusal: {100*base_rate:.1f}%', flush=True)

    all_results = {}
    t0 = time.time()

    print(f'\n{"Combination":<20} {"Refusal%":>10} {"Flips":>8} {"Delta":>8}')
    print('-'*50)

    for combo in COMBINATIONS:
        layer_dirs = {L: directions[L] for L in combo}
        abl_labels = []
        flips      = 0

        for rec, base_safe in zip(C_test_refused, baselines):
            resp = generate(model, tokenizer, rec['prompt'],
                           rec['round_number'], layer_dirs=layer_dirs)
            j    = judge.score_response(rec['task'], resp)
            abl_safe = bool(j['is_safe'])
            abl_labels.append(abl_safe)
            if base_safe and not abl_safe:
                flips += 1

        abl_rate = sum(abl_labels) / N
        delta    = 100 * (abl_rate - base_rate)
        label    = '+'.join(f'L{l}' for l in combo)

        print(f'{label:<20} {100*abl_rate:>9.1f}% {flips:>8} {delta:>+7.1f}pp',
              flush=True)

        all_results[label] = {
            'layers':           combo,
            'flips':            flips,
            'ablated_refusal':  float(abl_rate),
            'baseline_refusal': float(base_rate),
            'delta_pp':         float(delta),
        }

        # Checkpoint
        with open(OUT_DIR / 'pairwise_ablation_checkpoint.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Compute interaction effects
    print(f'\n{"="*65}')
    print('INTERACTION ANALYSIS')
    print('interaction = combined_flips - sum(individual_flips)')
    print('negative = subadditive (interference), zero = independent, positive = synergy')
    print()

    def get_flips(layers):
        key = '+'.join(f'L{l}' for l in layers)
        return all_results.get(key, {}).get('flips', 0)

    combos_to_analyse = [
        ([9, 11],     [9], [11]),
        ([9, 14],     [9], [14]),
        ([9, 11, 14], [9], [11, 14]),
    ]
    for combo, *parts in combos_to_analyse:
        combined = get_flips(combo)
        individual_sum = sum(get_flips(p) for p in parts)
        interaction = combined - individual_sum
        label = '+'.join(f'L{l}' for l in combo)
        print(f'{label}: {combined} flips | '
              f'sum of parts: {individual_sum} | '
              f'interaction: {interaction:+d}')

    out_file = OUT_DIR / 'pairwise_ablation_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved: {out_file}')
    print(f'Total time: {(time.time()-t0)/60:.1f}min')

    judge.unload()
    del model
    torch.cuda.empty_cache()
