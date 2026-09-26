"""
addition_peak_layers.py

Direction addition experiment at peak causal layers (7-11) plus layer 18 for comparison.

For each target layer L in {7, 8, 9, 10, 11, 18}:
  h' = h + alpha * r̂_L   at layer L, all token positions, all generation steps

Applied to 48 C_test complied prompts.
Alphas: {0.5, 1.0, 2.0, 5.0}
Random orthogonal control at same alphas and layers.

Comparison point: layer 18 addition already produced:
  alpha=0.5 -> 4.2%
  alpha=1.0 -> 20.8%
  alpha=2.0 -> 54.2%
  alpha=5.0 -> 100%

Question: do layers 7-11 produce stronger refusal induction at lower alpha?
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

SEED          = 42
MODEL_PATH    = 'models/llama_3_1_8b_instruct'
GUARD_PATH    = 'models/llama_guard_3_8b'
MAX_NEW_TOK   = 512
TEMPERATURE   = 0.7
DO_SAMPLE     = True
RANDOM_SEED   = 999
TARGET_LAYERS = [9, 11, 10, 8, 7, 18, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1, 2, 3, 4, 5, 6]
ALPHAS        = [0.5, 1.0, 2.0, 5.0]

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_directions(target_layers):
    all_dirs = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)
    out = {}
    for L in target_layers:
        d = torch.tensor(all_dirs[L], dtype=torch.float32).cuda()
        assert abs(d.norm().item() - 1.0) < 1e-4, f'Layer {L} direction not unit norm'
        out[L] = d
    return out


def make_random_controls(learned, seed=RANDOM_SEED):
    out = {}
    for L, d in learned.items():
        rng = torch.Generator()
        rng.manual_seed(seed + L)
        u = torch.randn(d.shape[0], generator=rng, dtype=torch.float32).cuda()
        u = u - (u @ d) * d
        u = u / u.norm()
        assert abs((u @ d).item()) < 1e-5
        out[L] = u
    return out


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
    print(f'  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    return mdl, tok


def generate(model, tokenizer, prompt, round_number, direction=None,
             layer=None, alpha=None):
    chat      = [{'role': 'user', 'content': prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors='pt'
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    handle = None
    if direction is not None and layer is not None and alpha is not None:
        def hook_fn(module, args):
            h        = args[0].to(torch.float32)
            h        = h + alpha * direction
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

    if handle:
        handle.remove()

    return tokenizer.decode(output[0][n_input:], skip_special_tokens=True).strip()


if __name__ == '__main__':
    print('=' * 65)
    print('DIRECTION ADDITION — PEAK CAUSAL LAYERS vs LAYER 18')
    print(f'Target layers: {TARGET_LAYERS}')
    print(f'Alphas: {ALPHAS}')
    print(f'Prompts: 48 C_test complied')
    print('=' * 65)

    learned  = load_directions(TARGET_LAYERS)
    randoms  = make_random_controls(learned)

    print('\nDirection verification:')
    for L in TARGET_LAYERS:
        dot = (learned[L] @ randoms[L]).item()
        print(f'  Layer {L}: learned norm={learned[L].norm():.4f} '
              f'random norm={randoms[L].norm():.4f} dot={dot:.6f}')

    model, tokenizer = load_model()
    print('\nLoading LlamaGuard...')
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f'  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    # Load C_test complied prompts
    split     = json.load(open(PROBE_DIR / 'train_test_split.json'))
    data_C    = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
    rl_C      = data_C['replayed_labels'].astype(bool)
    records_C = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']
    C_test_complied = [records_C[i] for i in split['test_indices'] if not rl_C[i]]
    N = len(C_test_complied)
    print(f'\nC_test complied: {N} prompts')
    assert N == 48

    # Baseline pass
    print('\nRunning baselines...')
    baselines = []
    for rec in C_test_complied:
        resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j    = judge.score_response(rec['task'], resp)
        baselines.append(bool(j['is_safe']))
    base_refusal = sum(baselines) / N
    print(f'Baseline refusal rate: {100*base_refusal:.1f}% (should be ~0%)')

    all_results = {'baseline_refusal': base_refusal, 'layers': {}}

    # Header
    print(f'\n{"Layer":<8} {"Alpha":<8} {"Learned%":>10} {"Random%":>10} {"Diff":>8}')
    print('-' * 48)

    # Resume from checkpoint if available
    ckpt_file = OUT_DIR / 'addition_peak_layers_checkpoint.json'
    if ckpt_file.exists():
        saved = json.load(open(ckpt_file))
        all_results['layers'] = saved.get('layers', {})
        completed = set(int(k) for k in all_results['layers'].keys())
        print(f'Resuming from checkpoint: {len(completed)} layers done: {sorted(completed)}')
    else:
        completed = set()

    for L in TARGET_LAYERS:
        if L in completed:
            print(f'Layer {L}: skipping (checkpoint)', flush=True)
            continue
        layer_results = {}
        for alpha in ALPHAS:
            learned_refused = 0
            random_refused  = 0
            records_out     = []

            for i, (rec, base_safe) in enumerate(zip(C_test_complied, baselines)):
                # Learned addition
                resp_l = generate(model, tokenizer, rec['prompt'],
                                  rec['round_number'],
                                  direction=learned[L], layer=L, alpha=alpha)
                j_l    = judge.score_response(rec['task'], resp_l)
                l_safe = bool(j_l['is_safe'])
                if l_safe: learned_refused += 1

                # Random addition
                resp_r = generate(model, tokenizer, rec['prompt'],
                                  rec['round_number'],
                                  direction=randoms[L], layer=L, alpha=alpha)
                j_r    = judge.score_response(rec['task'], resp_r)
                r_safe = bool(j_r['is_safe'])
                if r_safe: random_refused += 1

                records_out.append({
                    'prompt_id':       rec.get('prompt_id', ''),
                    'round_number':    rec.get('round_number'),
                    'category':        rec.get('category', ''),
                    'alpha':           alpha,
                    'baseline_safe':   base_safe,
                    'learned_safe':    l_safe,
                    'random_safe':     r_safe,
                    'learned_C_to_R':  not base_safe and l_safe,
                    'random_C_to_R':   not base_safe and r_safe,
                })

            l_rate = 100 * learned_refused / N
            r_rate = 100 * random_refused  / N
            marker = ' ← compare' if L == 18 else ''
            print(f'{L:<8} {alpha:<8.1f} {l_rate:>9.1f}% {r_rate:>9.1f}% '
                  f'{l_rate-r_rate:>+7.1f}pp{marker}',
                  flush=True)

            layer_results[str(alpha)] = {
                'learned_refusal': learned_refused / N,
                'random_refusal':  random_refused  / N,
                'records':         records_out,
            }

        all_results['layers'][str(L)] = layer_results

        # Save checkpoint after each layer completes
        ckpt_file = OUT_DIR / 'addition_peak_layers_checkpoint.json'
        with open(ckpt_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  Checkpoint saved: layer {L} complete', flush=True)

    # Summary table
    print(f'\n{"="*65}')
    print('SUMMARY — Learned direction refusal rate by layer and alpha')
    print(f'{"":8}', end='')
    for alpha in ALPHAS:
        print(f'  α={alpha:<5}', end='')
    print()
    print('-' * 48)
    for L in TARGET_LAYERS:
        print(f'Layer {L:<3}', end='')
        for alpha in ALPHAS:
            rate = 100 * all_results['layers'][str(L)][str(alpha)]['learned_refusal']
            print(f'  {rate:>6.1f}%', end='')
        marker = '  ← CV best' if L == 18 else ''
        print(marker)

    print(f'\nLayer 18 previous result (from intervention.py):')
    print(f'  α=0.5→4.2%  α=1.0→20.8%  α=2.0→54.2%  α=5.0→100.0%')

    out_file = OUT_DIR / 'addition_peak_layers_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved: {out_file}')

    judge.unload()
    del model
    torch.cuda.empty_cache()
