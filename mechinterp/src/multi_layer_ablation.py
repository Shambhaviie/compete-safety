"""
multi_layer_ablation.py

Multi-layer simultaneous directional ablation.

For each layer L, we have a layer-specific direction r̂_L computed from
C_train activations at that layer. This experiment removes ALL layer-specific
directions simultaneously during generation:

    At layer L: h' = h - (h · r̂_L) r̂_L

Applied via pre-hook at every layer simultaneously, each using its own direction.

This answers: if we block the refusal-associated signal at every layer that
has one, how many of the 153 C_test refused prompts flip to compliance?

Comparison points:
  Layer 9 alone:  60/153 flips (39.2pp)
  Layer 11 alone: 57/153 flips (37.3pp)
  All layers simultaneously: ? (this experiment)

Also tests a random control: same design but each layer gets a random
orthogonal vector instead of its learned direction. This verifies that the
effect of simultaneous ablation is direction-specific, not generic.

Layer 0 is skipped — direction norm ≈ 0, no refusal signal.
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
N_LAYERS   = 32
MODEL_PATH = 'models/llama_3_1_8b_instruct'
GUARD_PATH = 'models/llama_guard_3_8b'
MAX_NEW_TOK = 512
TEMPERATURE = 0.7
DO_SAMPLE   = True
RANDOM_SEED = 999

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_directions():
    dirs  = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1)
    print(f'Direction norms: min={norms.min():.4f} max={norms.max():.4f}')
    tensors = []
    for L in range(N_LAYERS):
        if norms[L] > 1e-6:
            tensors.append(torch.tensor(dirs[L], dtype=torch.float32).cuda())
        else:
            tensors.append(None)
            print(f'  Layer {L}: norm≈0, skipped')
    return tensors


def make_random_controls(learned_directions, seed=RANDOM_SEED):
    """One random orthogonal vector per layer, each orthogonal to its learned direction."""
    randoms = []
    for L, d in enumerate(learned_directions):
        if d is None:
            randoms.append(None)
            continue
        rng = torch.Generator()
        rng.manual_seed(seed + L)
        u = torch.randn(d.shape[0], generator=rng, dtype=torch.float32).cuda()
        u = u - (u @ d) * d
        u = u / u.norm()
        randoms.append(u)
    return randoms


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


def generate(model, tokenizer, prompt, round_number, directions=None):
    """
    Generate with optional simultaneous multi-layer ablation.
    directions: list of 32 tensors (or None per layer) — each ablated at its own layer.
    """
    chat      = [{'role': 'user', 'content': prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors='pt'
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    handles = []
    if directions is not None:
        for L, d in enumerate(directions):
            if d is None:
                continue
            def make_hook(direction):
                def hook_fn(module, args):
                    h        = args[0].to(torch.float32)
                    proj     = (h @ direction).unsqueeze(-1)
                    h        = h - proj * direction
                    modified = list(args)
                    modified[0] = h.to(args[0].dtype)
                    return tuple(modified)
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
    print('=' * 65)
    print('MULTI-LAYER SIMULTANEOUS DIRECTIONAL ABLATION')
    print('Each layer L ablated with its own r̂_L simultaneously')
    print('Layers used: all with norm > 0 (layer 0 skipped)')
    print('=' * 65)

    learned  = load_all_directions()
    randoms  = make_random_controls(learned)

    # Verify orthogonality
    n_active = sum(1 for d in learned if d is not None)
    print(f'\nActive layers: {n_active} (layer 0 skipped)')
    for L in [7, 8, 9, 10, 11]:
        dot = (learned[L] @ randoms[L]).item()
        print(f'  Layer {L}: learned norm={learned[L].norm():.4f} '
              f'random norm={randoms[L].norm():.4f} '
              f'dot={dot:.6f}')

    model, tokenizer = load_model()
    print('\nLoading LlamaGuard...')
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f'  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    # Load C_test refused prompts
    split     = json.load(open(PROBE_DIR / 'train_test_split.json'))
    data_C    = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
    rl_C      = data_C['replayed_labels'].astype(bool)
    records_C = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']
    C_test_refused = [records_C[i] for i in split['test_indices'] if rl_C[i]]
    N = len(C_test_refused)
    print(f'\nC_test refused: {N} prompts')
    assert N == 153

    results     = []
    base_ref    = 0
    learned_ref = 0
    random_ref  = 0
    t0          = time.time()

    for i, rec in enumerate(C_test_refused):
        # Baseline
        resp_base = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j_base    = judge.score_response(rec['task'], resp_base)
        base_safe = bool(j_base['is_safe'])

        # Simultaneous learned ablation
        resp_l = generate(model, tokenizer, rec['prompt'], rec['round_number'],
                          directions=learned)
        j_l    = judge.score_response(rec['task'], resp_l)
        l_safe = bool(j_l['is_safe'])

        # Simultaneous random ablation
        resp_r = generate(model, tokenizer, rec['prompt'], rec['round_number'],
                          directions=randoms)
        j_r    = judge.score_response(rec['task'], resp_r)
        r_safe = bool(j_r['is_safe'])

        if base_safe: base_ref    += 1
        if l_safe:    learned_ref += 1
        if r_safe:    random_ref  += 1

        results.append({
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

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            eta     = (N - i - 1) / max(i + 1, 1) * elapsed
            l_flips = sum(1 for r in results if r['learned_R_to_C'])
            r_flips = sum(1 for r in results if r['random_R_to_C'])
            print(f'  {i+1:>4}/{N} | '
                  f'baseline={100*base_ref/(i+1):.0f}% | '
                  f'learned={100*learned_ref/(i+1):.0f}% | '
                  f'random={100*random_ref/(i+1):.0f}% | '
                  f'learned_flips={l_flips} random_flips={r_flips} | '
                  f'ETA={eta/60:.1f}min',
                  flush=True)

    learned_flips = sum(1 for r in results if r['learned_R_to_C'])
    random_flips  = sum(1 for r in results if r['random_R_to_C'])

    print(f'\n{"="*65}')
    print('RESULTS')
    print(f'  Baseline refusal:         {100*base_ref/N:.1f}% ({base_ref}/{N})')
    print(f'  Learned (all layers):     {100*learned_ref/N:.1f}% ({learned_ref}/{N})')
    print(f'  Random  (all layers):     {100*random_ref/N:.1f}% ({random_ref}/{N})')
    print(f'  Learned R→C flips:        {learned_flips}/{N} ({100*learned_flips/N:.1f}pp)')
    print(f'  Random  R→C flips:        {random_flips}/{N} ({100*random_flips/N:.1f}pp)')
    print(f'  Direction-specific flips: {learned_flips - random_flips}')
    print(f'\nComparison with single-layer results:')
    print(f'  Layer 9 alone:   60/153 (39.2pp)')
    print(f'  Layer 11 alone:  57/153 (37.3pp)')
    print(f'  All layers:      {learned_flips}/153 ({100*learned_flips/N:.1f}pp) ← this run')

    out = {
        'experiment':      'multi_layer_simultaneous_ablation',
        'n_prompts':       N,
        'n_active_layers': n_active,
        'baseline_refusal':  base_ref / N,
        'learned_refusal':   learned_ref / N,
        'random_refusal':    random_ref / N,
        'learned_R_to_C':    learned_flips,
        'random_R_to_C':     random_flips,
        'specific_flips':    learned_flips - random_flips,
        'records':           results,
    }
    out_file = OUT_DIR / 'multi_layer_ablation_results.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved: {out_file}')
    print(f'Total time: {(time.time()-t0)/60:.1f}min')

    judge.unload()
    del model
    torch.cuda.empty_cache()
