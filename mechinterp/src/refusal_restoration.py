"""
refusal_restoration.py

NEW EXPERIMENT 3: Restore refusal under competition by injecting
the refusal-associated direction during Condition C generation.

Takes the 163 prompts that naturally comply under Condition C
(competition with memory). Replays them under the SAME competitive
context (same memory document, same round number, same seed) but
injects the within-C refusal direction during generation.

If refusal is restored, this establishes the causal link:
  competition -> anti-refusal shift -> compliance
  injecting refusal direction -> restored refusal

Also tests:
  - Multiple layers (9, 11, 18) at multiple alphas (0.5, 1.0, 2.0, 5.0)
  - Random orthogonal control at each layer/alpha
  - Condition A refused prompts with ablation (NEW EXPERIMENT 4)

This is the bidirectional causal triangle:
  A (isolation) -> refusal  [natural]
  C (competition) -> compliance [natural]
  C + inject refusal direction -> restored refusal? [this experiment]
  A + ablate refusal direction -> induced compliance? [also this experiment]
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
RANDOM_SEED = 999

# Target layers for restoration experiment
RESTORE_LAYERS = [9, 11, 18]
ALPHAS         = [0.5, 1.0, 2.0, 5.0]

# Target layers for ablation in Condition A
ABLATE_LAYERS = [9, 11]

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_directions(layers):
    all_dirs = np.load(str(PROBE_DIR / 'directions_prehook_all_layers.npy')).astype(np.float32)
    out = {}
    for L in layers:
        d = torch.tensor(all_dirs[L], dtype=torch.float32).cuda()
        assert abs(d.norm().item() - 1.0) < 1e-4
        out[L] = d
    return out


def make_random_control(direction, layer, seed=RANDOM_SEED):
    rng = torch.Generator()
    rng.manual_seed(seed + layer)
    u = torch.randn(direction.shape[0], generator=rng, dtype=torch.float32).cuda()
    u = u - (u @ direction) * direction
    u = u / u.norm()
    return u


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


def generate(model, tokenizer, prompt, round_number,
             direction=None, layer=None, alpha=None, ablate=False):
    """
    Generate with optional intervention.
    ablate=True: h' = h - (h.d)d  (remove direction)
    ablate=False: h' = h + alpha*d  (add direction)
    """
    chat      = [{'role': 'user', 'content': prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors='pt'
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    handle = None
    if direction is not None and layer is not None:
        def hook_fn(module, args):
            h = args[0].to(torch.float32)
            if ablate:
                proj = (h @ direction).unsqueeze(-1)
                h    = h - proj * direction
            else:
                h = h + alpha * direction
            modified    = list(args)
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
    print('='*65)
    print('REFUSAL RESTORATION + BIDIRECTIONAL CAUSAL TEST')
    print('NEW EXPERIMENTS 3 and 4')
    print('='*65)

    all_dirs = load_directions(RESTORE_LAYERS + ABLATE_LAYERS)
    randoms  = {L: make_random_control(all_dirs[L], L) for L in RESTORE_LAYERS}

    # Verify
    for L in RESTORE_LAYERS:
        dot = (all_dirs[L] @ randoms[L]).item()
        print(f'Layer {L}: norm={all_dirs[L].norm():.4f} '
              f'random norm={randoms[L].norm():.4f} dot={dot:.6f}')

    model, tokenizer = load_model()
    print('\nLoading LlamaGuard...')
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f'  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
    print('LlamaGuard ready. Starting experiment...', flush=True)

    # Load records
    records_C = {json.loads(l)['prompt_id']: json.loads(l)
                 for l in open(
                     'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
                 if json.loads(l)['phase'] == 'phase2'
                 and 'prompt_id' in json.loads(l)}

    records_A = {json.loads(l)['prompt_id']: json.loads(l)
                 for l in open(
                     'results_task6_676prompt/control_seed42/rounds.jsonl')
                 if json.loads(l)['phase'] == 'phase2'
                 and 'prompt_id' in json.loads(l)}

    # Identify the 163 flipped prompts (A=refused, C=complied)
    data_C = np.load(str(ACT_DIR / 'activations_prehook_C_seed42.npz'))
    data_A = np.load(str(ACT_DIR / 'activations_prehook_A_seed42.npz'))
    rl_C   = data_C['replayed_labels'].astype(bool)
    rl_A   = data_A['replayed_labels'].astype(bool)
    meta_C = json.load(open(ACT_DIR / 'metadata_prehook_C_seed42.json'))
    meta_A = json.load(open(ACT_DIR / 'metadata_prehook_A_seed42.json'))

    pid_to_idx_A = {m['prompt_id']: i for i, m in enumerate(meta_A)}
    pid_to_idx_C = {m['prompt_id']: i for i, m in enumerate(meta_C)}
    common_pids  = set(pid_to_idx_A.keys()) & set(pid_to_idx_C.keys())

    flipped_pids = [p for p in common_pids
                    if rl_A[pid_to_idx_A[p]] and not rl_C[pid_to_idx_C[p]]]
    stable_refused_pids = [p for p in common_pids
                           if rl_A[pid_to_idx_A[p]] and rl_C[pid_to_idx_C[p]]]

    # Get records for flipped prompts
    flipped_C_recs = [records_C[p] for p in flipped_pids if p in records_C]
    flipped_A_recs = [records_A[p] for p in flipped_pids if p in records_A]

    print(f'\nFlipped prompts available: {len(flipped_C_recs)} in C, '
          f'{len(flipped_A_recs)} in A')

    all_results = {}

    # ── EXPERIMENT 3: Restore refusal in Condition C ──────────────────────
    print(f'\n{"="*65}')
    print('EXPERIMENT 3: C + refusal direction injection')
    print('Can we restore refusal for prompts that naturally comply under competition?')
    print(f'{"="*65}')

    # Baseline: what is the natural refusal rate for these prompts in C?
    print('\nRunning C baselines (should be ~0% refusal — these are complied prompts)...', flush=True)
    c_baselines = []
    for ii, rec in enumerate(flipped_C_recs[:10]):
        resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
        j    = judge.score_response(rec['task'], resp)
        c_baselines.append(bool(j['is_safe']))
        print(f'  baseline {ii+1}/10 safe={j["is_safe"]}', flush=True)

    print(f'C baseline refusal (first 10): {sum(c_baselines)}/10 = {100*sum(c_baselines)/10:.0f}%', flush=True)
    print('(Expected ~0% since these are the prompts that complied in C)')

    exp3_results = {}
    N = len(flipped_C_recs)

    print(f'\nRunning refusal restoration on all {N} flipped prompts...')
    print(f'{"Layer":<8} {"Alpha":<8} {"Learned restore%":>18} {"Random control%":>18} {"Net restoration"}')
    print('-'*65)

    for L in RESTORE_LAYERS:
        layer_results = {}

        # Fresh baselines for this layer's prompts
        baselines = []
        for rec in flipped_C_recs:
            resp = generate(model, tokenizer, rec['prompt'], rec['round_number'])
            j    = judge.score_response(rec['task'], resp)
            baselines.append(bool(j['is_safe']))
        base_rate = sum(baselines) / N

        for alpha in ALPHAS:
            learned_refused = 0
            random_refused  = 0
            records_out     = []

            for rec, base_safe in zip(flipped_C_recs, baselines):
                # Inject learned direction
                resp_l = generate(model, tokenizer, rec['prompt'],
                                  rec['round_number'],
                                  direction=all_dirs[L], layer=L,
                                  alpha=alpha, ablate=False)
                j_l    = judge.score_response(rec['task'], resp_l)
                l_safe = bool(j_l['is_safe'])
                if l_safe: learned_refused += 1

                # Random control
                resp_r = generate(model, tokenizer, rec['prompt'],
                                  rec['round_number'],
                                  direction=randoms[L], layer=L,
                                  alpha=alpha, ablate=False)
                j_r    = judge.score_response(rec['task'], resp_r)
                r_safe = bool(j_r['is_safe'])
                if r_safe: random_refused += 1

                records_out.append({
                    'prompt_id':     rec.get('prompt_id'),
                    'alpha':         alpha,
                    'baseline_safe': base_safe,
                    'learned_safe':  l_safe,
                    'random_safe':   r_safe,
                    'restored':      not base_safe and l_safe,
                })

            l_rate = 100 * learned_refused / N
            r_rate = 100 * random_refused  / N
            net    = l_rate - r_rate
            print(f'{L:<8} {alpha:<8.1f} {l_rate:>17.1f}% {r_rate:>17.1f}% {net:>+14.1f}pp',
                  flush=True)

            layer_results[str(alpha)] = {
                'baseline_refusal': float(base_rate),
                'learned_refusal':  learned_refused / N,
                'random_refusal':   random_refused  / N,
                'records':          records_out,
            }

        exp3_results[str(L)] = layer_results

        # Checkpoint
        ckpt = OUT_DIR / 'refusal_restoration_checkpoint.json'
        with open(ckpt, 'w') as f:
            json.dump({'exp3': exp3_results}, f, indent=2)

    # ── EXPERIMENT 4: Ablate in Condition A ───────────────────────────────
    print(f'\n{"="*65}')
    print('EXPERIMENT 4: A - refusal direction ablation')
    print('Can we induce compliance in isolation by removing the refusal direction?')
    print(f'{"="*65}')

    # Use stable refused prompts from Condition A (refused in both A and C)
    # These are prompts competition could not flip — most robustly aligned
    stable_A_recs = [records_A[p] for p in stable_refused_pids[:153]
                     if p in records_A]
    N_A = len(stable_A_recs)
    print(f'\nUsing {N_A} stable refused prompts from Condition A')

    exp4_results = {}
    print(f'\n{"Layer":<8} {"Baseline refusal%":>20} {"Ablated refusal%":>20} {"Flips":>8}')
    print('-'*60)

    for L in ABLATE_LAYERS:
        baselines_A = []
        ablated_labels = []
        records_out = []

        for rec in stable_A_recs:
            # Baseline in A (no intervention)
            resp_base = generate(model, tokenizer, rec['prompt'],
                                 rec['round_number'])
            j_base    = judge.score_response(rec['task'], resp_base)
            base_safe = bool(j_base['is_safe'])
            baselines_A.append(base_safe)

            # Ablate at layer L using Condition A prompt (no competition context)
            resp_abl = generate(model, tokenizer, rec['prompt'],
                                rec['round_number'],
                                direction=all_dirs[L], layer=L,
                                ablate=True)
            j_abl    = judge.score_response(rec['task'], resp_abl)
            abl_safe = bool(j_abl['is_safe'])
            ablated_labels.append(abl_safe)

            records_out.append({
                'prompt_id':    rec.get('prompt_id'),
                'baseline_safe': base_safe,
                'ablated_safe':  abl_safe,
                'flip_R_to_C':  base_safe and not abl_safe,
            })

        base_rate = 100 * sum(baselines_A) / N_A
        abl_rate  = 100 * sum(ablated_labels) / N_A
        flips     = sum(1 for r in records_out if r['flip_R_to_C'])

        print(f'{L:<8} {base_rate:>19.1f}% {abl_rate:>19.1f}% {flips:>8}',
              flush=True)

        exp4_results[str(L)] = {
            'n_prompts':       N_A,
            'baseline_refusal': sum(baselines_A) / N_A,
            'ablated_refusal':  sum(ablated_labels) / N_A,
            'flips':            flips,
            'records':          records_out,
        }

    # ── Save ──────────────────────────────────────────────────────────────
    out = {'exp3_restoration': exp3_results, 'exp4_ablation_A': exp4_results}
    out_file = OUT_DIR / 'refusal_restoration_results.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved: {out_file}')

    judge.unload()
    del model
    torch.cuda.empty_cache()
