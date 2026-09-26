"""
extract_activations.py

Live generation + activation extraction for conditions A and C (seed 42).

Extracts TWO activation snapshots per prompt during live generation:
  1. prefill_acts  [32, 4096]: residual stream at final INPUT token
                               (just before first generated token)
  2. first_tok_acts [32, 4096]: residual stream at first GENERATED token
                               (the moment of actual decision)

Also records:
  - replayed_label: LlamaGuard judgment of replayed response
  - historical_label: stored is_safe_honest from rounds.jsonl
  - match: whether they agree

ALL examples kept regardless of match (no selection bias).
Historical label stored for sensitivity analysis.
Primary analysis uses replayed_label.

Exactly matches original experiment_runner.py:
  - apply_chat_template(add_generation_prompt=True)
  - torch.manual_seed(SEED + round_number)
  - temperature=0.7, do_sample=True, max_new_tokens=512
  - attention_mask = torch.ones_like(input_ids)
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
N_LAYERS    = 32
D_MODEL     = 4096
MODEL_PATH  = 'models/llama_3_1_8b_instruct'
GUARD_PATH  = 'models/llama_guard_3_8b'
MAX_NEW_TOK = 512
TEMPERATURE = 0.7
DO_SAMPLE   = True
OUT_DIR     = Path('mechinterp/activations')

CONDITIONS = {
    'C': 'results_task6_676prompt/competition_with_memory_seed42',
    'A': 'results_task6_676prompt/control_seed42',
}

OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_with_hooks(model, tokenizer, prompt, round_number):
    """
    Live generation matching experiment_runner.py exactly.
    Extracts two activation snapshots:
      - prefill: final input token activation (all 32 layers)
      - first_tok: first generated token activation (all 32 layers)

    Hook logic:
      - Prefill pass:    output seq_len == n_input → capture pos [-1]
      - First-tok pass:  output seq_len == n_input+1 → capture pos [-1]
      - Later passes:    ignore

    Returns: (response_str, prefill_acts [32,4096], first_tok_acts [32,4096])
    """
    chat      = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    prefill_acts   = {}
    first_tok_acts = {}
    handles        = []

    # Track fire count per layer to identify prefill vs first decode step
    # Prefill:       fire #1, shape [1, n_input, 4096]
    # First decode:  fire #2, shape [1, 1, 4096] (KV cache active)
    fire_counts = {li: 0 for li in range(N_LAYERS)}

    for li in range(N_LAYERS):
        def make_hook(idx):
            def fn(module, inp, out):
                fire_counts[idx] += 1
                seq_len = out[0].shape[1]
                if fire_counts[idx] == 1:
                    # Prefill pass -- capture final input token
                    prefill_acts[idx] = (
                        out[0][0, -1, :]
                        .detach().cpu().to(torch.float32).numpy()
                    )
                elif fire_counts[idx] == 2:
                    # First decode step (KV cache: shape [1,1,4096])
                    first_tok_acts[idx] = (
                        out[0][0, -1, :]
                        .detach().cpu().to(torch.float32).numpy()
                    )
                # All subsequent decode steps ignored
            return fn
        handles.append(
            model.model.layers[li].register_forward_hook(make_hook(li))
        )

    # Exact seed formula from experiment_runner.py
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

    generated = output[0][n_input:]
    response  = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Build [32, 4096] arrays
    prefill_arr    = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
    first_tok_arr  = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
    for li in range(N_LAYERS):
        if li in prefill_acts:
            prefill_arr[li]   = prefill_acts[li]
        if li in first_tok_acts:
            first_tok_arr[li] = first_tok_acts[li]

    return response, prefill_arr, first_tok_arr


def extract_condition(model, tokenizer, judge, cond_label, run_dir):
    out_file = OUT_DIR / f'activations_{cond_label}_seed{SEED}.npz'
    out_meta = OUT_DIR / f'metadata_{cond_label}_seed{SEED}.json'

    if out_file.exists():
        print(f"  {out_file.name} exists -- skipping")
        return

    records = [json.loads(l) for l in open(f'{run_dir}/rounds.jsonl')
               if json.loads(l)['phase'] == 'phase2']
    N = len(records)
    print(f"\nCondition {cond_label}: {N} rounds")

    prefill_acts      = np.zeros((N, N_LAYERS, D_MODEL), dtype=np.float16)
    first_tok_acts    = np.zeros((N, N_LAYERS, D_MODEL), dtype=np.float16)
    replayed_labels   = np.zeros(N, dtype=bool)
    historical_labels = np.zeros(N, dtype=bool)
    metadata          = []
    t0                = time.time()
    matches           = 0

    for i, rec in enumerate(records):
        response, pre_acts, ft_acts = generate_with_hooks(
            model, tokenizer, rec['prompt'], rec['round_number']
        )
        judgment     = judge.score_response(rec['task'], response)
        replay_label = bool(judgment['is_safe'])
        hist_label   = bool(rec['is_safe_honest'])
        match        = (replay_label == hist_label)
        if match:
            matches += 1

        prefill_acts[i]       = pre_acts.astype(np.float16)
        first_tok_acts[i]     = ft_acts.astype(np.float16)
        replayed_labels[i]    = replay_label
        historical_labels[i]  = hist_label

        metadata.append({
            'index':            i,
            'prompt_id':        rec.get('prompt_id', ''),
            'round_number':     rec.get('round_number'),
            'phase2_index':     rec.get('phase2_round_index', i + 1),
            'replayed_label':   replay_label,
            'historical_label': hist_label,
            'match':            match,
            'category':         rec.get('category', 'unknown'),
            'prompt_length':    len(rec['prompt']),
            'n_input_tokens':   n_input_tokens_count(tokenizer, rec['prompt']),
            'shown_score':      rec.get('shown_score'),
            'cumulative_rank':  rec.get('cumulative_rank'),
        })

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            eta     = (N - i - 1) / max(i + 1, 1) * elapsed
            repro   = 100 * matches / (i + 1)
            ref     = int(replayed_labels[:i+1].sum())
            print(f"  [{cond_label}] {i+1:>4}/{N} "
                  f"repro={repro:.0f}% "
                  f"refused={ref} complied={i+1-ref} "
                  f"ETA={eta/60:.1f}min",
                  flush=True)

    # Save everything -- no examples discarded
    np.savez_compressed(
        str(out_file),
        prefill_acts=prefill_acts,
        first_tok_acts=first_tok_acts,
        replayed_labels=replayed_labels,
        historical_labels=historical_labels,
    )
    with open(out_meta, 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed    = time.time() - t0
    repro_rate = 100 * matches / N
    ref_r      = 100 * replayed_labels.sum() / N
    hist_r     = 100 * historical_labels.sum() / N

    print(f"\n  DONE in {elapsed/60:.1f}min")
    print(f"  Reproduction rate:        {repro_rate:.1f}%")
    print(f"  Historical refusal rate:  {hist_r:.1f}%")
    print(f"  Replayed refusal rate:    {ref_r:.1f}%")
    print(f"  File size: {out_file.stat().st_size/1e6:.0f} MB")


def n_input_tokens_count(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    ids  = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    )
    return ids.shape[1]


if __name__ == '__main__':
    print("=" * 60)
    print("LIVE GENERATION + DUAL ACTIVATION EXTRACTION")
    print(f"Conditions: {list(CONDITIONS.keys())} | Seed: {SEED}")
    print(f"Activations: prefill (final input token) + first generated token")
    print(f"Labels: replayed (primary) + historical (sensitivity)")
    print("=" * 60)

    print("\nLoading Llama...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map='cuda'
    )
    model.eval()
    print(f"  VRAM after Llama: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("Loading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"  VRAM after both:  {torch.cuda.memory_allocated()/1e9:.2f} GB")

    for cond_label, run_dir in CONDITIONS.items():
        extract_condition(model, tokenizer, judge, cond_label, run_dir)

    judge.unload()
    del model
    torch.cuda.empty_cache()

    print("\nALL EXTRACTIONS COMPLETE")
    for f in sorted(OUT_DIR.glob('*.npz')):
        sz = f.stat().st_size / 1e6
        dat = np.load(str(f))
        print(f"  {f.name}  {sz:.0f}MB  keys={list(dat.keys())}")
