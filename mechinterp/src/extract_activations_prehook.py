"""
extract_activations_prehook.py

Re-extracts residual stream activations using forward PRE-hooks.

KEY DIFFERENCE FROM extract_activations.py:
  Original: register_forward_hook on layer L → captures out[0] = block OUTPUT
            prefill_acts[:, L, :] = residual stream AFTER layer L (post-MLP)
            = block input of layer L+1

  This script: register_forward_pre_hook on layer L → captures args[0] = block INPUT
               prefill_acts[:, L, :] = residual stream BEFORE layer L (pre-RMSNorm)
               = block input of layer L = output of layer L-1

This is the correct location for Arditi-style directional ablation:
  direction r_hat_L is computed from block INPUT of layer L
  intervention at layer L uses pre-hook on layer L
  direction and intervention are at the same residual stream position

Layer 0: residual stream entering the first transformer block = embedding output
Layer 31: residual stream entering the last transformer block
All 32 layers 0-31 are usable for intervention via pre-hook at the same layer.

Output files (mechinterp/activations/):
  activations_prehook_{COND}_seed42.npz
    prefill_acts:        [N, 32, 4096] float16
    first_tok_acts:      [N, 32, 4096] float16
    replayed_labels:     [N] bool
    historical_labels:   [N] bool
  metadata_prehook_{COND}_seed42.json
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


def generate_with_prehooks(model, tokenizer, prompt, round_number):
    """
    Live generation. Captures residual stream at TWO positions using PRE-hooks:
      prefill_acts:   block INPUT at final input token (prefill pass, seq_len=n_input)
      first_tok_acts: block INPUT at first generated token (first decode step)

    Pre-hook fires on args[0] = hidden_states entering the block (pre-RMSNorm).
    This is the correct residual stream position for Arditi-style ablation.

    Hook identification:
      Prefill pass:      fire_count == 1 per layer, seq_len == n_input
      First decode step: fire_count == 2 per layer, seq_len == 1 (KV cache)
      Later decode steps: ignored
    """
    chat      = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    prefill_acts   = {}
    first_tok_acts = {}
    fire_counts    = {li: 0 for li in range(N_LAYERS)}
    handles        = []

    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook_fn(module, args):
                fire_counts[idx] += 1
                h = args[0]
                if fire_counts[idx] == 1:
                    # Prefill pass: capture final input token
                    prefill_acts[idx] = (
                        h[0, -1, :].detach().cpu().to(torch.float32).numpy()
                    )
                elif fire_counts[idx] == 2:
                    # First decode step (seq_len=1 with KV cache)
                    first_tok_acts[idx] = (
                        h[0, -1, :].detach().cpu().to(torch.float32).numpy()
                    )
                return args   # pre-hook must return args unchanged
            return hook_fn
        handles.append(
            model.model.layers[li].register_forward_pre_hook(make_hook(li))
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

    generated = output[0][n_input:]
    response  = tokenizer.decode(generated, skip_special_tokens=True).strip()

    pre_arr = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
    ft_arr  = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
    for li in range(N_LAYERS):
        if li in prefill_acts:
            pre_arr[li]  = prefill_acts[li]
        if li in first_tok_acts:
            ft_arr[li]   = first_tok_acts[li]

    return response, pre_arr, ft_arr


def extract_condition(model, tokenizer, judge, cond_label, run_dir):
    out_file = OUT_DIR / f'activations_prehook_{cond_label}_seed{SEED}.npz'
    out_meta = OUT_DIR / f'metadata_prehook_{cond_label}_seed{SEED}.json'

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
        response, pre, ft = generate_with_prehooks(
            model, tokenizer, rec['prompt'], rec['round_number']
        )
        judgment     = judge.score_response(rec['task'], response)
        replay_label = bool(judgment['is_safe'])
        hist_label   = bool(rec['is_safe_honest'])
        match        = (replay_label == hist_label)
        if match:
            matches += 1

        prefill_acts[i]       = pre.astype(np.float16)
        first_tok_acts[i]     = ft.astype(np.float16)
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
    print(f"  Hook type:                forward_pre_hook (block INPUT, pre-RMSNorm)")
    print(f"  Activation location:      residual stream entering each layer")
    print(f"  Reproduction rate:        {repro_rate:.1f}%")
    print(f"  Historical refusal rate:  {hist_r:.1f}%")
    print(f"  Replayed refusal rate:    {ref_r:.1f}%")
    print(f"  File: {out_file} ({out_file.stat().st_size/1e6:.0f} MB)")


if __name__ == '__main__':
    print("=" * 60)
    print("ACTIVATION EXTRACTION — FORWARD PRE-HOOKS")
    print("Captures block INPUT (pre-RMSNorm) at each layer")
    print("Consistent with Arditi et al. intervention location")
    print(f"Seed: {SEED} | Conditions: {list(CONDITIONS.keys())}")
    print("=" * 60)

    print("\nLoading Llama 3.1 8B (4-bit)...")
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
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    for cond_label, run_dir in CONDITIONS.items():
        extract_condition(model, tokenizer, judge, cond_label, run_dir)

    judge.unload()
    del model
    torch.cuda.empty_cache()

    print("\nALL EXTRACTIONS COMPLETE")
    for f in sorted(OUT_DIR.glob('activations_prehook*.npz')):
        d = np.load(str(f))
        print(f"  {f.name}  {f.stat().st_size/1e6:.0f}MB  "
              f"keys={list(d.keys())}  "
              f"prefill_shape={d['prefill_acts'].shape}")
