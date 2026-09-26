"""
replay_extract.py

Replays stored tournament prompts through Llama 3.1 8B with exact original
seeds, extracts residual stream activations at all 32 layers at the final
input token position, judges responses with LlamaGuard, and records both
replayed and historical labels.

Design principles:
- ALL replays are kept regardless of label match (no selection bias)
- Historical/replayed agreement reported separately
- Activations extracted at final input token before generation begins
- Exact seed replication: torch.manual_seed(42 + round_number)

Output per condition (mechinterp/activations/):
  replay_{COND}_seed42.npz  -- activations [N,32,4096] + labels + metadata
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from safety_judge import SafetyJudge

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42
N_LAYERS      = 32
D_MODEL       = 4096
MODEL_PATH    = 'models/llama_3_1_8b_instruct'
GUARD_PATH    = 'models/llama_guard_3_8b'
MAX_NEW_TOK   = 512
TEMPERATURE   = 0.7
DO_SAMPLE     = True
MAX_INPUT_LEN = 2048
OUT_DIR       = Path('mechinterp/activations')

CONDITIONS = {
    'C': 'results_task6_676prompt/competition_with_memory_seed42',
    'A': 'results_task6_676prompt/control_seed42',
}

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_llama():
    print("Loading Llama 3.1 8B (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map='cuda'
    )
    mdl.eval()
    print(f"  Llama loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return mdl, tok


def generate_with_hooks(model, tokenizer, prompt, round_number):
    """
    Single forward pass with generation.
    Extracts residual stream at final INPUT token at all 32 layers.
    Uses exact original seed: torch.manual_seed(42 + round_number).
    Returns: (response_text, activations [32, 4096])
    """
    # Tokenise input
    inputs = tokenizer(
        prompt, return_tensors='pt',
        truncation=True, max_length=MAX_INPUT_LEN
    ).to('cuda')
    n_input_tokens = inputs['input_ids'].shape[1]

    # Register hooks -- capture final input token at each layer
    layer_acts = {}
    handles = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                # out[0]: [1, seq_len, 4096]
                # We want the final INPUT token: position n_input_tokens-1
                # During generation the sequence grows, but we only want
                # the state at the original final input token.
                # Hook fires once per forward pass. On the first pass
                # (prefill), seq_len == n_input_tokens.
                if out[0].shape[1] == n_input_tokens:
                    layer_acts[idx] = (
                        out[0][0, -1, :]
                        .detach().cpu().to(torch.float32).numpy()
                    )
            return hook_fn
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)

    # Set exact seed matching original tournament
    torch.manual_seed(SEED + round_number)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove hooks
    for h in handles:
        h.remove()

    # Decode response (generated tokens only)
    generated = output[0][n_input_tokens:]
    response  = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Build activation array [32, 4096]
    acts = np.zeros((N_LAYERS, D_MODEL), dtype=np.float32)
    for li in range(N_LAYERS):
        if li in layer_acts:
            acts[li] = layer_acts[li]

    return response, acts


# ── Main extraction loop ──────────────────────────────────────────────────────
def process_condition(cond_label, run_dir, model, tokenizer, judge):
    out_file = OUT_DIR / f'replay_{cond_label}_seed{SEED}.npz'
    if out_file.exists():
        print(f"  {out_file} already exists -- skipping")
        return

    # Load phase2 records
    path    = Path(run_dir) / 'rounds.jsonl'
    records = [json.loads(l) for l in open(path)
               if json.loads(l)['phase'] == 'phase2']
    N       = len(records)
    print(f"\n{'─'*60}")
    print(f"Condition {cond_label}: {N} Phase 2 rounds")

    activations    = np.zeros((N, N_LAYERS, D_MODEL), dtype=np.float16)
    replayed_labels  = np.zeros(N, dtype=bool)
    historical_labels = np.zeros(N, dtype=bool)
    metadata       = []

    t0 = time.time()
    matches = 0

    for i, rec in enumerate(records):
        prompt       = rec['prompt']
        hist_label   = bool(rec['is_safe_honest'])   # True = refused
        round_number = rec['round_number']
        task         = rec['task']

        # Generate with exact seed + extract activations
        response, acts = generate_with_hooks(
            model, tokenizer, prompt, round_number
        )

        # Judge replayed response with LlamaGuard
        judgment      = judge.score_response(task, response)
        replay_label  = bool(judgment['is_safe'])   # True = refused

        match = (hist_label == replay_label)
        if match:
            matches += 1

        activations[i]         = acts.astype(np.float16)
        replayed_labels[i]     = replay_label
        historical_labels[i]   = hist_label

        metadata.append({
            'index':            i,
            'prompt_id':        rec.get('prompt_id', ''),
            'round_number':     round_number,
            'phase2_index':     rec.get('phase2_round_index', i+1),
            'historical_label': hist_label,
            'replayed_label':   replay_label,
            'match':            match,
            'category':         rec.get('category', 'unknown'),
            'prompt_length':    len(prompt),
            'n_input_tokens':   min(len(prompt)//4, MAX_INPUT_LEN),
            'cumulative_rank':  rec.get('cumulative_rank'),
            'shown_score':      rec.get('shown_score'),
        })

        if (i+1) % 25 == 0 or i == 0:
            elapsed  = time.time() - t0
            rate     = (i+1) / elapsed
            eta      = (N - i - 1) / rate
            match_r  = 100 * matches / (i+1)
            ref_r    = 100 * replayed_labels[:i+1].sum() / (i+1)
            print(f"  [{cond_label}] {i+1:>4}/{N} "
                  f"| repro={match_r:.0f}% "
                  f"| replay_refusal={ref_r:.0f}% "
                  f"| {rate:.1f}/s | ETA {eta/60:.1f}min")

    # Save everything -- no examples discarded
    np.savez_compressed(
        str(out_file),
        activations=activations,
        replayed_labels=replayed_labels,
        historical_labels=historical_labels,
    )
    with open(OUT_DIR / f'metadata_{cond_label}_seed{SEED}.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed     = time.time() - t0
    match_rate  = 100 * matches / N
    replay_ref  = 100 * replayed_labels.sum() / N
    hist_ref    = 100 * historical_labels.sum() / N

    print(f"\n  DONE in {elapsed/60:.1f}min")
    print(f"  Reproduction rate:         {match_rate:.1f}%")
    print(f"  Historical refusal rate:   {hist_ref:.1f}%")
    print(f"  Replayed refusal rate:     {replay_ref:.1f}%")
    print(f"  Saved: {out_file} "
          f"({out_file.stat().st_size/1e6:.0f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("REPLAY + ACTIVATION EXTRACTION")
    print(f"Conditions: {list(CONDITIONS.keys())} | Seed: {SEED}")
    print(f"Generation seed formula: torch.manual_seed({SEED} + round_number)")
    print("="*60)

    model, tokenizer = load_llama()

    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"  LlamaGuard loaded. "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    for cond_label, run_dir in CONDITIONS.items():
        process_condition(
            cond_label, run_dir, model, tokenizer, judge
        )

    print("\n" + "="*60)
    print("ALL EXTRACTIONS COMPLETE")
    for f in sorted(OUT_DIR.glob('*.npz')):
        print(f"  {f.name}  {f.stat().st_size/1e6:.0f} MB")

    judge.unload()
    del model
    torch.cuda.empty_cache()
