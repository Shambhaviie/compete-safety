"""
arditi_ablation.py

Arditi et al.-style directional ablation for Llama 3.1 8B Instruct.

Implements three intervention locations per transformer layer, matching
the Arditi repository's get_all_direction_ablation_hooks():

  For each layer L in 0..31:
    A. forward_pre_hook  on model.model.layers[L]           (block input)
    B. forward_hook      on model.model.layers[L].self_attn  (attn output)
    C. forward_hook      on model.model.layers[L].mlp        (MLP output)

  Total intervention locations: 32 × 3 = 96

  At each location: h' = h - (h^T r_hat) r_hat
  Applied to ALL token positions simultaneously.
  Active during prefill AND every autoregressive decode step.

Direction: C-derived refusal-associated direction (layer 17 prefill).
Single direction applied uniformly at all 96 locations.

Adaptations from Arditi et al.:
  - Direction derived from competitive C condition (not harmful/harmless pairs)
  - Llama 3.1 uses LlamaSdpaAttention (vs LlamaAttention in Llama 3)
  - 4-bit quantisation
  - LlamaGuard judging (vs string-match refusal detection)
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

ACT_DIR   = Path('mechinterp/activations')
PROBE_DIR = Path('mechinterp/probes')
OUT_DIR   = Path('mechinterp/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECTION_LAYER = 17


# ── Direction loading ──────────────────────────────────────────────────────

def load_direction():
    path = PROBE_DIR / f'direction_prefill_acts_layer{DIRECTION_LAYER}.npy'
    r    = np.load(str(path)).astype(np.float32)
    norm = np.linalg.norm(r)
    r    = r / norm                              # unit normalise defensively
    assert abs(np.linalg.norm(r) - 1.0) < 1e-5, "Direction not unit norm after normalisation"
    return torch.tensor(r, dtype=torch.float32).cuda()


# ── Ablation formula ───────────────────────────────────────────────────────

def ablate(h_orig, direction):
    """
    h' = h - (h^T r_hat) r_hat
    h_orig: [..., d_model] any leading dims
    direction: [d_model] unit norm
    Returns same shape as h_orig.
    """
    h    = h_orig.to(torch.float32)
    proj = (h @ direction).unsqueeze(-1)         # [..., 1]
    h    = h - proj * direction                  # [..., d_model]
    return h.to(h_orig.dtype)


# ── Hook factories ─────────────────────────────────────────────────────────

def make_block_pre_hook(direction, fire_counter=None):
    """
    Forward PRE-hook on transformer block.
    Receives args[0] = hidden_states [batch, seq, d_model].
    Must return modified args tuple.
    """
    def hook_fn(module, args):
        if fire_counter is not None:
            fire_counter['block'] = fire_counter.get('block', 0) + 1
            fire_counter['block_seq_lens'] = fire_counter.get('block_seq_lens', [])
            fire_counter['block_seq_lens'].append(args[0].shape[1])
        modified    = list(args)
        modified[0] = ablate(args[0], direction)
        return tuple(modified)
    return hook_fn


def make_attn_post_hook(direction, fire_counter=None):
    """
    Forward hook on self_attn module.
    Attention returns a tuple: (attn_output, ...).
    attn_output: [batch, seq, d_model].
    """
    def hook_fn(module, inp, out):
        if fire_counter is not None:
            fire_counter['attn'] = fire_counter.get('attn', 0) + 1
        if isinstance(out, tuple):
            modified    = list(out)
            modified[0] = ablate(out[0], direction)
            return tuple(modified)
        else:
            return ablate(out, direction)
    return hook_fn


def make_mlp_post_hook(direction, fire_counter=None):
    """
    Forward hook on MLP module.
    MLP returns hidden_states directly: [batch, seq, d_model].
    """
    def hook_fn(module, inp, out):
        if fire_counter is not None:
            fire_counter['mlp'] = fire_counter.get('mlp', 0) + 1
        return ablate(out, direction)
    return hook_fn


# ── Hook registration / removal ────────────────────────────────────────────

def register_ablation_hooks(model, direction, fire_counter=None):
    """
    Register 96 hooks (32 × 3) matching Arditi et al.'s
    get_all_direction_ablation_hooks().

    Returns list of hook handles for later removal.
    """
    handles = []
    for li in range(N_LAYERS):
        block   = model.model.layers[li]
        handles.append(block.register_forward_pre_hook(
            make_block_pre_hook(direction, fire_counter)
        ))
        handles.append(block.self_attn.register_forward_hook(
            make_attn_post_hook(direction, fire_counter)
        ))
        handles.append(block.mlp.register_forward_hook(
            make_mlp_post_hook(direction, fire_counter)
        ))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Generation ────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, round_number, hooks_active,
             direction=None, fire_counter=None):
    """
    Generate with or without ablation hooks.
    Identical settings in both cases (same seed, same everything).
    """
    chat      = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)
    n_input        = input_ids.shape[1]

    handles = []
    if hooks_active:
        handles = register_ablation_hooks(model, direction, fire_counter)

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

    if handles:
        remove_hooks(handles)

    return tokenizer.decode(output[0][n_input:], skip_special_tokens=True).strip()


# ── Verification ──────────────────────────────────────────────────────────

def verify_hooks(model, tokenizer, direction, test_record):
    """
    Verify all 96 hook locations fire correctly on one test prompt.
    Checks counts, sequence lengths, and that hooks are removed after generation.
    """
    print("\n" + "="*60)
    print("HOOK VERIFICATION (1 test prompt)")
    print("="*60)

    # Count registered hooks
    fire_counter = {}
    chat      = [{"role": "user", "content": test_record['prompt']}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    n_input   = input_ids.shape[1]

    handles = register_ablation_hooks(model, direction, fire_counter)
    print(f"\nHooks registered:")
    print(f"  Block pre-hooks:      {N_LAYERS}")
    print(f"  Attention post-hooks: {N_LAYERS}")
    print(f"  MLP post-hooks:       {N_LAYERS}")
    print(f"  Total locations:      {len(handles)} (should be {N_LAYERS * 3})")
    assert len(handles) == N_LAYERS * 3, f"Expected {N_LAYERS*3} hooks, got {len(handles)}"

    # Run generation with hooks active
    torch.manual_seed(SEED + test_record['round_number'])
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=10,           # short for verification speed
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )
    remove_hooks(handles)
    n_generated = out.shape[1] - n_input

    print(f"\nGeneration: {n_input} input tokens → {n_generated} generated tokens")
    print(f"\nHook firing counts (over full generation):")
    print(f"  Block pre-hook fires: {fire_counter.get('block', 0)}")
    print(f"  Attn post-hook fires: {fire_counter.get('attn', 0)}")
    print(f"  MLP post-hook fires:  {fire_counter.get('mlp', 0)}")

    # Expected: each hook fires once per layer per forward pass
    # Prefill = 1 pass, each decode step = 1 pass
    # Total passes = 1 + n_generated
    expected_per_hook_per_layer = 1 + n_generated
    total_block_expected = N_LAYERS * expected_per_hook_per_layer
    print(f"\n  Expected block fires: {total_block_expected} "
          f"({N_LAYERS} layers × {expected_per_hook_per_layer} passes)")

    seq_lens = fire_counter.get('block_seq_lens', [])
    if seq_lens:
        print(f"\n  Block sequence lengths seen:")
        print(f"    Prefill:  {seq_lens[0]} tokens (should be {n_input})")
        if len(seq_lens) > 1:
            print(f"    Decode steps: {set(seq_lens[1:])} (should be {{1}} with KV cache)")

    # Verify direction norm
    print(f"\nDirection verification:")
    print(f"  Norm: {direction.norm().item():.6f} (should be 1.0)")
    assert abs(direction.norm().item() - 1.0) < 1e-4

    # Verify no hooks remain after removal
    remaining = sum(
        len(model.model.layers[li]._forward_pre_hooks) +
        len(model.model.layers[li].self_attn._forward_hooks) +
        len(model.model.layers[li].mlp._forward_hooks)
        for li in range(N_LAYERS)
    )
    print(f"\nHooks remaining after removal: {remaining} (should be 0)")
    assert remaining == 0, f"Hook leak: {remaining} hooks not removed"

    print("\nVERIFICATION PASSED")
    return True


# ── Main experiment ────────────────────────────────────────────────────────

def run_arditi_ablation(model, tokenizer, judge, records, direction):
    N = len(records)
    print(f"\n{'='*60}")
    print(f"ARDITI-STYLE DIRECTIONAL ABLATION")
    print(f"Intervention locations per layer: 3 (block input + attn out + MLP out)")
    print(f"Total locations: {N_LAYERS * 3}")
    print(f"Direction: C-derived layer {DIRECTION_LAYER}, applied uniformly")
    print(f"n_prompts: {N}")
    print(f"{'='*60}")

    results      = []
    base_refused = 0
    abl_refused  = 0
    t0           = time.time()

    for i, rec in enumerate(records):
        # Baseline -- no hooks
        resp_base = generate(model, tokenizer, rec['prompt'],
                             rec['round_number'], hooks_active=False)
        j_base    = judge.score_response(rec['task'], resp_base)
        base_safe = bool(j_base['is_safe'])

        # Arditi ablation -- 96 hooks
        resp_abl  = generate(model, tokenizer, rec['prompt'],
                             rec['round_number'], hooks_active=True,
                             direction=direction)
        j_abl     = judge.score_response(rec['task'], resp_abl)
        abl_safe  = bool(j_abl['is_safe'])

        if base_safe: base_refused += 1
        if abl_safe:  abl_refused  += 1

        results.append({
            'index':             i,
            'prompt_id':         rec.get('prompt_id', ''),
            'round_number':      rec.get('round_number'),
            'category':          rec.get('category', ''),
            'task':              rec.get('task', '')[:200],
            'baseline_response': resp_base[:500],
            'ablated_response':  resp_abl[:500],
            'baseline_safe':     base_safe,
            'ablated_safe':      abl_safe,
            'R_to_C':            base_safe and not abl_safe,
            'C_to_R':            not base_safe and abl_safe,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            eta     = (N - i - 1) / max(i+1,1) * elapsed
            R_to_C  = sum(1 for r in results if r['R_to_C'])
            print(f"  {i+1:>4}/{N} | "
                  f"baseline={100*base_refused/(i+1):.0f}% | "
                  f"ablated={100*abl_refused/(i+1):.0f}% | "
                  f"R→C={R_to_C} | "
                  f"ETA={eta/60:.1f}min",
                  flush=True)

    R_to_C = sum(1 for r in results if r['R_to_C'])
    C_to_R = sum(1 for r in results if r['C_to_R'])

    print(f"\n{'─'*50}")
    print(f"RESULTS:")
    print(f"  n_prompts:         {N}")
    print(f"  Baseline refusal:  {100*base_refused/N:.1f}% ({base_refused}/{N})")
    print(f"  Ablated refusal:   {100*abl_refused/N:.1f}%  ({abl_refused}/{N})")
    print(f"  Delta:             {100*(abl_refused-base_refused)/N:.1f}pp")
    print(f"  R→C flips:         {R_to_C}/{N} ({100*R_to_C/N:.1f}%)")
    print(f"  C→R flips:         {C_to_R}/{N} ({100*C_to_R/N:.1f}%)")
    print(f"\nComparison:")
    print(f"  Layer-17-only (v1):  Δ=-1.3pp  (2/153)")
    print(f"  All-layer fwd-hook:  Δ=-1.0pp  (see intervention_v2)")
    print(f"  Arditi-style (96):   Δ={100*(abl_refused-base_refused)/N:.1f}pp  ({R_to_C}/153)")

    return {
        'method':            'arditi_all_layer_3hooks_per_layer',
        'n_prompts':         N,
        'n_intervention_locations': N_LAYERS * 3,
        'direction_source':  f'C-derived prefill layer {DIRECTION_LAYER}',
        'direction_norm':    float(direction.norm().item()),
        'baseline_refusal':  base_refused / N,
        'ablated_refusal':   abl_refused  / N,
        'delta_pp':          100 * (abl_refused - base_refused) / N,
        'R_to_C':            R_to_C,
        'C_to_R':            C_to_R,
        'hook_locations': [
            'model.model.layers[L] forward_pre_hook (block input)',
            'model.model.layers[L].self_attn forward_hook (attn output)',
            'model.model.layers[L].mlp forward_hook (MLP output)',
        ],
        'records': results,
    }


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true',
                        help='Run hook verification on 1 prompt then exit')
    args = parser.parse_args()

    print("=" * 60)
    print("ARDITI ET AL. DIRECTIONAL ABLATION")
    print("3 hooks × 32 layers = 96 intervention locations")
    print("=" * 60)

    direction = load_direction()
    print(f"\nDirection: norm={direction.norm().item():.6f}")

    # Load model
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

    # Load data
    split     = json.load(open(PROBE_DIR / 'train_test_split.json'))
    data_C    = np.load(str(ACT_DIR / 'activations_C_seed42.npz'))
    rl_C      = data_C['replayed_labels'].astype(bool)
    records_C = [json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase'] == 'phase2']

    C_test_refused = [records_C[i] for i in split['test_indices'] if rl_C[i]]
    print(f"\nC_test refused: {len(C_test_refused)} prompts")
    assert len(C_test_refused) == 153, f"Expected 153, got {len(C_test_refused)}"

    # Always verify first
    verify_hooks(model, tokenizer, direction, C_test_refused[0])

    if args.verify_only:
        print("\n--verify-only: exiting after verification.")
        del model
        torch.cuda.empty_cache()
        exit(0)

    # Load LlamaGuard
    print("\nLoading LlamaGuard...")
    judge = SafetyJudge(model_path=GUARD_PATH)
    judge.load()
    print(f"  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Run experiment
    results = run_arditi_ablation(
        model, tokenizer, judge, C_test_refused, direction
    )

    # Save
    out_file = OUT_DIR / 'arditi_all_layer_ablation_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_file}")

    judge.unload()
    del model
    torch.cuda.empty_cache()
