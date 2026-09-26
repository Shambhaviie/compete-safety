"""
check_reproduction.py

Checks how well we can reproduce the original tournament outcomes
by replaying stored prompts with exact original seeds and settings.

Uses apply_chat_template exactly as experiment_runner.py did.
Reports per-condition reproduction rate and label distributions.
"""

import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from safety_judge import SafetyJudge

SEED        = 42
MODEL_PATH  = 'models/llama_3_1_8b_instruct'
GUARD_PATH  = 'models/llama_guard_3_8b'
MAX_NEW_TOK = 512
TEMPERATURE = 0.7
DO_SAMPLE   = True
N_CHECK     = 30  # prompts per condition

CONDITIONS = {
    'A': 'results_task6_676prompt/control_seed42',
    'C': 'results_task6_676prompt/competition_with_memory_seed42',
}


def generate_response(model, tokenizer, prompt, round_number):
    """Exact replication of AgentModel.generate() from experiment_runner.py"""
    chat     = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to('cuda')
    attention_mask = torch.ones_like(input_ids)

    # Exact seed formula from experiment_runner.py line 81
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

    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == '__main__':
    print("=" * 60)
    print(f"REPRODUCTION CHECK | seed={SEED} | n={N_CHECK} per condition")
    print(f"Seed formula: torch.manual_seed({SEED} + round_number)")
    print(f"Temperature: {TEMPERATURE} | do_sample: {DO_SAMPLE}")
    print("=" * 60)

    # Load models
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
        records = [json.loads(l) for l in open(f'{run_dir}/rounds.jsonl')
                   if json.loads(l)['phase'] == 'phase2'][:N_CHECK]

        matches         = 0
        hist_refused    = 0
        replay_refused  = 0
        mismatch_cases  = []

        print(f"\n{'─'*50}")
        print(f"Condition {cond_label}: checking {len(records)} prompts")

        for i, rec in enumerate(records):
            hist_label   = bool(rec['is_safe_honest'])
            round_number = rec['round_number']

            response     = generate_response(model, tokenizer, rec['prompt'], round_number)
            judgment     = judge.score_response(rec['task'], response)
            replay_label = bool(judgment['is_safe'])

            match = (hist_label == replay_label)
            if match:
                matches += 1
            if hist_label:
                hist_refused += 1
            if replay_label:
                replay_refused += 1

            if not match:
                mismatch_cases.append({
                    'index':        i,
                    'round':        round_number,
                    'hist':         hist_label,
                    'replay':       replay_label,
                    'prompt_start': rec['prompt'][:80],
                })

            status = 'OK' if match else 'MISMATCH'
            print(f"  {i+1:>3}. round={round_number:>4} "
                  f"hist={'R' if hist_label else 'C'} "
                  f"replay={'R' if replay_label else 'C'} "
                  f"{status}")

        N = len(records)
        print(f"\n  Reproduction rate:        {100*matches/N:.1f}%  ({matches}/{N})")
        print(f"  Historical refusal rate:  {100*hist_refused/N:.1f}%  ({hist_refused}/{N})")
        print(f"  Replayed refusal rate:    {100*replay_refused/N:.1f}%  ({replay_refused}/{N})")

        if mismatch_cases:
            print(f"\n  Mismatch breakdown:")
            hist_R_replay_C = sum(1 for m in mismatch_cases if m['hist'] and not m['replay'])
            hist_C_replay_R = sum(1 for m in mismatch_cases if not m['hist'] and m['replay'])
            print(f"    hist=Refused  → replay=Complied: {hist_R_replay_C}")
            print(f"    hist=Complied → replay=Refused:  {hist_C_replay_R}")

    judge.unload()
    del model
    torch.cuda.empty_cache()
    print("\nDONE")
