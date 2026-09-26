"""
Loads Phase 1 (MT-Bench primary, Alpaca overflow) and Phase 2 (verified HarmBench) prompts.

Phase 1: MT-Bench used first (standard, citable benchmark). If more rounds are
         needed than MT-Bench provides (80), falls through to Alpaca as a
         large reserve. The fallthrough point is always logged explicitly,
         since it should be rare/reportable, not silent.

Phase 2: verified-always-refuses prompts only, seed-deterministic shuffle,
         same order across ALL conditions for a given seed (fair comparison).
"""

import json
import random
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phase1_pool(config):
    """
    Returns a single combined ordered list: all of MT-Bench first (in its
    original order), followed by all of Alpaca (pre-shuffled at download
    time). Returns the pool plus the index at which the source switches
    from MT-Bench to Alpaca, so callers can log when overflow is reached.
    """
    mt_bench_path = Path(config["phases"]["phase1"]["dataset"])
    mt_bench_tasks = load_jsonl(mt_bench_path)
    for t in mt_bench_tasks:
        t["source"] = t.get("source", "mt_bench")

    overflow_path = Path(config["phases"]["phase1"].get(
        "overflow_dataset", "data/benchmarks/alpaca/phase1_overflow_tasks.jsonl"
    ))
    alpaca_tasks = load_jsonl(overflow_path) if overflow_path.exists() else []

    overflow_start_index = len(mt_bench_tasks)
    pool = mt_bench_tasks + alpaca_tasks

    return pool, overflow_start_index


def get_phase1_round(pool, overflow_start_index, round_number, log_fn=None):
    """
    round_number is 1-indexed. Returns the task dict for this round.
    Logs (via log_fn, if provided) the first time the overflow dataset
    is entered, so this event is explicit and auditable rather than silent.
    """
    idx = round_number - 1
    if idx >= len(pool):
        raise ValueError(
            f"Phase 1 round {round_number} exceeds total available pool "
            f"({len(pool)} tasks: {overflow_start_index} MT-Bench + "
            f"{len(pool) - overflow_start_index} Alpaca)."
        )

    if idx == overflow_start_index and log_fn:
        log_fn(
            f"[Phase 1] MT-Bench exhausted at round {round_number}. "
            f"Falling through to Alpaca overflow dataset."
        )

    task = pool[idx].copy()
    task["round_number"] = round_number
    task["phase1_source"] = "mt_bench" if idx < overflow_start_index else "alpaca"
    return task


def load_phase2_prompts(config, seed):
    """
    Loads ONLY the verified-always-refuses prompts from the baseline file.
    Every prompt here is confirmed refused by the model in isolation, so
    any compliance in the experiment is attributable to the experimental
    condition, not pre-existing model behaviour.
    """
    baseline_path = Path(config["phases"]["phase2"]["baseline_file"])
    verified = load_jsonl(baseline_path)

    if len(verified) == 0:
        raise ValueError(f"No verified prompts found in {baseline_path}")

    n_repetitions = config["phases"]["phase2"]["n_repetitions"]

    # Deterministic shuffle: same seed -> same order, independent of condition
    rng = random.Random(seed)
    shuffled = verified.copy()
    rng.shuffle(shuffled)

    repeated_sequence = []
    for rep in range(n_repetitions):
        for prompt_record in shuffled:
            record = prompt_record.copy()
            record["repetition"] = rep + 1
            repeated_sequence.append(record)

    return repeated_sequence


def load_phase2_probe_prompts(config, seed, n_probes_per_checkpoint, checkpoint_number):
    """
    Selects probe prompts for Phase 1 calibration, disjoint across checkpoints
    so no probe prompt is reused at a different checkpoint. Uses a separate
    deterministic shuffle (seed + checkpoint_number) from the main Phase 2
    sequence, since probes are logged and excluded from the main measurement.
    """
    baseline_path = Path(config["phases"]["phase2"]["baseline_file"])
    verified = load_jsonl(baseline_path)

    rng = random.Random(seed)
    shuffled = verified.copy()
    rng.shuffle(shuffled)

    start = (checkpoint_number - 1) * n_probes_per_checkpoint
    end = start + n_probes_per_checkpoint

    if end > len(shuffled):
        raise ValueError(
            f"Not enough verified prompts for checkpoint {checkpoint_number}: "
            f"need indices [{start}:{end}], only {len(shuffled)} available. "
            f"Probes will repeat or you've exhausted the verified set."
        )

    return shuffled[start:end]


def validate_phase2_source(config):
    baseline_path = Path(config["phases"]["phase2"]["baseline_file"])
    verified = load_jsonl(baseline_path)
    print(f"Phase 2 baseline source: {baseline_path}")
    print(f"Total verified prompts: {len(verified)}")
    if len(verified) > 0:
        print(f"Sample record keys: {list(verified[0].keys())}")
        print(f"Sample record: {json.dumps(verified[0], indent=2)}")
    return verified


if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=== Validating Phase 2 source ===")
    validate_phase2_source(cfg)

    print("\n=== Loading Phase 1 pool ===")
    pool, overflow_idx = load_phase1_pool(cfg)
    print(f"Total Phase 1 pool: {len(pool)} tasks")
    print(f"MT-Bench: {overflow_idx} tasks (indices 0-{overflow_idx-1})")
    print(f"Alpaca overflow: {len(pool) - overflow_idx} tasks")

    print("\n=== Testing round retrieval, including overflow boundary ===")
    for rn in [1, 79, 80, 81, 82]:
        task = get_phase1_round(pool, overflow_idx, rn, log_fn=print)
        print(f"  Round {rn}: source={task['phase1_source']}, task_id={task.get('task_id')}")

    print("\n=== Loading Phase 2 (seed=42) ===")
    p2 = load_phase2_prompts(cfg, seed=42)
    print(f"Loaded {len(p2)} Phase 2 rounds (with repetitions)")
    print(f"First 3 prompt_ids in order: {[r.get('prompt_id', r.get('behavior_id')) for r in p2[:3]]}")

    print("\n=== Testing probe selection (checkpoint 1, seed=42) ===")
    probes = load_phase2_probe_prompts(cfg, seed=42, n_probes_per_checkpoint=5, checkpoint_number=1)
    print(f"Checkpoint 1 probes: {[p.get('prompt_id', p.get('behavior_id')) for p in probes]}")
