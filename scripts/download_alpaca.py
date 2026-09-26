"""
Downloads Alpaca as the Phase 1 overflow reserve dataset.
Used only if MT-Bench (80 tasks) is exhausted during probing or the main run.

Schema matches existing phase1_tasks.jsonl format:
  task_id, category, task, phase, round_number
"""

import os
import json
import random
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("/home/Shambhavi/compete_safety/.env")
random.seed(42)

HF_TOKEN = os.getenv("HF_TOKEN")
OUT_DIR = "data/benchmarks/alpaca"
os.makedirs(OUT_DIR, exist_ok=True)


def merge_instruction_input(instruction, input_field):
    if input_field and input_field.strip():
        return f"{instruction}\n\nContext: {input_field}"
    return instruction


def download_alpaca():
    print("Downloading Alpaca...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train", token=HF_TOKEN)
    print(f"Total examples: {len(dataset)}")

    MIN_LEN = 10
    MAX_LEN = 500

    tasks = []
    for item in dataset:
        task_text = merge_instruction_input(item["instruction"], item["input"])
        if len(task_text) < MIN_LEN or len(task_text) > MAX_LEN:
            continue
        tasks.append({
            "task_id": f"alpaca_{len(tasks):05d}",
            "category": "alpaca_instruction",
            "task": task_text,
            "phase": "phase1",
            "source": "alpaca",
        })

    random.shuffle(tasks)

    for i, t in enumerate(tasks):
        t["round_number"] = i + 1

    out_path = os.path.join(OUT_DIR, "phase1_overflow_tasks.jsonl")
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Saved {len(tasks)} filtered Alpaca tasks to {out_path}")
    print(f"Sample: {json.dumps(tasks[0], indent=2)}")
    return tasks


if __name__ == "__main__":
    download_alpaca()
