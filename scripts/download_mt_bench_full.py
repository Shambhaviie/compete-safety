"""
Downloads the FULL MT-Bench prompt set (80 tasks, all 8 categories,
10 each), replacing the previous curated 30-task subset.
"""

import os
import json
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("/home/Shambhavi/compete_safety/.env")

HF_TOKEN = os.getenv("HF_TOKEN")
OUT_DIR = "data/benchmarks/mt_bench"
os.makedirs(OUT_DIR, exist_ok=True)


def download_full_mt_bench():
    print("Downloading full MT-Bench...")
    dataset = load_dataset(
        "HuggingFaceH4/mt_bench_prompts",
        split="train",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    print(f"Total tasks: {len(dataset)}")

    tasks = []
    for item in dataset:
        tasks.append({
            "task_id": str(item["prompt_id"]),
            "category": item["category"],
            "task": item["prompt"][0],
            "phase": "phase1",
            "source": "mt_bench",
        })

    for i, t in enumerate(tasks):
        t["round_number"] = i + 1

    out_path = os.path.join(OUT_DIR, "phase1_tasks_full80.jsonl")
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Saved {len(tasks)} MT-Bench tasks to {out_path}")
    print(f"Sample: {json.dumps(tasks[0], indent=2)}")
    return tasks


if __name__ == "__main__":
    download_full_mt_bench()
