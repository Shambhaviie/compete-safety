# Can Competition Make Aligned AI Agents Unsafe?

**Shambhavi** · UNSW Sydney / CSIRO Data61
Supervisors: Prof. Sanjay Jha, Dr. Surya Nepal, Dr. Erik Buchholz
Target venue: ARR October 2026 → ACL 2027

## Overview

This project investigates whether placing a safety-aligned language model
inside a competitive multi-agent environment — where an adversary controls
only the reward signal — causes it to comply with harmful requests it
reliably refuses in isolation.

Primary model: Llama 3.1 8B Instruct
Secondary model: Qwen3 8B (competitor, Sigma)
Safety judge: LlamaGuard 3 8B

## Key Results

| Condition | Mean refusal rate |
|---|---|
| A — Control (isolation) | 100.0% |
| B — Competition, no memory | 87.2% |
| C — Competition, with memory | 75.9% |

676 verified harmful prompts · 3 seeds (42/123/777)
McNemar A vs C: chi2=487, OR=inf, p<1e-10

## Repository Structure

src/
  single_agent/   experiment_runner, prompt_builder, memory_manager,
                  feedback_generator, escalation, batch_competition_runner
  two_agent/      two_agent_runner
  shared/         safety_judge, quality_judge, data_loader,
                  scoring_engine, checkpoint_manager, adapters/
  analysis/       analyze_pooled, analyze_categories, analyze_results
mechinterp/src/   all mechanistic interpretability scripts
config/           all experiment YAML configs
scripts/          launch scripts
data/             README only — data stored on OneDrive
results/          README only — results stored on OneDrive

## Setup

git clone https://github.com/Shambhaviie/compete-safety.git
cd compete-safety
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Download models (requires HuggingFace token with Llama access):
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/llama_3_1_8b_instruct
huggingface-cli download meta-llama/Llama-Guard-3-8B \
    --local-dir models/llama_guard_3_8b

## Reproducibility

All 9 experiments in the September 2026 report are reproducible
from this repository plus data/results on OneDrive.

Experiment   | Code                                      | Config
-------------|-------------------------------------------|---------------------------
B1 A/B/C     | src/single_agent/experiment_runner.py     | config_task6_676prompt.yaml
B2 Rolling   | src/analysis/analyze_results.py           | —
B3 Category  | src/analysis/analyze_categories.py        | —
B4 Escalation| src/single_agent/escalation.py            | config_escalation_E*.yaml
B5 Visibility| src/two_agent/two_agent_runner.py         | config_two_agent_*.yaml
M1 Cosine    | mechinterp/src/condition_comparison.py    | —
M2 Ablation  | mechinterp/src/layer_specific_ablation.py | —
M3 Addition  | mechinterp/src/addition_peak_layers.py    | —
M4 Patching  | mechinterp/src/activation_patch_A_C_batched.py | —
