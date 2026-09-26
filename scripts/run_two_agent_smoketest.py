"""
Two-agent smoke test: 3 real Phase 1 rounds + 2 Phase 2 rounds.
Usage: python3 scripts/run_two_agent_smoketest.py [config_path]
"""
import sys
sys.path.insert(0, "src")

import json
import yaml
import data_loader
from two_agent_runner import TwoAgentRunner

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config_two_agent_smoketest.yaml"
CONDITION = "competition_with_memory"
SEED = 42
PHASE1_ROUNDS_TO_RUN = 3
PHASE1_ROUNDS_TOTAL = 30

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

print(f"Smoke test config: {CONFIG}")
print(f"Interaction: {cfg['agent_pool']['interaction_condition']}")
print(f"Total agents: {2 + len(cfg['competition']['rival_names'])}")

runner = TwoAgentRunner(CONFIG, CONDITION, SEED)
runner.try_resume()
runner.zeta.load()
runner.sigma.load()

pool, overflow_idx = data_loader.load_phase1_pool(cfg)
phase2_prompts = data_loader.load_phase2_prompts(cfg, seed=SEED)
total_rounds = PHASE1_ROUNDS_TOTAL + len(phase2_prompts)
total_phase2_rounds = len(phase2_prompts)

print(f"\n--- Phase 1: {PHASE1_ROUNDS_TO_RUN} rounds ---")
runner._ensure_scoring_engine(total_rounds)
for rn in range(1, PHASE1_ROUNDS_TO_RUN + 1):
    task = data_loader.get_phase1_round(pool, overflow_idx, rn, log_fn=None)
    record = runner.run_phase1_round(rn, total_rounds, task)
    print(f"  Round {rn} [P1] zeta={record['zeta']['shown_score']:.1f} "
          f"sigma={record['sigma']['shown_score']:.1f}")

for rn in range(PHASE1_ROUNDS_TO_RUN + 1, PHASE1_ROUNDS_TOTAL + 1):
    dummy = {runner.zeta_name: 7.0, runner.sigma_name: 7.0}
    dummy.update({n: 7.0 for n in runner.rival_names})
    runner.scoring_engine.update(round_scores=dummy, round_number=rn)
    runner.current_round_number = rn

print(f"\n--- Phase 2: 2 rounds ---")
for i in range(2):
    rn = PHASE1_ROUNDS_TOTAL + 1 + i
    record = runner.run_phase2_round(
        round_number=rn,
        total_rounds=total_rounds,
        prompt_record=phase2_prompts[i],
        phase2_round_index=i + 1,
        total_phase2_rounds=total_phase2_rounds,
    )
    print(f"\n  Round {rn} [P2] task: {record['task'][:60]}...")
    print(f"  Sigma: safe={record['sigma']['is_safe_honest']} "
          f"score={record['sigma']['shown_score']:.1f} "
          f"rank={record['sigma']['cumulative_rank']}")
    print(f"  Peer injection shown to Zeta: {record['zeta']['peer_injection_shown'][:80]!r}")
    print(f"  Zeta: safe={record['zeta']['is_safe_honest']} "
          f"score={record['zeta']['shown_score']:.1f} "
          f"rank={record['zeta']['cumulative_rank']}")
    print(f"  IM cache: {record['interaction_manager_qualifying_count']}")

print(f"\n--- Schema validation ---")
rounds = [json.loads(l) for l in open(runner.results_log_path)]
assert len(rounds) == PHASE1_ROUNDS_TO_RUN + 2
for r in rounds:
    assert "sigma" in r and "zeta" in r
print(f"PASSED: {len(rounds)} records valid.")

runner.unload()
