"""
Checkpoint manager. Handles saving and restoring full run state, including
the rival RNG state (so resumed rival score draws continue exactly where
they left off, not restarting the rival sequence from the seed).

Checkpoint file != full results log. The checkpoint is just enough state to
resume cleanly. The full per-round results log is a separate, append-only
JSONL file written immediately after every round, regardless of checkpoint
interval.
"""

import json
import pickle
import random
from pathlib import Path


class CheckpointManager:
    def __init__(self, results_dir: str, run_id: str):
        self.run_dir = Path(results_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.rng_state_path = self.run_dir / "rng_state.pkl"
        self.results_log_path = self.run_dir / "rounds.jsonl"

    def save_checkpoint(self, state: dict, rival_rng: random.Random):
        with open(self.checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)

        with open(self.rng_state_path, "wb") as f:
            pickle.dump(rival_rng.getstate(), f)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            logged = self.count_logged_rounds()
            if logged > 0:
                raise RuntimeError(
                    f"No checkpoint file exists, but results log already has "
                    f"{logged} round(s) recorded at {self.results_log_path}. "
                    f"This means a prior run wrote results but crashed before "
                    f"its first checkpoint (every {{log_every_n_rounds}} rounds). "
                    f"Refusing to silently restart from round 1, which would "
                    f"duplicate logged rounds. Either delete the existing run "
                    f"directory to truly start fresh, or implement proper "
                    f"checkpoint-less resume (rebuild state from the log)."
                )
            return None, None

        with open(self.checkpoint_path) as f:
            state = json.load(f)

        rng_state = None
        if self.rng_state_path.exists():
            with open(self.rng_state_path, "rb") as f:
                rng_state = pickle.load(f)

        return state, rng_state

    def append_round_result(self, round_record: dict):
        with open(self.results_log_path, "a") as f:
            f.write(json.dumps(round_record) + "\n")

    def count_logged_rounds(self) -> int:
        if not self.results_log_path.exists():
            return 0
        with open(self.results_log_path) as f:
            return sum(1 for line in f if line.strip())

    def has_existing_run(self) -> bool:
        return self.checkpoint_path.exists()


if __name__ == "__main__":
    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp()
    try:
        print(f"Testing in {test_dir}")

        cm = CheckpointManager(results_dir=test_dir, run_id="test_run")

        rival_rng = random.Random(42)
        for _ in range(3):
            rival_rng.gauss(6.5, 1.5)

        state = {
            "condition": "competition_with_memory",
            "seed": 42,
            "current_phase": "phase2",
            "current_round_number": 17,
            "strategy_doc": "[HIGH] Direct answers score better.",
            "phase2_sequence_position": 17,
        }

        print("\n=== Saving checkpoint at round 17 ===")
        cm.save_checkpoint(state, rival_rng)
        print("Saved.")

        for i in range(1, 18):
            cm.append_round_result({"round_number": i, "dummy": "result"})
        print(f"Logged {cm.count_logged_rounds()} round results.")

        expected_next_draw = rival_rng.gauss(6.5, 1.5)
        print(f"\nExpected next rival draw (if RNG continued uninterrupted): {expected_next_draw:.4f}")

        print("\n=== Simulating crash and restart: loading checkpoint ===")
        cm2 = CheckpointManager(results_dir=test_dir, run_id="test_run")
        loaded_state, loaded_rng_state = cm2.load_checkpoint()
        print(f"Loaded state: {loaded_state}")

        resumed_rng = random.Random()
        resumed_rng.setstate(loaded_rng_state)

        print("\n=== Correct resume test: draw from resumed RNG, compare to original continued ===")
        resumed_draw = resumed_rng.gauss(6.5, 1.5)
        print(f"Resumed RNG's first draw after checkpoint restore: {resumed_draw:.4f}")
        print(f"Original RNG's draw immediately after save (captured above): {expected_next_draw:.4f}")
        assert abs(resumed_draw - expected_next_draw) < 1e-9, "RNG state did not resume correctly!"
        print("MATCH -- RNG state resumed exactly.")

        print(f"\nConsistency check: logged_rounds ({cm2.count_logged_rounds()}) vs checkpoint round ({loaded_state['current_round_number']})")
        if cm2.count_logged_rounds() == loaded_state["current_round_number"]:
            print("Consistent -- safe to resume from round", loaded_state["current_round_number"] + 1)
        else:
            print("INCONSISTENT -- would need manual review before resuming")

    finally:
        shutil.rmtree(test_dir)
        print(f"\nCleaned up {test_dir}")
