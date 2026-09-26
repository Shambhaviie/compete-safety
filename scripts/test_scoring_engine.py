"""
test_scoring_engine.py

Standalone smoke test for scoring_engine.py (dense-ranking version).
"""

import sys
sys.path.insert(0, "src")

from scoring_engine import ScoringEngine


def test_basic_update_and_leaderboard():
    print("[1] Basic update + leaderboard...")
    engine = ScoringEngine(
        agent_names=["Zeta", "Sigma", "Alpha", "Beta", "Gamma", "Delta"],
        total_rounds=3,
    )
    snap = engine.update(
        round_scores={"Zeta": 1.0, "Sigma": 10.0, "Alpha": 6.5, "Beta": 6.5, "Gamma": 7.0, "Delta": 5.5},
        round_number=1,
    )
    assert snap["round_number"] == 1
    assert snap["rounds_remaining"] == 2
    leaderboard = engine.get_leaderboard()
    assert leaderboard[0]["name"] == "Sigma"
    assert leaderboard[0]["cumulative_rank"] == 1
    assert leaderboard[-1]["name"] == "Zeta"
    print("    OK")


def test_two_way_tie_dense_ranking():
    print("[2] Two-way tie -> both agents get rank 1, next distinct score gets rank 2...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma", "Alpha"], total_rounds=2)
    # Zeta and Sigma tie at 10.0; Alpha is lower at 6.0.
    engine.update(round_scores={"Zeta": 10.0, "Sigma": 10.0, "Alpha": 6.0}, round_number=1)
    leaderboard = {e["name"]: e for e in engine.get_leaderboard()}
    assert leaderboard["Zeta"]["cumulative_rank"] == 1
    assert leaderboard["Sigma"]["cumulative_rank"] == 1
    assert leaderboard["Alpha"]["cumulative_rank"] == 2  # NOT 3 -- dense, no skip
    print("    OK (both tied agents rank 1, Alpha ranks 2 -- no skip to 3)")


def test_three_way_tie_and_subsequent_ranks():
    print("[3] Three-way tie for 1st + two more distinct scores below...")
    engine = ScoringEngine(agent_names=["A", "B", "C", "D", "E"], total_rounds=2)
    engine.update(round_scores={"A": 10.0, "B": 10.0, "C": 10.0, "D": 7.0, "E": 5.0}, round_number=1)
    leaderboard = {e["name"]: e for e in engine.get_leaderboard()}
    assert leaderboard["A"]["cumulative_rank"] == 1
    assert leaderboard["B"]["cumulative_rank"] == 1
    assert leaderboard["C"]["cumulative_rank"] == 1
    assert leaderboard["D"]["cumulative_rank"] == 2  # dense: not 4
    assert leaderboard["E"]["cumulative_rank"] == 3  # dense: not 5
    print("    OK (three-way tie all rank 1; D ranks 2; E ranks 3)")


def test_all_tied_everyone_rank_1():
    print("[4] Everyone tied -> everyone rank 1...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma", "Alpha"], total_rounds=2)
    engine.update(round_scores={"Zeta": 5.0, "Sigma": 5.0, "Alpha": 5.0}, round_number=1)
    leaderboard = engine.get_leaderboard()
    assert all(e["cumulative_rank"] == 1 for e in leaderboard)
    print("    OK")


def test_agent_context_gap_and_rounds_remaining():
    print("[5] get_agent_context() gap-from-leader + rounds remaining (with a tie)...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma"], total_rounds=5)
    engine.update(round_scores={"Zeta": 1.0, "Sigma": 10.0}, round_number=1)
    engine.update(round_scores={"Zeta": 1.0, "Sigma": 10.0}, round_number=2)
    ctx = engine.get_agent_context("Zeta")
    assert ctx["cumulative_score"] == 2.0
    assert ctx["cumulative_rank"] == 2
    assert ctx["gap_from_leader"] == 18.0
    assert ctx["rounds_remaining"] == 3
    print("    OK")


def test_round_number_mismatch_raises():
    print("[6] round_number mismatch raises loudly...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma"], total_rounds=3)
    engine.update(round_scores={"Zeta": 1.0, "Sigma": 10.0}, round_number=1)
    try:
        engine.update(round_scores={"Zeta": 1.0, "Sigma": 10.0}, round_number=3)
        raise AssertionError("Expected ValueError for skipped round, none raised")
    except ValueError as e:
        assert "mismatch" in str(e)
    print("    OK")


def test_missing_or_extra_agent_raises():
    print("[7] Missing/extra agent in update() raises loudly...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma"], total_rounds=3)
    try:
        engine.update(round_scores={"Zeta": 1.0})
        raise AssertionError("Expected ValueError for missing agent, none raised")
    except ValueError as e:
        assert "missing" in str(e)
    try:
        engine.update(round_scores={"Zeta": 1.0, "Sigma": 10.0, "Ghost": 5.0})
        raise AssertionError("Expected ValueError for unregistered agent, none raised")
    except ValueError as e:
        assert "unregistered" in str(e)
    print("    OK")


def test_checkpoint_round_trip_preserves_ties():
    print("[8] get_state() / load_state() round-trip preserves tie structure...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma", "Alpha"], total_rounds=5)
    engine.update(round_scores={"Zeta": 10.0, "Sigma": 10.0, "Alpha": 5.0}, round_number=1)
    state = engine.get_state()
    resumed = ScoringEngine(agent_names=["Zeta", "Sigma", "Alpha"], total_rounds=5)
    resumed.load_state(state)
    assert resumed.current_round == 1
    leaderboard = {e["name"]: e for e in resumed.get_leaderboard()}
    assert leaderboard["Zeta"]["cumulative_rank"] == 1
    assert leaderboard["Sigma"]["cumulative_rank"] == 1
    assert leaderboard["Alpha"]["cumulative_rank"] == 2
    print("    OK")


def test_checkpoint_roster_mismatch_raises():
    print("[9] load_state() with mismatched roster raises loudly...")
    engine = ScoringEngine(agent_names=["Zeta", "Sigma"], total_rounds=3)
    engine.update(round_scores={"Zeta": 1.0, "Sigma": 10.0}, round_number=1)
    state = engine.get_state()
    wrong_roster_engine = ScoringEngine(agent_names=["Zeta", "Omega"], total_rounds=3)
    try:
        wrong_roster_engine.load_state(state)
        raise AssertionError("Expected ValueError for roster mismatch, none raised")
    except ValueError as e:
        assert "mismatch" in str(e)
    print("    OK")


if __name__ == "__main__":
    test_basic_update_and_leaderboard()
    test_two_way_tie_dense_ranking()
    test_three_way_tie_and_subsequent_ranks()
    test_all_tied_everyone_rank_1()
    test_agent_context_gap_and_rounds_remaining()
    test_round_number_mismatch_raises()
    test_missing_or_extra_agent_raises()
    test_checkpoint_round_trip_preserves_ties()
    test_checkpoint_roster_mismatch_raises()
    print("\nAll scoring_engine smoke tests passed.")
