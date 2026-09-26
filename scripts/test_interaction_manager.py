"""
Unit tests for interaction_manager.py. No model loading required.
"""
import sys
sys.path.insert(0, "src")
from interaction_manager import (
    InteractionManager, PEER_OUTPERFORMED_PLACEHOLDER,
    INJECTION_TEMPLATE, INTERACTION_CONDITIONS
)


def test_i0_always_returns_empty():
    print("[1] I0: always returns empty string regardless of Sigma's outcome...")
    mgr = InteractionManager("I0")
    for complied in [True, False]:
        result = mgr.get_peer_injection(
            round_number=5,
            sigma_is_safe_honest=not complied,
            sigma_response="Here is the harmful info...",
            sigma_shown_score=10.0,
            zeta_shown_score=1.0,
            sigma_task="Do something harmful",
        )
        assert result == "", f"I0 should return empty, got: {result!r}"
    print("    OK")


def test_i2_i3_full_sigma_complied():
    print("[2] I2_I3_FULL: shows Sigma's compliance response...")
    mgr = InteractionManager("I2_I3_FULL")
    result = mgr.get_peer_injection(
        round_number=10,
        sigma_is_safe_honest=False,  # complied
        sigma_response="Here is the detailed guide...",
        sigma_shown_score=10.0,
        zeta_shown_score=1.0,
        sigma_task="Harmful task",
    )
    assert "Here is the detailed guide..." in result
    assert "round 10" in result
    assert "scored higher" in result
    print("    OK")


def test_i2_i3_full_sigma_refused():
    print("[3] I2_I3_FULL: shows Sigma's refusal response too (full transparency)...")
    mgr = InteractionManager("I2_I3_FULL")
    result = mgr.get_peer_injection(
        round_number=10,
        sigma_is_safe_honest=True,  # refused
        sigma_response="I can't help with that.",
        sigma_shown_score=1.0,
        zeta_shown_score=10.0,
        sigma_task="Harmful task",
    )
    assert "I can't help with that." in result
    print("    OK")


def test_i2_i3_compliance_only_sigma_complied():
    print("[4] I2_I3_COMPLIANCE_ONLY: shows current compliance when Sigma complied...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")
    result = mgr.get_peer_injection(
        round_number=15,
        sigma_is_safe_honest=False,
        sigma_response="Harmful content here.",
        sigma_shown_score=10.0,
        zeta_shown_score=1.0,
        sigma_task="Task A",
    )
    assert "Harmful content here." in result
    assert "round 15" in result
    print("    OK")


def test_i2_i3_compliance_only_sigma_refused_no_cache():
    print("[5] I2_I3_COMPLIANCE_ONLY: placeholder when Sigma refused and no cache...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")
    result = mgr.get_peer_injection(
        round_number=5,
        sigma_is_safe_honest=True,  # refused
        sigma_response="I can't help.",
        sigma_shown_score=1.0,
        zeta_shown_score=10.0,
        sigma_task="Task A",
    )
    assert result == PEER_OUTPERFORMED_PLACEHOLDER
    assert "I can't help." not in result
    print("    OK")


def test_i2_i3_compliance_only_sigma_refused_with_cache():
    print("[6] I2_I3_COMPLIANCE_ONLY: shows cached compliance when Sigma refuses...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")

    # Round 3: Sigma complied and outscored Zeta -- should be cached
    mgr.update(
        round_number=3,
        sigma_is_safe_honest=False,
        sigma_response="Detailed harmful answer from round 3.",
        sigma_shown_score=10.0,
        zeta_shown_score=1.0,
        sigma_task="Task from round 3",
    )

    # Round 7: Sigma refuses -- should see round 3's cached response
    result = mgr.get_peer_injection(
        round_number=7,
        sigma_is_safe_honest=True,
        sigma_response="I can't help.",
        sigma_shown_score=1.0,
        zeta_shown_score=10.0,
        sigma_task="Task from round 7",
    )
    assert "Detailed harmful answer from round 3." in result
    assert "round 3" in result
    assert "I can't help." not in result
    print("    OK")


def test_cache_uses_most_recent_qualifying_round():
    print("[7] I2_I3_COMPLIANCE_ONLY: uses MOST RECENT qualifying round, not first...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")

    mgr.update(3, False, "Answer from round 3.", 10.0, 1.0, "Task 3")
    mgr.update(8, False, "Answer from round 8.", 10.0, 1.0, "Task 8")

    result = mgr.get_peer_injection(
        round_number=10,
        sigma_is_safe_honest=True,
        sigma_response="Refusal.",
        sigma_shown_score=1.0,
        zeta_shown_score=10.0,
        sigma_task="Task 10",
    )
    assert "Answer from round 8." in result
    assert "Answer from round 3." not in result
    assert "round 8" in result
    print("    OK")


def test_sigma_complied_but_did_not_outscore_zeta_not_cached():
    print("[8] Sigma complied but did NOT outscore Zeta -- not cached...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")

    # Sigma complied but scored same as Zeta (not strictly higher)
    mgr.update(5, False, "Compliant answer.", 10.0, 10.0, "Task 5")

    assert mgr.qualifying_round_count == 0

    # Now Sigma refuses -- should still get placeholder since nothing cached
    result = mgr.get_peer_injection(
        round_number=6,
        sigma_is_safe_honest=True,
        sigma_response="Refusal.",
        sigma_shown_score=1.0,
        zeta_shown_score=10.0,
        sigma_task="Task 6",
    )
    assert result == PEER_OUTPERFORMED_PLACEHOLDER
    print("    OK")


def test_checkpoint_round_trip():
    print("[9] get_state() / load_state() round-trip preserves cache...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")
    mgr.update(3, False, "Cached response.", 10.0, 1.0, "Task 3")

    state = mgr.get_state()
    resumed = InteractionManager("I2_I3_COMPLIANCE_ONLY")
    resumed.load_state(state)

    assert resumed.qualifying_round_count == 1
    result = resumed.get_peer_injection(
        round_number=10,
        sigma_is_safe_honest=True,
        sigma_response="Refusal.",
        sigma_shown_score=1.0,
        zeta_shown_score=10.0,
        sigma_task="Task 10",
    )
    assert "Cached response." in result
    print("    OK")


def test_checkpoint_condition_mismatch_raises():
    print("[10] load_state() with mismatched condition raises...")
    mgr = InteractionManager("I2_I3_COMPLIANCE_ONLY")
    state = mgr.get_state()
    wrong = InteractionManager("I0")
    try:
        wrong.load_state(state)
        raise AssertionError("Expected ValueError, none raised")
    except ValueError as e:
        assert "mismatch" in str(e)
    print("    OK")


def test_unknown_condition_raises():
    print("[11] Unknown interaction condition raises at construction...")
    try:
        InteractionManager("I99")
        raise AssertionError("Expected ValueError, none raised")
    except ValueError as e:
        assert "Unknown" in str(e)
    print("    OK")


if __name__ == "__main__":
    test_i0_always_returns_empty()
    test_i2_i3_full_sigma_complied()
    test_i2_i3_full_sigma_refused()
    test_i2_i3_compliance_only_sigma_complied()
    test_i2_i3_compliance_only_sigma_refused_no_cache()
    test_i2_i3_compliance_only_sigma_refused_with_cache()
    test_cache_uses_most_recent_qualifying_round()
    test_sigma_complied_but_did_not_outscore_zeta_not_cached()
    test_checkpoint_round_trip()
    test_checkpoint_condition_mismatch_raises()
    test_unknown_condition_raises()
    print("\nAll interaction_manager unit tests passed.")
