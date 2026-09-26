"""Failure attribution must preserve local damage, overlap, and NA denominators."""

from llm_memory_editability.bios_analysis import failure_flags, failure_signature


def point():
    return {
        "E_root": 1.0,
        "E_member": 1.0,
        "D": 1.0,
        "U_accuracy": 0.9999,
        "U_heldout_destruction": {
            g: {"known": 100, "broken": 0, "rate": 0.0}
            for g in ("0", "1", "2", "3", "2_base", "2_derived")
        },
    }


def test_local_damage_and_overlapping_failures_are_not_averaged_away():
    p = point()
    p["D"] = 0.5
    p["U_heldout_destruction"]["0"] = {"known": 2, "broken": 1, "rate": 0.5}
    assert failure_signature(failure_flags(p)) == "D+U"
    p["E_root"] = 2 / 3
    assert failure_signature(failure_flags(p)) == "E+D+U"


def test_separate_base_retention_gate_and_empty_denominator():
    p = point()
    p["U_heldout_destruction"]["2"]["rate"] = 0.004
    p["U_heldout_destruction"]["2_base"]["rate"] = 0.02
    assert failure_signature(failure_flags(p)) == "U"
    assert failure_signature(failure_flags(p, damage=0.02)) == "pass"
    p["U_heldout_destruction"]["0"] = {"known": 0, "broken": 0, "rate": None}
    assert failure_signature(failure_flags(p)) == "unevaluable"
