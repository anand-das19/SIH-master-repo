"""
AgriNode AI - Phase 5 Decision Engine Tests
tests/test_phase5.py

Verifies:
  1. Action matrix completeness — every tier maps to actions for all 3 hazards
  2. Priority escalation — SEVERE → CRITICAL, HIGH → HIGH, LOW/MODERATE → ROUTINE
  3. Drought HIGH/SEVERE triggers correct irrigation + field ops
  4. Flood HIGH/SEVERE triggers correct irrigation + field ops
  5. Heat HIGH/SEVERE triggers correct irrigation + field ops
  6. Low-risk report → all ROUTINE, no elevated actions
  7. Multi-hazard deduplication — routine generics removed when alerts exist
  8. Full pipeline integration — benchmark cases produce valid decisions
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.decision_engine import (
    generate_decision,
    format_decision_report,
    DROUGHT_ACTIONS,
    FLOOD_ACTIONS,
    HEAT_ACTIONS,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_ROUTINE,
)


# ── Helper: Build a minimal risk report ────────────────────────────────────────

def _make_report(
    drought_tier="LOW", drought_score=10.0,
    flood_tier="LOW",   flood_score=10.0,
    heat_tier="LOW",    heat_score=10.0,
    overall_tier=None,  primary_hazard=None,
):
    """Construct a minimal risk report dict matching compute_overall_risk output."""
    scores = {"DROUGHT": drought_score, "FLOOD": flood_score, "HEAT": heat_score}
    if overall_tier is None:
        # Simple dominance
        for t in [drought_tier, flood_tier, heat_tier]:
            if t == "SEVERE":
                overall_tier = "SEVERE"
                break
        else:
            for t in [drought_tier, flood_tier, heat_tier]:
                if t == "HIGH":
                    overall_tier = "HIGH"
                    break
            else:
                for t in [drought_tier, flood_tier, heat_tier]:
                    if t == "MODERATE":
                        overall_tier = "MODERATE"
                        break
                else:
                    overall_tier = "LOW"

    if primary_hazard is None:
        primary_hazard = max(scores, key=scores.get)

    return {
        "overall_tier": overall_tier,
        "overall_score": round(sum(scores.values()) / 3.0, 2),
        "primary_hazard": primary_hazard,
        "drought": {
            "drought_score": drought_score,
            "drought_tier": drought_tier,
            "drought_factors": {},
        },
        "flood": {
            "flood_score": flood_score,
            "flood_tier": flood_tier,
            "flood_factors": {},
        },
        "heat": {
            "heat_score": heat_score,
            "heat_tier": heat_tier,
            "heat_factors": {},
        },
        "spatial_context": {
            "district": "TestDistrict",
            "state": "TestState",
            "date": "2024-06-15",
            "distance_km": 0.0,
        },
        "inference_latency_ms": 0.05,
    }


# ── Test 1: Action matrix completeness ────────────────────────────────────────

def test_action_matrix_completeness():
    """Every tier must have both 'irrigation' and 'field_ops' lists for all 3 hazards."""
    for name, matrix in [("DROUGHT", DROUGHT_ACTIONS),
                         ("FLOOD", FLOOD_ACTIONS),
                         ("HEAT", HEAT_ACTIONS)]:
        for tier in ["LOW", "MODERATE", "HIGH", "SEVERE"]:
            assert tier in matrix, f"{name} missing tier {tier}"
            assert "irrigation" in matrix[tier], f"{name}/{tier} missing irrigation"
            assert "field_ops" in matrix[tier], f"{name}/{tier} missing field_ops"
            assert len(matrix[tier]["irrigation"]) > 0, f"{name}/{tier} empty irrigation"
            assert len(matrix[tier]["field_ops"]) > 0, f"{name}/{tier} empty field_ops"


# ── Test 2: Priority escalation ───────────────────────────────────────────────

def test_priority_escalation():
    """SEVERE → CRITICAL, HIGH → HIGH, MODERATE/LOW → ROUTINE."""
    report_severe = _make_report(drought_tier="SEVERE", drought_score=90.0)
    d_severe = generate_decision(report_severe)
    assert d_severe["priority_level"] == PRIORITY_CRITICAL

    report_high = _make_report(flood_tier="HIGH", flood_score=70.0)
    d_high = generate_decision(report_high)
    assert d_high["priority_level"] == PRIORITY_HIGH

    report_mod = _make_report(heat_tier="MODERATE", heat_score=45.0)
    d_mod = generate_decision(report_mod)
    assert d_mod["priority_level"] == PRIORITY_ROUTINE

    report_low = _make_report()
    d_low = generate_decision(report_low)
    assert d_low["priority_level"] == PRIORITY_ROUTINE


# ── Test 3: Drought SEVERE actions ────────────────────────────────────────────

def test_drought_severe_actions():
    """Drought SEVERE must trigger drip irrigation + mulch + fertilizer suspend."""
    report = _make_report(drought_tier="SEVERE", drought_score=90.0)
    decision = generate_decision(report)
    dd = decision["hazard_decisions"]["DROUGHT"]

    assert dd["tier"] == "SEVERE"
    assert dd["priority"] == PRIORITY_CRITICAL

    irr_text = " ".join(dd["irrigation"]).lower()
    assert "drip irrigation" in irr_text
    assert "root-zone" in irr_text

    ops_text = " ".join(dd["field_ops"]).lower()
    assert "mulch" in ops_text
    assert "fertilizer" in ops_text


# ── Test 4: Flood HIGH actions ────────────────────────────────────────────────

def test_flood_high_actions():
    """Flood HIGH must trigger pump shutdown + drainage + anti-fungal prep."""
    report = _make_report(flood_tier="HIGH", flood_score=70.0)
    decision = generate_decision(report)
    fd = decision["hazard_decisions"]["FLOOD"]

    assert fd["tier"] == "HIGH"
    assert fd["priority"] == PRIORITY_HIGH

    irr_text = " ".join(fd["irrigation"]).lower()
    assert "shutdown" in irr_text

    ops_text = " ".join(fd["field_ops"]).lower()
    assert "drainage" in ops_text
    assert "anti-fungal" in ops_text


# ── Test 5: Heat SEVERE actions ───────────────────────────────────────────────

def test_heat_severe_actions():
    """Heat SEVERE must trigger canopy cooling + labor shift + no mid-day spray."""
    report = _make_report(heat_tier="SEVERE", heat_score=90.0)
    decision = generate_decision(report)
    hd = decision["hazard_decisions"]["HEAT"]

    assert hd["tier"] == "SEVERE"
    assert hd["priority"] == PRIORITY_CRITICAL

    irr_text = " ".join(hd["irrigation"]).lower()
    assert "canopy cooling" in irr_text
    assert "morning" in irr_text

    ops_text = " ".join(hd["field_ops"]).lower()
    assert "10:00 am" in ops_text
    assert "mid-day" in ops_text
    assert "spraying" in ops_text


# ── Test 6: All-low report → routine only ─────────────────────────────────────

def test_all_low_is_routine():
    """When all hazards are LOW, every action should be ROUTINE."""
    report = _make_report()
    decision = generate_decision(report)

    assert decision["priority_level"] == PRIORITY_ROUTINE
    assert "ALL CLEAR" in decision["advisory_summary"]

    for hazard_name in ["DROUGHT", "FLOOD", "HEAT"]:
        hd = decision["hazard_decisions"][hazard_name]
        assert hd["priority"] == PRIORITY_ROUTINE

    # No elevated actions in unified plan
    elevated = [a for a in decision["unified_action_plan"]
                if a["priority"] in (PRIORITY_CRITICAL, PRIORITY_HIGH)]
    assert len(elevated) == 0


# ── Test 7: Deduplication removes routine generics ────────────────────────────

def test_deduplication():
    """When drought is SEVERE, routine 'no adjustment needed' irrigation lines
    from flood LOW should be removed from unified plan."""
    report = _make_report(drought_tier="SEVERE", drought_score=90.0)
    decision = generate_decision(report)

    irr_actions = [a for a in decision["unified_action_plan"]
                   if a["category"] == "IRRIGATION"]

    # Should not contain any "No ... adjustments needed" lines
    for a in irr_actions:
        text_lower = a["action"].lower()
        assert not ("no " in text_lower and "adjustment" in text_lower), \
            f"Routine generic not deduplicated: {a['action']}"


# ── Test 8: Format report renders without error ───────────────────────────────

def test_format_report_renders():
    """format_decision_report should return a non-empty ASCII string for any tier."""
    for tier_combo in [
        {},
        {"drought_tier": "SEVERE", "drought_score": 90.0},
        {"flood_tier": "HIGH", "flood_score": 70.0},
        {"heat_tier": "MODERATE", "heat_score": 45.0},
        {"drought_tier": "HIGH", "drought_score": 70.0,
         "flood_tier": "SEVERE", "flood_score": 92.0,
         "heat_tier": "HIGH", "heat_score": 75.0},
    ]:
        report = _make_report(**tier_combo)
        decision = generate_decision(report)
        text = format_decision_report(decision)
        assert isinstance(text, str)
        assert len(text) > 100
        # Verify ASCII-safe (no emoji)
        text.encode("ascii")


# ── Test 9: Full pipeline integration with a benchmark case ───────────────────

def test_pipeline_integration():
    """Run the full pipeline (features → risk → decision) on Ernakulam 2018 flood."""
    from src.features import get_feature_extractor
    from src.overall_risk import compute_overall_risk

    fe = get_feature_extractor()
    features = fe.extract_features(lat=9.98, lon=76.27, query_date="2018-08-16")
    risk_report = compute_overall_risk(features)
    decision = generate_decision(risk_report)

    # Ernakulam 2018 was a major flood → overall should be HIGH or SEVERE
    assert decision["overall_tier"] in ("HIGH", "SEVERE")
    assert decision["priority_level"] in (PRIORITY_CRITICAL, PRIORITY_HIGH)

    # Flood actions should include shutdown
    fd = decision["hazard_decisions"]["FLOOD"]
    assert fd["tier"] in ("HIGH", "SEVERE")
    irr_text = " ".join(fd["irrigation"]).lower()
    assert ("shutdown" in irr_text) or ("shut down" in irr_text)

    # Advisory should not be "ALL CLEAR"
    assert "ALL CLEAR" not in decision["advisory_summary"]

    # Decision report should render
    text = format_decision_report(decision)
    assert "Farm Decision Advisory" in text
    print("\n" + text)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
