"""
AgriNode AI - Environmental Risk Intelligence
Phase 5: Farm Decision Engine (src/decision_engine.py)

Single Responsibility:
    Translate abstract risk categories (from src/overall_risk.py) into
    concrete, operational field commands for farmers.

The engine consumes the full report dict returned by `compute_overall_risk()`
and emits a structured decision plan with:
    - Per-hazard action items (irrigation orders + field operations)
    - A unified action plan with priority ranking
    - Human-readable ASCII advisory for CLI / SMS / dashboard display

Action Matrix (from Phase_Wise_Plan.md Sec.5):
    Drought  HIGH/SEVERE -> deficit drip irrigation, organic mulch, suspend fertilizer
    Flood    HIGH/SEVERE -> pump shutdown, open drainage, clear furrows, anti-fungal prep
    Heat     HIGH/SEVERE -> canopy cooling, shift labor before 10 AM, no mid-day spraying
    Low/Moderate          -> routine monitoring & seasonal maintenance
"""

from __future__ import annotations
from typing import Dict, Any, List


# -- Priority constants ---------------------------------------------------------
PRIORITY_CRITICAL = "CRITICAL"
PRIORITY_HIGH     = "HIGH"
PRIORITY_ROUTINE  = "ROUTINE"

TIER_PRIORITY = {
    "SEVERE":   PRIORITY_CRITICAL,
    "HIGH":     PRIORITY_HIGH,
    "MODERATE": PRIORITY_ROUTINE,
    "LOW":      PRIORITY_ROUTINE,
}


# -- Action Matrix --------------------------------------------------------------
# Each entry: { "irrigation": [...], "field_ops": [...] }

DROUGHT_ACTIONS = {
    "SEVERE": {
        "irrigation": [
            "IMMEDIATE: Trigger deficit drip irrigation cycle targeting root-zone replenishment.",
            "Reduce irrigation intervals to maximize soil moisture retention.",
        ],
        "field_ops": [
            "Apply organic mulch (straw / crop residue) to all exposed beds to reduce evapotranspiration.",
            "Suspend ALL non-essential fertilizer applications -- dry soil limits nutrient uptake and risks salt burn.",
            "Consider emergency foliar micro-nutrient spray if wilting is observed.",
        ],
    },
    "HIGH": {
        "irrigation": [
            "Trigger deficit drip irrigation cycle; target root-zone replenishment.",
        ],
        "field_ops": [
            "Apply organic mulch to reduce evapotranspiration.",
            "Suspend non-essential fertilizer applications until moisture conditions improve.",
        ],
    },
    "MODERATE": {
        "irrigation": [
            "Continue scheduled irrigation; monitor soil moisture sensors closely.",
        ],
        "field_ops": [
            "Routine field monitoring. Watch for early signs of moisture stress.",
        ],
    },
    "LOW": {
        "irrigation": [
            "Normal irrigation schedule. No drought-related adjustments needed.",
        ],
        "field_ops": [
            "Routine seasonal maintenance.",
        ],
    },
}

FLOOD_ACTIONS = {
    "SEVERE": {
        "irrigation": [
            "IMMEDIATE: Shut down ALL pumps and irrigation lines.",
            "Close all intake valves to prevent backflow contamination.",
        ],
        "field_ops": [
            "Open ALL drainage channels to maximum capacity.",
            "Clear field furrows and ditches to prevent waterlogging.",
            "Move harvested produce and inputs to elevated storage.",
            "Prepare preventative anti-fungal spray for application immediately post-recession.",
        ],
    },
    "HIGH": {
        "irrigation": [
            "Immediate shutdown of all pumps and irrigation lines.",
        ],
        "field_ops": [
            "Open drainage channels; clear field furrows to prevent waterlogging.",
            "Prepare preventative anti-fungal spray for post-recession application.",
        ],
    },
    "MODERATE": {
        "irrigation": [
            "Reduce irrigation output. Monitor drainage flow rates.",
        ],
        "field_ops": [
            "Inspect drainage channels for blockages. Clear if needed.",
        ],
    },
    "LOW": {
        "irrigation": [
            "Normal irrigation schedule. No flood-related adjustments needed.",
        ],
        "field_ops": [
            "Routine seasonal maintenance.",
        ],
    },
}

HEAT_ACTIONS = {
    "SEVERE": {
        "irrigation": [
            "IMMEDIATE: Schedule early-morning (before 06:00 AM) light canopy cooling cycle.",
            "Increase irrigation frequency with shallow applications to cool root zone.",
        ],
        "field_ops": [
            "Shift ALL manual farm labor to before 10:00 AM -- heat illness risk is extreme.",
            "STOP all mid-day chemical spraying to prevent leaf scorching and phytotoxicity.",
            "Deploy shade nets on vulnerable nurseries and transplants if available.",
            "Ensure livestock have access to shade and clean water.",
        ],
    },
    "HIGH": {
        "irrigation": [
            "Schedule early-morning light canopy cooling cycle.",
        ],
        "field_ops": [
            "Shift all manual farm labor to before 10:00 AM.",
            "Avoid mid-day chemical spraying to prevent leaf scorching.",
        ],
    },
    "MODERATE": {
        "irrigation": [
            "Consider light morning irrigation if canopy temperature exceeds 38 C.",
        ],
        "field_ops": [
            "Monitor crop canopy for signs of heat stress (leaf curling, wilting).",
        ],
    },
    "LOW": {
        "irrigation": [
            "Normal irrigation schedule. No heat-related adjustments needed.",
        ],
        "field_ops": [
            "Routine seasonal maintenance.",
        ],
    },
}

# Map hazard names to their action matrices
ACTION_MATRICES = {
    "DROUGHT": DROUGHT_ACTIONS,
    "FLOOD":   FLOOD_ACTIONS,
    "HEAT":    HEAT_ACTIONS,
}

# Map hazard names to the report sub-dict key and tier key
HAZARD_KEYS = {
    "DROUGHT": ("drought", "drought_tier", "drought_score"),
    "FLOOD":   ("flood",   "flood_tier",   "flood_score"),
    "HEAT":    ("heat",    "heat_tier",     "heat_score"),
}


# -- Core decision function -----------------------------------------------------

def generate_decision(risk_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured farm decision plan from a risk report.

    Parameters
    ----------
    risk_report : dict
        Output of `src.overall_risk.compute_overall_risk()`.

    Returns
    -------
    dict with keys:
        - overall_tier        : str   (echoed from risk report)
        - overall_score       : float (echoed)
        - primary_hazard      : str   (echoed)
        - priority_level      : str   (CRITICAL / HIGH / ROUTINE)
        - hazard_decisions     : dict  keyed by hazard name -> {tier, score, irrigation, field_ops}
        - unified_action_plan : list[dict]  all actions merged and priority-sorted
        - spatial_context     : dict  (echoed)
        - advisory_summary    : str   one-liner human-readable summary
    """
    overall_tier   = risk_report["overall_tier"]
    overall_score  = risk_report["overall_score"]
    primary_hazard = risk_report["primary_hazard"]
    priority_level = TIER_PRIORITY.get(overall_tier, PRIORITY_ROUTINE)

    hazard_decisions: Dict[str, Dict[str, Any]] = {}
    unified_actions: List[Dict[str, Any]] = []

    for hazard_name, (sub_key, tier_key, score_key) in HAZARD_KEYS.items():
        sub_result = risk_report[sub_key]
        tier  = sub_result[tier_key]
        score = sub_result[score_key]

        matrix = ACTION_MATRICES[hazard_name]
        actions = matrix.get(tier, matrix["LOW"])

        hazard_decisions[hazard_name] = {
            "tier": tier,
            "score": score,
            "priority": TIER_PRIORITY.get(tier, PRIORITY_ROUTINE),
            "irrigation": actions["irrigation"],
            "field_ops": actions["field_ops"],
        }

        # Add to unified plan with source tagging
        action_priority = TIER_PRIORITY.get(tier, PRIORITY_ROUTINE)
        for action_text in actions["irrigation"]:
            unified_actions.append({
                "category": "IRRIGATION",
                "hazard": hazard_name,
                "priority": action_priority,
                "tier": tier,
                "action": action_text,
            })
        for action_text in actions["field_ops"]:
            unified_actions.append({
                "category": "FIELD_OPS",
                "hazard": hazard_name,
                "priority": action_priority,
                "tier": tier,
                "action": action_text,
            })

    # Sort unified actions: CRITICAL first, then HIGH, then ROUTINE
    priority_order = {PRIORITY_CRITICAL: 0, PRIORITY_HIGH: 1, PRIORITY_ROUTINE: 2}
    unified_actions.sort(key=lambda a: (priority_order.get(a["priority"], 9), a["hazard"]))

    # De-duplicate routine "Normal irrigation schedule" / "Routine seasonal maintenance"
    # entries when higher-priority actions already exist for the same category
    unified_actions = _deduplicate_routine(unified_actions)

    advisory = _build_advisory_summary(overall_tier, primary_hazard, hazard_decisions)

    return {
        "overall_tier": overall_tier,
        "overall_score": overall_score,
        "primary_hazard": primary_hazard,
        "priority_level": priority_level,
        "hazard_decisions": hazard_decisions,
        "unified_action_plan": unified_actions,
        "spatial_context": risk_report.get("spatial_context", {}),
        "advisory_summary": advisory,
    }


def _deduplicate_routine(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove LOW/MODERATE 'no adjustment needed' or 'routine maintenance'
    actions when there are already CRITICAL or HIGH actions in the same
    category (IRRIGATION or FIELD_OPS).
    """
    has_elevated = {
        "IRRIGATION": False,
        "FIELD_OPS":  False,
    }
    for a in actions:
        if a["priority"] in (PRIORITY_CRITICAL, PRIORITY_HIGH):
            has_elevated[a["category"]] = True

    result = []
    for a in actions:
        if a["priority"] == PRIORITY_ROUTINE and has_elevated.get(a["category"], False):
            # Skip generic routine actions when a real alert exists in same category
            text_lower = a["action"].lower()
            if ("no " in text_lower and "adjustment" in text_lower) or \
               "routine seasonal maintenance" in text_lower or \
               "normal irrigation schedule" in text_lower:
                continue
        result.append(a)
    return result


def _build_advisory_summary(
    overall_tier: str,
    primary_hazard: str,
    hazard_decisions: Dict[str, Dict[str, Any]],
) -> str:
    """Build a one-line human-readable advisory string."""
    if overall_tier == "SEVERE":
        return (
            f"SEVERE ALERT: {primary_hazard} risk is critical. "
            f"Immediate protective actions required -- see action plan."
        )
    elif overall_tier == "HIGH":
        return (
            f"HIGH ALERT: Elevated {primary_hazard.lower()} risk detected. "
            f"Activate precautionary field operations immediately."
        )
    elif overall_tier == "MODERATE":
        return (
            f"ADVISORY: Moderate environmental risk. "
            f"Increase monitoring frequency; be prepared to act."
        )
    else:
        return (
            "ALL CLEAR: Low environmental risk. "
            "Continue routine monitoring and scheduled maintenance."
        )


# -- Human-readable CLI formatter ----------------------------------------------

def format_decision_report(decision: Dict[str, Any]) -> str:
    """
    Render the decision plan as a human-readable ASCII report for CLI output.
    Compatible with Windows cp1252 terminal (no emoji / Unicode symbols).
    """
    ctx = decision.get("spatial_context", {})
    lines = [
        "=" * 70,
        "  AgriNode AI -- Farm Decision Advisory",
        "=" * 70,
        f"  District : {ctx.get('district', 'N/A')}, {ctx.get('state', 'N/A')}",
        f"  Date     : {ctx.get('date', 'N/A')}",
        "-" * 70,
        f"  OVERALL RISK  : [{decision['overall_tier']}]  (score: {decision['overall_score']:.1f}/100)",
        f"  Priority      : {decision['priority_level']}",
        f"  Primary Hazard: {decision['primary_hazard']}",
        "-" * 70,
        f"  >> {decision['advisory_summary']}",
        "-" * 70,
    ]

    # Per-hazard breakdown
    for hazard_name in ["DROUGHT", "FLOOD", "HEAT"]:
        hd = decision["hazard_decisions"][hazard_name]
        lines.append(f"\n  [{hazard_name}]  Tier: {hd['tier']}  |  Score: {hd['score']:.1f}/100")
        lines.append(f"  {'~' * 40}")

        if hd["irrigation"]:
            lines.append("    Irrigation Orders:")
            for item in hd["irrigation"]:
                lines.append(f"      * {item}")

        if hd["field_ops"]:
            lines.append("    Field Operations:")
            for item in hd["field_ops"]:
                lines.append(f"      * {item}")

    # Unified action summary (only elevated actions)
    elevated = [a for a in decision["unified_action_plan"]
                if a["priority"] in (PRIORITY_CRITICAL, PRIORITY_HIGH)]
    if elevated:
        lines.append("\n" + "-" * 70)
        lines.append("  PRIORITY ACTIONS (requires immediate attention):")
        lines.append("  " + "-" * 48)
        for i, a in enumerate(elevated, 1):
            lines.append(
                f"    {i}. [{a['priority']}] [{a['hazard']}] {a['action']}"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
