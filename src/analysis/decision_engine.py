def build_risk_decision(contractual_analysis: dict, scoring: dict) -> dict:
    """
    Convert contractual analysis + risk scoring into a
    structured, deterministic risk decision.
    """

    notice_requirements = contractual_analysis.get(
        "notice_or_claim_requirements", ""
    )

    evidence = contractual_analysis.get("evidence", [])

    notice_text = notice_requirements.lower()

    claim_required = (
        "claim" in notice_text
        or "20.2" in notice_requirements
    )

    notice_required = (
        "notice" in notice_text
        or "20.2" in notice_requirements
        or "1.3" in notice_requirements
    )

    notice_deadline_days = 28 if "28" in notice_requirements else None

    detailed_claim_deadline_days = (
        84 if "84" in notice_requirements else None
    )

    # ------------------------------------------------------------------
    # Evidence-based confidence
    # ------------------------------------------------------------------

    evidence_count = len(evidence)

    has_primary_clause = bool(
        contractual_analysis.get("primary_clause")
    )

    has_cost_analysis = bool(
        contractual_analysis.get("cost_entitlement")
    )

    has_time_analysis = bool(
        contractual_analysis.get("time_entitlement")
    )

    has_claim_analysis = bool(
        contractual_analysis.get("notice_or_claim_requirements")
    )

    confidence_points = 0

    if has_primary_clause:
        confidence_points += 2

    if evidence_count >= 3:
        confidence_points += 2
    elif evidence_count >= 1:
        confidence_points += 1

    if has_cost_analysis:
        confidence_points += 1

    if has_time_analysis:
        confidence_points += 1

    if has_claim_analysis:
        confidence_points += 1


    if confidence_points >= 6:
        confidence = "High"
    elif confidence_points >= 3:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "risk_level": scoring.get("risk_level"),

        "primary_clause": contractual_analysis.get(
            "primary_clause"
        ),

        "cost": {
            "status": contractual_analysis.get(
                "cost_entitlement"
            )
        },

        "time": {
            "status": contractual_analysis.get(
                "time_entitlement"
            )
        },

        "claim": {
            "required": claim_required,
            "notice_required": notice_required,
            "notice_deadline_days": notice_deadline_days,
            "detailed_claim_deadline_days": detailed_claim_deadline_days,
        },

        "contractor_actions": contractual_analysis.get(
            "contractor_actions", []
        ),

        "employer_actions": contractual_analysis.get(
            "employer_actions", []
        ),

        "recommended_next_steps": contractual_analysis.get(
            "recommended_next_steps", []
        ),

        "uncertainties": contractual_analysis.get(
            "uncertainties", []
        ),

        "confidence": {
            "level": confidence,
            "evidence_count": evidence_count,
            "reason": (
                f"Confidence based on {evidence_count} evidence items, "
                f"primary clause availability, and completeness of "
                f"cost, time, and claim analysis."
            ),
        },

        "evidence": evidence,
    }