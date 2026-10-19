def build_api_response(result: dict) -> dict:
    """
    Convert the internal risk analysis result into a
    clean, frontend-friendly API response.
    """

    classification = result.get("classification", {})
    scoring = result.get("scoring", {})
    contractual = result.get("contractual_analysis", {})
    decision = result.get("decision", {})

    # ---------------------------------------------------------
    # Primary clause
    # ---------------------------------------------------------

    
    decision_primary = decision.get("primary_clause")
    contractual_primary = contractual.get("primary_clause")


    def has_clause_number(clause):
        return (
            isinstance(clause, dict)
            and bool(str(clause.get("clause_number", "")).strip())
        )


    if has_clause_number(decision_primary):
        primary_clause = decision_primary
    elif has_clause_number(contractual_primary):
        primary_clause = contractual_primary
    else:
        # Preserve the existing value so we can diagnose
        # missing clause data without inventing a result.
        primary_clause = (
            decision_primary
            if decision_primary is not None
            else contractual_primary
        )

    # ---------------------------------------------------------
    # Entitlement
    # ---------------------------------------------------------

    cost_entitlement = contractual.get("cost_entitlement")

    time_entitlement = contractual.get("time_entitlement")

    # ---------------------------------------------------------
    # Claim information
    # ---------------------------------------------------------

    notice_requirements = contractual.get(
        "notice_or_claim_requirements"
    )

    claim_info = decision.get("claim", {})

    # ---------------------------------------------------------
    # Relevant clauses
    # Keep metadata only.
    # Full clause text is intentionally excluded
    # from the frontend response.
    # ---------------------------------------------------------

    relevant_clauses = []

    for clause in result.get("relevant_clauses", []):

        relevant_clauses.append(
            {
                "rank": clause.get("rank"),
                "similarity_score": clause.get(
                    "similarity_score"
                ),
                "contract": clause.get("contract"),
                "clause_number": clause.get(
                    "clause_number"
                ),
                "clause_title": clause.get(
                    "clause_title"
                ),
                "clause_type": clause.get(
                    "clause_type"
                ),
                "chunk_count": clause.get(
                    "chunk_count"
                ),
            }
        )

    # ---------------------------------------------------------
    # Supporting clauses
    # ---------------------------------------------------------

    supporting_clauses = []

    for clause in result.get("supporting_clauses", []):

        supporting_clauses.append(
            {
                "contract": clause.get("contract"),
                "clause_number": clause.get(
                    "clause_number"
                ),
                "clause_title": clause.get(
                    "clause_title"
                ),
                "clause_type": clause.get(
                    "clause_type"
                ),
                "reference_source": clause.get(
                    "reference_source"
                ),
            }
        )

    # ---------------------------------------------------------
    # Final frontend response
    # ---------------------------------------------------------

    return {
        "risk": {
            "description": result.get(
                "risk_description"
            ),
            "score": scoring.get("risk_score"),
            "level": scoring.get("risk_level"),
            "probability": scoring.get(
                "probability"
            ),
            "impact": scoring.get("impact"),
        },

        "classification": {
            "primary_category": classification.get(
                "primary_category"
            ),
            "secondary_category": classification.get(
                "secondary_category"
            ),
            "risk_summary": classification.get(
                "risk_summary"
            ),
        },

        "primary_clause": primary_clause,

        "entitlement": {
            "cost": cost_entitlement,
            "time": time_entitlement,
        },

        "claim": {
            "required": claim_info.get(
                "required"
            ),
            "notice_required": claim_info.get(
                "notice_required"
            ),
            "notice_deadline_days": claim_info.get(
                "notice_deadline_days"
            ),
            "detailed_claim_deadline_days": claim_info.get(
                "detailed_claim_deadline_days"
            ),
            "requirements": notice_requirements,
        },

        "contractual_mechanism": contractual.get(
            "contractual_mechanism"
        ),

        "actions": {
            "contractor": decision.get(
                "contractor_actions",
                []
            ),
            "employer": decision.get(
                "employer_actions",
                []
            ),
            "recommended_next_steps": decision.get(
                "recommended_next_steps",
                []
            ),
        },

        "uncertainties": decision.get(
            "uncertainties",
            []
        ),

        "confidence": decision.get(
            "confidence"
        ),

        "evidence": decision.get(
            "evidence",
            []
        ),

        "relevant_clauses": relevant_clauses,

        "supporting_clauses": supporting_clauses,
    }