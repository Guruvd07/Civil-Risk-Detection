"""
FIDIC Risk Scoring

Calculates risk score from probability and impact.

Scale:
1 = Very Low
2 = Low
3 = Moderate
4 = High
5 = Very High

Risk Score = Probability × Impact
"""

PROBABILITY_LEVELS = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}

IMPACT_LEVELS = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}


def calculate_risk_score(probability: int, impact: int) -> dict:
    """
    Calculate risk score and risk level.
    """

    if probability not in PROBABILITY_LEVELS:
        raise ValueError("Probability must be between 1 and 5.")

    if impact not in IMPACT_LEVELS:
        raise ValueError("Impact must be between 1 and 5.")

    score = probability * impact

    if score <= 4:
        level = "LOW"
    elif score <= 9:
        level = "MEDIUM"
    elif score <= 16:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "probability": probability,
        "probability_label": PROBABILITY_LEVELS[probability],
        "impact": impact,
        "impact_label": IMPACT_LEVELS[impact],
        "risk_score": score,
        "risk_level": level,
    }


if __name__ == "__main__":

    test_cases = [
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]

    print("=" * 60)
    print("FIDIC RISK SCORING")
    print("=" * 60)

    for probability, impact in test_cases:

        result = calculate_risk_score(
            probability,
            impact
        )

        print(
            f"\nProbability : {result['probability']} "
            f"({result['probability_label']})"
        )

        print(
            f"Impact      : {result['impact']} "
            f"({result['impact_label']})"
        )

        print(f"Risk Score  : {result['risk_score']}")
        print(f"Risk Level  : {result['risk_level']}")