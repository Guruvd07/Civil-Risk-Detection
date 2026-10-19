"""
FIDIC Construction Risk Taxonomy

Defines the standardized risk categories used by the
risk classification and analysis pipeline.
"""

RISK_TAXONOMY = {
    "TIME_DELAY": {
        "name": "Time / Delay",
        "description": "Risks affecting project schedule, completion dates, or progress.",
        "keywords": [
            "delay",
            "late",
            "completion",
            "schedule",
            "extension of time",
            "slow progress",
            "disruption",
        ],
    },

    "COST_FINANCIAL": {
        "name": "Cost / Financial",
        "description": "Risks causing additional costs, payment issues, or financial losses.",
        "keywords": [
            "cost",
            "payment",
            "price",
            "inflation",
            "financial",
            "compensation",
            "additional cost",
        ],
    },

    "TECHNICAL_DESIGN": {
        "name": "Technical / Design",
        "description": "Risks related to engineering, design, specifications, or technical requirements.",
        "keywords": [
            "design",
            "drawing",
            "specification",
            "engineering",
            "technical",
            "design error",
            "technical requirement",
        ],
    },

    "SITE_PHYSICAL": {
        "name": "Site / Physical Conditions",
        "description": "Risks arising from physical, geological, hydrological, or site conditions.",
        "keywords": [
            "site",
            "ground",
            "soil",
            "geological",
            "physical condition",
            "unforeseeable condition",
            "underground",
            "hydrological",
        ],
    },

    "HEALTH_SAFETY": {
        "name": "Health & Safety",
        "description": "Risks involving worker health, safety, accidents, or site protection.",
        "keywords": [
            "safety",
            "accident",
            "injury",
            "health",
            "worker",
            "personnel",
            "hazard",
        ],
    },

    "QUALITY_DEFECTS": {
        "name": "Quality / Defects",
        "description": "Risks involving defective work, quality failures, testing, or non-conformance.",
        "keywords": [
            "defect",
            "quality",
            "testing",
            "failure",
            "non-conformance",
            "remedy",
            "inspection",
        ],
    },

    "LEGAL_CONTRACTUAL": {
        "name": "Legal / Contractual",
        "description": "Risks related to contractual obligations, interpretation, notices, or legal requirements.",
        "keywords": [
            "contract",
            "contractual",
            "obligation",
            "notice",
            "breach",
            "legal",
            "compliance",
        ],
    },

    "ENVIRONMENTAL": {
        "name": "Environmental",
        "description": "Risks related to environmental impacts, pollution, permits, or environmental compliance.",
        "keywords": [
            "environment",
            "pollution",
            "environmental",
            "waste",
            "permit",
            "ecological",
        ],
    },

    "EMPLOYER": {
        "name": "Employer-related",
        "description": "Risks caused by or associated with the Employer's actions, instructions, or obligations.",
        "keywords": [
            "employer",
            "employer delay",
            "instruction",
            "access",
            "approval",
            "payment by employer",
        ],
    },

    "CONTRACTOR": {
        "name": "Contractor-related",
        "description": "Risks caused by Contractor performance, resources, obligations, or failures.",
        "keywords": [
            "contractor",
            "contractor failure",
            "contractor obligation",
            "resources",
            "workmanship",
            "performance",
        ],
    },

    "EXCEPTIONAL_EVENTS": {
        "name": "Force Majeure / Exceptional Events",
        "description": "Risks arising from extraordinary events outside the parties' reasonable control.",
        "keywords": [
            "force majeure",
            "exceptional event",
            "war",
            "natural disaster",
            "earthquake",
            "flood",
            "emergency",
        ],
    },

    "CLAIMS_DISPUTES": {
        "name": "Claims / Disputes",
        "description": "Risks involving claims, disagreements, adjudication, arbitration, or dispute resolution.",
        "keywords": [
            "claim",
            "dispute",
            "adjudication",
            "arbitration",
            "settlement",
            "disagreement",
        ],
    },
}


def get_categories():
    """Return all available risk category codes."""
    return list(RISK_TAXONOMY.keys())


def get_category(category_code):
    """Return details for a specific risk category."""
    return RISK_TAXONOMY.get(category_code)


if __name__ == "__main__":

    print("=" * 60)
    print("FIDIC RISK TAXONOMY")
    print("=" * 60)

    for code, category in RISK_TAXONOMY.items():
        print(f"\n{code}")
        print(f"  Name: {category['name']}")
        print(f"  Description: {category['description']}")

    print("\n" + "=" * 60)
    print(f"Total categories: {len(RISK_TAXONOMY)}")
    print("=" * 60)