import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from google import genai

from src.taxonomy.risk_taxonomy import RISK_TAXONOMY


# ==========================================================
# Gemini Setup
# ==========================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"


# ==========================================================
# Semantic Classification Model
# ==========================================================

SEMANTIC_MODEL_NAME = "BAAI/bge-small-en-v1.5"

SEMANTIC_MODEL = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)


# ==========================================================
# Fast Rule-Based Classification Profiles
# ==========================================================

CATEGORY_PROFILES = {

    # ------------------------------------------------------
    # TIME / DELAY
    # ------------------------------------------------------

    "TIME_DELAY": {
        "strong": [
            "extension of time",
            "extension of time for completion",
            "delay to completion",
            "delay in completion",
            "delayed completion",
            "completion date",
            "time for completion",
            "progress of the works",
            "slow progress",
        ],
        "medium": [
            "delay",
            "late",
            "completion",
            "schedule",
            "progress",
            "disruption",
        ],
    },


    # ------------------------------------------------------
    # COST / FINANCIAL
    # ------------------------------------------------------

    "COST_FINANCIAL": {
        "strong": [
            "additional cost",
            "additional costs",
            "additional payment",
            "additional expenditure",
            "cost incurred",
            "costs incurred",
            "financial loss",
            "financial consequences",
            "compensation",
        ],
        "medium": [
            "cost",
            "costs",
            "payment",
            "price",
            "inflation",
            "financial",
        ],
    },


    # ------------------------------------------------------
    # TECHNICAL / DESIGN
    # ------------------------------------------------------

    "TECHNICAL_DESIGN": {
        "strong": [
            "design error",
            "design defect",
            "design requirement",
            "technical requirement",
            "technical specification",
            "engineering design",
            "contractor's design",
        ],
        "medium": [
            "design",
            "drawing",
            "specification",
            "engineering",
            "technical",
        ],
    },


    # ------------------------------------------------------
    # SITE / PHYSICAL CONDITIONS
    # ------------------------------------------------------

    "SITE_PHYSICAL": {
        "strong": [
            "unforeseeable physical conditions",
            "unforeseen physical conditions",
            "unforeseeable conditions",
            "physical conditions",
            "ground conditions",
            "subsurface conditions",
            "geological conditions",
            "underground conditions",
            "hydrological conditions",
        ],
        "medium": [
            "site conditions",
            "ground",
            "soil",
            "geological",
            "underground",
            "hydrological",
        ],
    },


    # ------------------------------------------------------
    # HEALTH / SAFETY
    # ------------------------------------------------------

    "HEALTH_SAFETY": {
        "strong": [
            "health and safety",
            "health & safety",
            "safety of personnel",
            "safety of workers",
            "accident at site",
            "worker injury",
            "injury to personnel",
            "site safety",
        ],
        "medium": [
            "safety",
            "accident",
            "injury",
            "health",
            "worker",
            "personnel",
            "hazard",
        ],
    },


    # ------------------------------------------------------
    # QUALITY / DEFECTS
    # ------------------------------------------------------

    "QUALITY_DEFECTS": {
        "strong": [
            "defective work",
            "defective works",
            "defect in the works",
            "defects in the works",
            "remedy defects",
            "remedial work",
            "non-conformance",
            "nonconformance",
            "quality of the works",
        ],
        "medium": [
            "defect",
            "defects",
            "quality",
            "testing",
            "failure",
            "remedy",
            "inspection",
        ],
    },


    # ------------------------------------------------------
    # LEGAL / CONTRACTUAL
    # ------------------------------------------------------

    "LEGAL_CONTRACTUAL": {
        "strong": [
            "contractual obligation",
            "contractual obligations",
            "contractual requirement",
            "breach of contract",
            "contract interpretation",
            "legal requirement",
            "contractual notice",
            "compliance with the contract",
        ],
        "medium": [
            "contract",
            "contractual",
            "obligation",
            "obligations",
            "breach",
            "legal",
            "compliance",
            "notice",
        ],
    },


    # ------------------------------------------------------
    # ENVIRONMENTAL
    # ------------------------------------------------------

    "ENVIRONMENTAL": {
        "strong": [
            "environmental impact",
            "environmental requirements",
            "environmental protection",
            "environmental permit",
            "pollution control",
            "environmental damage",
        ],
        "medium": [
            "environment",
            "environmental",
            "pollution",
            "waste",
            "permit",
            "ecological",
        ],
    },


    # ------------------------------------------------------
    # EMPLOYER
    # ------------------------------------------------------

    "EMPLOYER": {
        "strong": [
            "employer's delay",
            "employer delay",
            "employer's instruction",
            "employer instruction",
            "employer's failure",
            "employer failure",
            "employer's obligation",
            "employer obligation",
            "access provided by the employer",
            "approval by the employer",
        ],
        "medium": [
            "employer",
            "instruction",
            "access",
            "approval",
        ],
    },


    # ------------------------------------------------------
    # CONTRACTOR
    # ------------------------------------------------------

    "CONTRACTOR": {
        "strong": [
            "contractor's failure",
            "contractor failure",
            "contractor's obligation",
            "contractor obligation",
            "contractor's responsibility",
            "contractor responsibility",
            "contractor's performance",
            "contractor performance",
            "contractor resources",
            "contractor workmanship",
        ],
        "medium": [
            "contractor",
            "resources",
            "workmanship",
            "performance",
        ],
    },


    # ------------------------------------------------------
    # EXCEPTIONAL EVENTS
    # ------------------------------------------------------

    "EXCEPTIONAL_EVENTS": {
        "strong": [
            "force majeure",
            "exceptional event",
            "exceptional events",
            "natural disaster",
            "extraordinary event",
            "event beyond reasonable control",
        ],
        "medium": [
            "war",
            "earthquake",
            "flood",
            "emergency",
            "disaster",
        ],
    },


    # ------------------------------------------------------
    # CLAIMS / DISPUTES
    # ------------------------------------------------------

    "CLAIMS_DISPUTES": {
        "strong": [
            "claim and entitlement",
            "claim for payment",
            "claim for extension of time",
            "notice of claim",
            "notice of entitlement",
            "dispute resolution",
            "adjudication",
            "arbitration",
            "settlement of disputes",
        ],
        "medium": [
            "claim",
            "claims",
            "dispute",
            "disputes",
            "settlement",
            "disagreement",
        ],
    },
}


# ==========================================================
# Semantic Category Profiles
# ==========================================================

CATEGORY_SEMANTIC_PROFILES = {

    "TIME_DELAY": (
        "Time and delay risk affecting project schedule, "
        "progress, completion date, extension of time, "
        "late completion, disruption, or delayed works."
    ),

    "COST_FINANCIAL": (
        "Cost and financial risk involving additional costs, "
        "additional payment, compensation, expenditure, "
        "financial loss, price changes, inflation, or payment."
    ),

    "TECHNICAL_DESIGN": (
        "Technical and design risk involving engineering design, "
        "drawings, specifications, technical requirements, "
        "design errors, or technical failures."
    ),

    "SITE_PHYSICAL": (
        "Site and physical condition risk involving ground, soil, "
        "geological, underground, subsurface, hydrological, "
        "or unforeseeable physical conditions."
    ),

    "HEALTH_SAFETY": (
        "Health and safety risk involving workers, personnel, "
        "injury, accidents, unsafe conditions, hazards, "
        "occupational health, or site safety."
    ),

    "QUALITY_DEFECTS": (
        "Quality and defects risk involving defective work, "
        "defects, testing, inspection, non-conformance, "
        "remedial work, failure, or poor quality."
    ),

    "LEGAL_CONTRACTUAL": (
        "Legal and contractual risk involving contractual "
        "obligations, contract requirements, notices, breach, "
        "legal requirements, compliance, or interpretation."
    ),

    "ENVIRONMENTAL": (
        "Environmental risk involving environmental impacts, "
        "pollution, waste, ecological damage, environmental "
        "requirements, permits, or environmental protection."
    ),

    "EMPLOYER": (
        "Employer-related risk involving Employer actions, "
        "instructions, approvals, access, Employer delay, "
        "Employer obligations, or Employer failures."
    ),

    "CONTRACTOR": (
        "Contractor-related risk involving Contractor performance, "
        "resources, workmanship, obligations, responsibilities, "
        "failures, or inadequate performance."
    ),

    "EXCEPTIONAL_EVENTS": (
        "Exceptional event or force majeure risk involving "
        "extraordinary events outside reasonable control, "
        "war, earthquake, flood, natural disaster, or emergency."
    ),

    "CLAIMS_DISPUTES": (
        "Claims and disputes risk involving claims, entitlement, "
        "notices of claim, disagreements, dispute resolution, "
        "adjudication, arbitration, or settlement."
    ),
}


# ==========================================================
# Semantic Category Embeddings
# ==========================================================

CATEGORY_CODES = list(
    CATEGORY_SEMANTIC_PROFILES.keys()
)

CATEGORY_TEXTS = [
    CATEGORY_SEMANTIC_PROFILES[code]
    for code in CATEGORY_CODES
]

CATEGORY_EMBEDDINGS = SEMANTIC_MODEL.encode(
    CATEGORY_TEXTS,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
).astype(np.float32)


# ==========================================================
# Text Matching Helpers
# ==========================================================

def _phrase_count(
    text: str,
    phrase: str,
) -> int:
    """
    Count occurrences of a phrase in the text.
    """

    return text.count(
        phrase.lower()
    )


def _word_count(
    text: str,
    word: str,
) -> int:
    """
    Count whole-word occurrences of a keyword.
    """

    matches = re.findall(
        rf"\b{re.escape(word.lower())}\b",
        text,
    )

    return len(matches)


def _calculate_category_score(
    text: str,
    profile: dict,
) -> int:
    """
    Calculate category relevance.

    Strong contractual phrases receive significantly
    higher weights than generic keywords.
    """

    score = 0

    # ------------------------------------------------------
    # Strong phrases
    # ------------------------------------------------------

    for phrase in profile.get(
        "strong",
        [],
    ):

        count = _phrase_count(
            text,
            phrase,
        )

        if count:
            score += (
                6
                + min(
                    count - 1,
                    2,
                ) * 2
            )

    # ------------------------------------------------------
    # Medium keywords
    # ------------------------------------------------------

    for keyword in profile.get(
        "medium",
        [],
    ):

        count = _word_count(
            text,
            keyword,
        )

        if count:
            score += min(
                count,
                3,
            )

    return score


# ==========================================================
# Hybrid Fast Classifier
# ==========================================================

def _fast_classify(
    risk_description: str,
):
    """
    Hybrid deterministic + semantic risk classification.

    Strategy:
        1. Calculate rule-based category scores.
        2. Calculate semantic similarity against category descriptions.
        3. Treat EMPLOYER / CONTRACTOR primarily as actor dimensions.
        4. Give explicit domain-risk evidence priority over actor mentions.
        5. Combine semantic and rule signals.
        6. Select the strongest category and optionally a secondary category.

    Gemini is not called by this function.
    """

    text = (
        risk_description
        .lower()
        .strip()
    )

    if not text:
        return None

    # ======================================================
    # 1. Rule-Based Scores
    # ======================================================

    rule_scores = {}

    for category_code, profile in CATEGORY_PROFILES.items():
        rule_scores[category_code] = _calculate_category_score(
            text,
            profile,
        )

    # ======================================================
    # 2. Protect Risk-Domain Categories from Actor Dominance
    # ======================================================
    #
    # EMPLOYER and CONTRACTOR are useful classifications when
    # the risk is primarily about that party's own performance
    # or action. However, they should not automatically defeat
    # an explicit domain risk such as cost, defects, safety,
    # delay, site conditions, etc.
    #
    # Example:
    #   "The contractor failed to remedy defective work."
    #
    # The primary risk is QUALITY_DEFECTS; CONTRACTOR is the
    # responsible actor and can remain secondary.
    #
    # Therefore:
    #   - cap raw actor evidence
    #   - preserve strong domain evidence
    #   - add a controlled domain boost when explicit phrases
    #     are present.

    ACTOR_CATEGORIES = {
        "EMPLOYER",
        "CONTRACTOR",
    }

    DOMAIN_CATEGORIES = [
        code
        for code in CATEGORY_PROFILES
        if code not in ACTOR_CATEGORIES
    ]

    # Actor evidence is intentionally capped so repeated words
    # such as "contractor" do not overpower the actual risk type.
    for actor_code in ACTOR_CATEGORIES:
        rule_scores[actor_code] = min(
            rule_scores[actor_code],
            8,
        )

    # Explicit domain phrases receive a modest additional boost.
    # This only applies when the phrase is genuinely present in
    # the risk description.
    DOMAIN_STRONG_BOOST = 2.0

    domain_boosted_scores = dict(rule_scores)

    for category_code in DOMAIN_CATEGORIES:
        strong_phrases = CATEGORY_PROFILES[category_code].get(
            "strong",
            [],
        )

        if any(
            _phrase_count(text, phrase) > 0
            for phrase in strong_phrases
        ):
            domain_boosted_scores[category_code] += DOMAIN_STRONG_BOOST

    # ======================================================
    # 3. Normalize Rule Scores
    # ======================================================

    max_rule_score = (
        max(domain_boosted_scores.values())
        if domain_boosted_scores
        else 0
    )

    normalized_rule_scores = {}

    for category_code, score in domain_boosted_scores.items():
        if max_rule_score > 0:
            normalized_rule_scores[category_code] = (
                score / max_rule_score
            )
        else:
            normalized_rule_scores[category_code] = 0.0

    # ======================================================
    # 4. Semantic Similarity
    # ======================================================

    risk_embedding = SEMANTIC_MODEL.encode(
        [risk_description],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    semantic_scores = np.dot(
        risk_embedding,
        CATEGORY_EMBEDDINGS.T,
    )[0]

    # ======================================================
    # 5. Hybrid Scoring
    # ======================================================

    hybrid_scores = {}

    for index, category_code in enumerate(CATEGORY_CODES):

        semantic_score = float(
            semantic_scores[index]
        )

        rule_score = normalized_rule_scores.get(
            category_code,
            0.0,
        )

        # Base hybrid score.
        hybrid_score = (
            0.70 * semantic_score
            + 0.30 * rule_score
        )

        # --------------------------------------------------
        # Actor-vs-domain correction
        # --------------------------------------------------
        #
        # When a clear domain phrase exists, reduce the chance
        # that EMPLOYER / CONTRACTOR becomes primary solely
        # because the text repeatedly mentions the responsible
        # party.
        #
        # Do NOT suppress the actor completely: it can still
        # become secondary when appropriate.

        if category_code in ACTOR_CATEGORIES:

            explicit_domain_exists = any(
                any(
                    _phrase_count(text, phrase) > 0
                    for phrase in CATEGORY_PROFILES[domain_code].get(
                        "strong",
                        [],
                    )
                )
                for domain_code in DOMAIN_CATEGORIES
            )

            if explicit_domain_exists:
                hybrid_score *= 0.82

        hybrid_scores[category_code] = hybrid_score

    # ======================================================
    # 6. Rank Categories
    # ======================================================

    ranked = sorted(
        hybrid_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        return None

    primary_code, primary_score = ranked[0]

    second_code, second_score = (
        ranked[1]
        if len(ranked) > 1
        else (None, 0.0)
    )

    # ======================================================
    # 7. Secondary Category
    # ======================================================

    secondary_category = None

    score_gap = (
        primary_score
        - second_score
    )

    if (
        second_code is not None
        and second_score >= 0.70 * primary_score
        and score_gap <= 0.12
    ):
        secondary_category = second_code

    # ======================================================
    # 8. Prefer Actor as Secondary when a Domain Risk is Clear
    # ======================================================
    #
    # If a domain risk is primary and EMPLOYER / CONTRACTOR has
    # meaningful evidence, expose the actor as secondary. This
    # preserves useful responsibility information without
    # allowing it to replace the actual risk domain.

    if primary_code not in ACTOR_CATEGORIES:

        actor_candidates = sorted(
            (
                (
                    code,
                    hybrid_scores[code],
                )
                for code in ACTOR_CATEGORIES
                if rule_scores.get(code, 0) > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        if actor_candidates:
            actor_code, actor_score = actor_candidates[0]

            if (
                actor_score >= 0.55 * primary_score
                and actor_code != primary_code
            ):
                secondary_category = actor_code

    # ======================================================
    # 9. Category Names
    # ======================================================

    primary_name = (
        RISK_TAXONOMY[
            primary_code
        ]["name"]
    )

    if secondary_category:
        secondary_name = (
            RISK_TAXONOMY[
                secondary_category
            ]["name"]
        )

        risk_summary = (
            f"The risk primarily concerns "
            f"{primary_name.lower()}, "
            f"with a secondary "
            f"{secondary_name.lower()} aspect."
        )

        reasoning = (
            f"Hybrid classification identified "
            f"{primary_name} as the primary risk domain "
            f"using semantic similarity and contractual "
            f"risk indicators. {secondary_name} was identified "
            f"as a related secondary aspect."
        )

    else:
        risk_summary = (
            f"The risk primarily concerns "
            f"{primary_name.lower()}."
        )

        reasoning = (
            f"Hybrid classification identified "
            f"{primary_name} as the strongest category "
            f"using semantic similarity and contractual "
            f"risk indicators."
        )

    # ======================================================
    # 10. Return Result
    # ======================================================

    return {
        "primary_category": primary_code,
        "secondary_category": secondary_category,
        "risk_summary": risk_summary,
        "reasoning": reasoning,
        "classification_method": (
            "hybrid_semantic_rule_based_actor_aware"
        ),
    }


# ==========================================================
# Gemini Fallback
# ==========================================================

def _gemini_classify(
    risk_description: str,
) -> dict:
    """
    Use Gemini only when the deterministic classifier
    cannot confidently classify the risk.
    """

    category_text = "\n".join(
        f"- {code}: {data['name']} — "
        f"{data['description']}"
        for code, data in RISK_TAXONOMY.items()
    )

    prompt = f"""
You are a construction contract risk-analysis expert.

Classify the following project risk using ONLY the
provided FIDIC risk taxonomy.

RISK:
{risk_description}

AVAILABLE CATEGORIES:
{category_text}

Return ONLY valid JSON in exactly this structure:

{{
    "primary_category": "CATEGORY_CODE",
    "secondary_category": "CATEGORY_CODE or null",
    "risk_summary": "one concise sentence",
    "reasoning": "brief explanation"
}}

Rules:
1. primary_category must be one of the provided category codes.
2. secondary_category must be one of the provided category codes or null.
3. Do not invent categories.
4. Focus on the actual risk described.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    text = response.text.strip()

    # ======================================================
    # Handle Markdown JSON Fences
    # ======================================================

    if text.startswith("```"):

        text = text.replace(
            "```json",
            "",
        )

        text = text.replace(
            "```",
            "",
        )

        text = text.strip()

    # ======================================================
    # Parse JSON
    # ======================================================

    try:

        result = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Gemini returned invalid JSON:\n{text}"
        ) from exc

    # ======================================================
    # Validate Categories
    # ======================================================

    primary = result.get(
        "primary_category"
    )

    secondary = result.get(
        "secondary_category"
    )

    if primary not in RISK_TAXONOMY:

        raise ValueError(
            f"Invalid primary category: {primary}"
        )

    if (
        secondary is not None
        and secondary not in RISK_TAXONOMY
    ):

        raise ValueError(
            f"Invalid secondary category: {secondary}"
        )

    result[
        "classification_method"
    ] = "gemini_fallback"

    return result


# ==========================================================
# Public Classifier
# ==========================================================

def classify_risk(
    risk_description: str,
) -> dict:
    """
    Classify a construction-project risk.

    Strategy:

        1. Fast hybrid semantic + rule-based classification.
        2. Gemini fallback for unexpected failures.
    """

    if not risk_description.strip():

        raise ValueError(
            "Risk description cannot be empty."
        )

    # ======================================================
    # Fast Hybrid Path
    # ======================================================

    fast_result = _fast_classify(
        risk_description
    )

    if fast_result is not None:

        return fast_result

    # ======================================================
    # Gemini Fallback
    # ======================================================

    return _gemini_classify(
        risk_description
    )


# ==========================================================
# Local Test
# ==========================================================

if __name__ == "__main__":

    test_risks = [

        "Unexpected geological conditions are causing additional excavation costs.",

        "The contractor is falling behind the planned completion schedule.",

        "The completed work contains defects that the contractor has failed to remedy.",

        "A flood has damaged the construction site.",

        "The employer has delayed access to the site.",

        "The contractor has failed to provide sufficient resources.",

        "The employer has failed to provide the required approval.",

        "The contractor submitted a claim for additional payment.",

        "An unexpected design error has affected the construction works.",

        "Workers suffered an injury because of unsafe site conditions.",

    ]

    for risk in test_risks:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"RISK: {risk}"
        )

        print(
            "=" * 70
        )

        try:

            result = classify_risk(
                risk
            )

            print(
                json.dumps(
                    result,
                    indent=4,
                )
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )