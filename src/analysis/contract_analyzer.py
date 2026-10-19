"""
FIDIC Contractual Risk Analyzer

Uses retrieved FIDIC clauses + Gemini to generate
a structured contractual analysis.
"""

import json
import os

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_client():
    """
    Create Gemini client using GEMINI_API_KEY.

    The google-genai SDK automatically reads the
    GEMINI_API_KEY environment variable.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(api_key=api_key)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    risk_description: str,
    classification: dict,
    scoring: dict,
    clauses: list,
) -> str:

# --------------------------------------------------------
# Build compact retrieved clause context
# --------------------------------------------------------

    clause_context = []

    MAX_PRIMARY_CHARS = 6000
    MAX_SUPPORTING_CHARS = 2500
    MAX_TOTAL_CHARS = 18000

    total_chars = 0

    for index, clause in enumerate(clauses):

        contract = clause.get("contract", "")
        clause_number = clause.get("clause_number", "")
        clause_title = clause.get("clause_title", "")

        text = clause.get(
            "text",
            clause.get("chunk_text", "")
        )

        # Primary/retrieved clauses get more context.
        # Supporting clauses get a smaller excerpt.
        if index == 0:
            max_chars = MAX_PRIMARY_CHARS
        else:
            max_chars = MAX_SUPPORTING_CHARS

        text = str(text)

        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + "\n[Evidence excerpt truncated for efficiency.]"
            )

        remaining = MAX_TOTAL_CHARS - total_chars

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = (
                text[:remaining]
                + "\n[Evidence context limit reached.]"
            )

        
        clause_context.append(
            f"""
Contract: {contract}
Clause: {clause_number}
Title: {clause_title}

Evidence:
{text}
"""
        )

        total_chars += len(text)


    clause_text = "\n".join(clause_context)
    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This prompt is an f-string.
    # Therefore JSON braces MUST be escaped as {{ and }}.
    # --------------------------------------------------------

    prompt = f"""
You are an expert FIDIC contract risk analyst.

Analyze the construction risk using ONLY the
provided risk information and retrieved FIDIC
contract clauses.

Do not invent contractual provisions.

If the retrieved clauses do not establish a
specific entitlement, clearly state that the
available evidence is insufficient.

==================================================
RISK
==================================================

Risk Description:
{risk_description}

Primary Category:
{classification.get('primary_category', '')}

Secondary Category:
{classification.get('secondary_category', '')}

Risk Summary:
{classification.get('risk_summary', '')}

Classification Reasoning:
{classification.get('reasoning', '')}

==================================================
RISK SCORE
==================================================

Probability:
{scoring.get('probability', '')}

Impact:
{scoring.get('impact', '')}

Risk Score:
{scoring.get('risk_score', '')}

Risk Level:
{scoring.get('risk_level', '')}

==================================================
RETRIEVED FIDIC CLAUSES
==================================================

{clause_text}

==================================================
TASK
==================================================

Provide a structured contractual risk analysis.

Determine:

1. What contractual clause is most directly
   relevant to the root cause?

2. What contractual mechanism applies?

3. Whether the Contractor may have an entitlement
   to additional Cost.

4. Whether the Contractor may have an entitlement
   to Extension of Time.

5. What notice, claim, or procedural requirement
   may be relevant.

6. What obligations or actions are relevant to the
   Contractor?

7. What obligations or actions are relevant to the
   Employer?

8. What practical action should the project team
   take next?

9. What uncertainty or limitation exists in the
   retrieved evidence?

==================================================
IMPORTANT ANALYSIS RULES
==================================================

- Do not invent clause numbers.

- Do not invent contractual rights.

- Do not assume entitlement merely because a clause
  is semantically similar.

- Distinguish clearly between:
  a) explicit contractual basis
  b) reasonable inference
  c) insufficient evidence.

- For every important contractual conclusion,
  identify the specific supplied clause that
  supports it.

- Use only clause text provided in the context
  as evidence.

- Do not quote or fabricate text that is not
  present in the supplied clause text.

- If a conclusion depends on multiple clauses,
  list each relevant clause as evidence.

- Referenced supporting clauses may be used as
  evidence, but they must not be treated as
  primary clauses unless they are directly
  relevant to the root cause.

- If no supplied clause supports a conclusion,
  use "insufficient_evidence".

- Do not use outside FIDIC knowledge to create
  missing clause content.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not wrap the JSON in ```json fences.

Use exactly this structure:

{{
    "primary_clause": {{
        "contract": "",
        "clause_number": "",
        "clause_title": "",
        "reason": ""
    }},

    "contractual_mechanism": "",

    "cost_entitlement": {{
        "status": "",
        "explanation": ""
    }},

    "time_entitlement": {{
        "status": "",
        "explanation": ""
    }},

    "notice_or_claim_requirements": "",

    "contractor_actions": [],

    "employer_actions": [],

    "recommended_next_steps": [],

    "uncertainties": [],

    "evidence": [
        {{
            "contract": "",
            "clause_number": "",
            "clause_title": "",
            "supports": "",
            "evidence": ""
        }}
    ]
}}

==================================================
ALLOWED STATUS VALUES
==================================================

- "supported"
- "potential"
- "not_supported"
- "insufficient_evidence"
"""

    return prompt


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_contract(
    risk_description: str,
    classification: dict,
    scoring: dict,
    clauses: list,
) -> dict:

    # --------------------------------------------------------
    # Validate clauses
    # --------------------------------------------------------

    if not clauses:

        raise ValueError(
            "At least one retrieved clause is required."
        )

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    client = get_client()

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        risk_description=risk_description,
        classification=classification,
        scoring=scoring,
        clauses=clauses,
    )

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if not response or not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    raw_text = response.text.strip()

    # --------------------------------------------------------
    # Remove accidental Markdown JSON fences
    # --------------------------------------------------------

    if raw_text.startswith("```json"):

        raw_text = raw_text[len("```json"):]

    elif raw_text.startswith("```"):

        raw_text = raw_text[len("```"):]

    if raw_text.endswith("```"):

        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(raw_text)

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Gemini returned invalid JSON.\n\n"
            f"Raw response:\n{raw_text}"
        ) from exc

    # --------------------------------------------------------
    # Basic output validation
    # --------------------------------------------------------

    if not isinstance(result, dict):

        raise ValueError(
            "Gemini response JSON must be an object."
        )

    # --------------------------------------------------------
    # Ensure expected top-level fields exist
    # --------------------------------------------------------

    expected_fields = [
        "primary_clause",
        "contractual_mechanism",
        "cost_entitlement",
        "time_entitlement",
        "notice_or_claim_requirements",
        "contractor_actions",
        "employer_actions",
        "recommended_next_steps",
        "uncertainties",
        "evidence",
    ]

    for field in expected_fields:

        if field not in result:

            result[field] = (
                []
                if field in [
                    "contractor_actions",
                    "employer_actions",
                    "recommended_next_steps",
                    "uncertainties",
                    "evidence",
                ]
                else ""
            )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    risk = (
        "Unexpected geological conditions are "
        "causing additional excavation costs "
        "and delaying the work."
    )

    classification = {

        "primary_category":
            "SITE_PHYSICAL",

        "secondary_category":
            "COST_FINANCIAL",

        "risk_summary":
            (
                "Unforeseen subsurface geological "
                "conditions are increasing excavation "
                "costs and causing project delays."
            ),

        "reasoning":
            (
                "The root cause is an unexpected "
                "physical site condition."
            ),
    }

    scoring = {

        "probability": 4,

        "impact": 5,

        "risk_score": 20,

        "risk_level": "CRITICAL",
    }

    clauses = [

        {
            "contract":
                "FIDIC Red Book 2017",

            "clause_number":
                "4.12",

            "clause_title":
                "Unforeseeable Physical Conditions",

            "text":
                (
                    "Retrieved clause text should "
                    "be supplied here."
                ),
        }
    ]

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    result = analyze_contract(

        risk_description=risk,

        classification=classification,

        scoring=scoring,

        clauses=clauses,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\n")

    print("=" * 80)

    print(
        "FIDIC CONTRACTUAL ANALYSIS"
    )

    print("=" * 80)

    print()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )