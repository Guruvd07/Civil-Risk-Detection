"""
End-to-End FIDIC Risk Analyzer

Pipeline:

Risk Description
      ↓
Gemini Risk Classification
      ↓
Probability + Impact + Risk Score
      ↓
Concept-Aware FAISS Retrieval
      ↓
Actual FIDIC Clause Text
      ↓
Gemini Contractual Analysis
      ↓
Risk Decision Engine
      ↓
Final Risk + Contractual Explanation
"""
import time
from src.models.risk_classifier import classify_risk
from src.analysis.risk_scoring import calculate_risk_score
from src.retrieval.retrieve import multi_query_retrieve
from src.analysis.contract_analyzer import analyze_contract
from src.analysis.decision_engine import build_risk_decision


def analyze_risk(
    risk_description: str,
    probability: int,
    impact: int,
    contract: str = "FIDIC Red Book 2017",
    top_k: int = 5,
) -> dict:

    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------

    if not risk_description.strip():
        raise ValueError("Risk description cannot be empty.")

    # ------------------------------------------------------------------
    # 1.5 Validate contract
    # ------------------------------------------------------------------

    allowed_contracts = {
        "FIDIC Red Book 2017",
        "FIDIC Silver Book 2017",
        "FIDIC Silver Book 1999",
    }

    if contract not in allowed_contracts:
        raise ValueError(
            f"Unsupported contract: {contract}. "
            f"Choose from: {sorted(allowed_contracts)}"
        )

    # ------------------------------------------------------------------
    # 2. Classify risk using Gemini
    # ------------------------------------------------------------------

    start = time.time()

    classification = classify_risk(risk_description)

    print(f"[TIME] Risk Classification: {time.time() - start:.2f}s")

    primary_category = classification["primary_category"]
    secondary_category = classification.get("secondary_category")

    # ------------------------------------------------------------------
    # 3. Calculate risk score
    # ------------------------------------------------------------------

    scoring = calculate_risk_score(
        probability=probability,
        impact=impact,
    )

    # ------------------------------------------------------------------
    # 4. Retrieve relevant FIDIC clauses
    # ------------------------------------------------------------------
    start = time.time()
    
    retrieved = multi_query_retrieve(
        risk_description=risk_description,
        primary_category=primary_category,
        secondary_category=secondary_category,
        contract=contract,
        top_k=top_k,
    )
    
    print(f"[TIME] FAISS Retrieval: {time.time() - start:.2f}s")

    # ------------------------------------------------------------------
    # 5. Convert retrieved clauses + referenced clauses
    #    into JSON-compatible records
    # ------------------------------------------------------------------

    clauses = []
    supporting_clauses = []

    for _, row in retrieved.iterrows():

        clause = {
            "rank": int(row["rank"]),
            "similarity_score": round(
                float(row["similarity_score"]),
                4,
            ),
            "contract": str(row["contract"]),
            "clause_number": str(row["clause_number"]),
            "clause_title": str(row["clause_title"]),
            "chunk_id": str(row["chunk_id"]),
            "text": str(row["chunk_text"]),
            "clause_type": "primary_retrieved",
        }

        # Include source chunks if available
        if "source_chunk_ids" in row:
            clause["source_chunk_ids"] = str(
                row["source_chunk_ids"]
            )

        if "chunk_count" in row:
            clause["chunk_count"] = int(
                row["chunk_count"]
            )

        clauses.append(clause)

        # --------------------------------------------------------------
        # Add referenced clauses as supporting context
        # --------------------------------------------------------------

        if "referenced_clauses" in row:

            referenced_clauses = row["referenced_clauses"]

            if isinstance(referenced_clauses, list):

                for ref in referenced_clauses:

                    supporting_clause = {
                        "rank": None,
                        "similarity_score": None,
                        "contract": str(ref["contract"]),
                        "clause_number": str(
                            ref["clause_number"]
                        ),
                        "clause_title": str(
                            ref["clause_title"]
                        ),
                        "chunk_id": None,
                        "text": str(ref["chunk_text"]),
                        "clause_type": "referenced_supporting",
                        "reference_source": str(
                            ref["reference_source"]
                        ),
                    }

                    if "source_chunk_ids" in ref:
                        supporting_clause[
                            "source_chunk_ids"
                        ] = str(
                            ref["source_chunk_ids"]
                        )

                    if "chunk_count" in ref:
                        supporting_clause[
                            "chunk_count"
                        ] = int(
                            ref["chunk_count"]
                        )

                    supporting_clauses.append(
                        supporting_clause
                    )

    # ------------------------------------------------------------------
    # 6. Contractual analysis using ACTUAL retrieved clause text
    # ------------------------------------------------------------------
    start = time.time()
    
    contractual_analysis = analyze_contract(
        risk_description=risk_description,
        classification=classification,
        scoring=scoring,
        clauses=clauses + supporting_clauses,
    )
    
    print(
    "[DEBUG] Retrieved primary candidates:",
    [
        {
            "contract": c.get("contract"),
            "clause_number": c.get("clause_number"),
            "clause_title": c.get("clause_title"),
        }
        for c in clauses[:5]
    ],
    )

    print(
        "[DEBUG] Gemini primary clause:",
        contractual_analysis.get("primary_clause"),
    )
    
    print(f"[TIME] Gemini Contract Analysis: {time.time() - start:.2f}s")
    


    # ------------------------------------------------------------------
    # 7. Build structured risk decision
    # ------------------------------------------------------------------

    decision = build_risk_decision(
        contractual_analysis,
        scoring,
    )

    # ------------------------------------------------------------------
    # 8. Return complete analysis
    # ------------------------------------------------------------------
    print("[TIME] Risk analysis pipeline completed")
    
    return {
        "risk_description": risk_description,

        "classification": classification,

        "scoring": scoring,

        "relevant_clauses": clauses,

        "supporting_clauses": supporting_clauses,

        "contractual_analysis": contractual_analysis,

        "decision": decision,
    }


# ==========================================================================
# TEST
# ==========================================================================

if __name__ == "__main__":

    risk = (
        "Unexpected geological conditions are causing "
        "additional excavation costs and delaying the work."
    )

    result = analyze_risk(
        risk_description=risk,
        probability=4,
        impact=5,
        contract="FIDIC Red Book 2017",
        top_k=5,
    )

    print("\n")
    print("=" * 80)
    print("FIDIC END-TO-END RISK ANALYSIS")
    print("=" * 80)

    print("\nRISK:")
    print(result["risk_description"])

    # ------------------------------------------------------------------
    # RISK CLASSIFICATION
    # ------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("RISK CLASSIFICATION")
    print("-" * 80)

    print(
        f"Primary Category   : "
        f"{result['classification']['primary_category']}"
    )

    print(
        f"Secondary Category : "
        f"{result['classification'].get('secondary_category')}"
    )

    print(
        f"Summary            : "
        f"{result['classification']['risk_summary']}"
    )

    # ------------------------------------------------------------------
    # RISK SCORING
    # ------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("RISK SCORING")
    print("-" * 80)

    print(
        f"Probability : "
        f"{result['scoring']['probability']}"
    )

    print(
        f"Impact      : "
        f"{result['scoring']['impact']}"
    )

    print(
        f"Risk Score  : "
        f"{result['scoring']['risk_score']}"
    )

    print(
        f"Risk Level  : "
        f"{result['scoring']['risk_level']}"
    )

    # ------------------------------------------------------------------
    # RETRIEVED CLAUSES
    # ------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("RETRIEVED FIDIC CLAUSES")
    print("-" * 80)

    for clause in result["relevant_clauses"]:

        print(
            f"\nRank {clause['rank']} | "
            f"Score: {clause['similarity_score']}"
        )

        print(
            f"Contract : {clause['contract']}"
        )

        print(
            f"Clause   : {clause['clause_number']}"
        )

        print(
            f"Title    : {clause['clause_title']}"
        )

        if "chunk_count" in clause:
            print(
                f"Chunks   : {clause['chunk_count']}"
            )

    # ------------------------------------------------------------------
    # CONTRACTUAL ANALYSIS
    # ------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("CONTRACTUAL ANALYSIS")
    print("-" * 80)

    print(
        result["contractual_analysis"]
    )

    # ------------------------------------------------------------------
    # FINAL DECISION
    # ------------------------------------------------------------------

    print("\n")
    print("-" * 80)
    print("FINAL DECISION")
    print("-" * 80)

    decision = result["decision"]

    print(
        f"Risk Level              : "
        f"{decision['risk_level']}"
    )

    print(
        f"Primary Clause          : "
        f"{decision['primary_clause']['clause_number']} - "
        f"{decision['primary_clause']['clause_title']}"
    )

    print(
        f"Contract                : "
        f"{decision['primary_clause']['contract']}"
    )

    print(
        f"Cost Entitlement        : "
        f"{decision['cost']['status']}"
    )

    print(
        f"Time Entitlement        : "
        f"{decision['time']['status']}"
    )

    print(
        f"Claim Required          : "
        f"{decision['claim']['required']}"
    )

    print(
        f"Notice Required         : "
        f"{decision['claim']['notice_required']}"
    )

    print(
        f"Notice Deadline         : "
        f"{decision['claim']['notice_deadline_days']} days"
    )

    print(
        f"Detailed Claim Deadline : "
        f"{decision['claim']['detailed_claim_deadline_days']} days"
    )

    print(
        f"Confidence              : "
        f"{decision['confidence']['level']}"
    )

    print(
        f"Evidence Count          : "
        f"{decision['confidence']['evidence_count']}"
    )

    print(
        f"Confidence Reason       : "
        f"{decision['confidence']['reason']}"
    )

    print(
        f"Evidence Items          : "
        f"{len(decision['evidence'])}"
    )

    print("\n")
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)