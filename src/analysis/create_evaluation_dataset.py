import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.taxonomy.risk_taxonomy import RISK_TAXONOMY


INPUT_FILE = Path("data/processed/all_fidic_clauses_clean.csv")
OUTPUT_FILE = Path("data/annotations/risk_evaluation.csv")


# ---------------------------------------------------------------------------
# Controlled evaluation scenarios
# ---------------------------------------------------------------------------
# These scenarios are independently phrased. They intentionally do NOT use
# the FIDIC clause title or raw clause text, so the classifier cannot simply
# recognize the answer from the clause wording.
#
# The same scenario is used for each contract version containing the clause.
# expected_clause is retained separately for retrieval evaluation.
SCENARIOS = {
    "1.9": (
        "A required project drawing or instruction is not issued when needed, "
        "creating a risk that the affected work will be delayed or disrupted."
    ),
    "2.1": (
        "The contractor cannot obtain timely access to a required part of the "
        "construction site, preventing planned work from proceeding."
    ),
    "2.5": (
        "Important information about site conditions available to the employer "
        "has not been provided to the contractor, creating uncertainty for the works."
    ),
    "4.1": (
        "The contractor is not fulfilling its overall contractual obligations "
        "or is failing to provide the personnel, equipment, materials, or services "
        "needed to execute the works."
    ),
    "4.2": (
        "The contractor has not provided the required security for proper "
        "performance of its contractual obligations."
    ),
    "4.12": (
        "Unexpected ground or subsurface conditions are encountered during the "
        "works, causing unforeseen difficulties and potentially increasing the "
        "work required, cost, or completion time."
    ),
    "4.18": (
        "Construction activities are creating pollution or other environmental "
        "impacts that require additional controls to protect the environment."
    ),
    "6.7": (
        "Workers on the project are exposed to unsafe conditions or inadequate "
        "health and safety arrangements, creating a risk of injury or illness."
    ),
    "7.1": (
        "Construction work is being carried out without the required standard of "
        "care, proper workmanship, or compliance with the specified method of execution."
    ),
    "7.3": (
        "The quality of materials, plant, or workmanship needs to be examined or "
        "tested to verify that the completed work meets the required standards."
    ),
    "7.4": (
        "The contractor must carry out specified tests on the works but may fail "
        "to provide the necessary testing equipment, staff, or arrangements."
    ),
    "8.4": (
        "A known or foreseeable event may interfere with construction performance "
        "and could increase costs or delay completion if it is not addressed early."
    ),
    "8.5": (
        "Progress of the works is being affected and the planned completion date "
        "may need to be extended because of a qualifying delay."
    ),
    "8.7": (
        "The contractor is experiencing disruption that is affecting progress and "
        "may prevent the works from being completed within the required time."
    ),
    "8.8": (
        "Construction activities have been suspended or interrupted, creating a "
        "risk of further delay to completion."
    ),
    "10.1": (
        "The works are ready to be taken over, but the process of obtaining "
        "completion and possession is affecting the planned project schedule."
    ),
    "11.1": (
        "Defects or other non-conforming work remain in the completed works and "
        "must be corrected before the required quality can be achieved."
    ),
    "11.2": (
        "Work, materials, or workmanship do not comply with the contract and "
        "must be remedied, potentially creating additional corrective work."
    ),
    "13.7": (
        "A change in legislation or regulatory requirements increases the cost "
        "of carrying out the contracted works."
    ),
    "14.2": (
        "The contractor requires the agreed advance payment to support the "
        "financial requirements of carrying out the project."
    ),
    "14.3": (
        "The contractor submits a payment application for amounts it believes "
        "are currently due under the contract."
    ),
    "14.6": (
        "An interim payment certificate is delayed or cannot be issued because "
        "the contractual conditions for payment have not been satisfied."
    ),
    "14.7": (
        "A payment due under the contract is not made within the required period, "
        "creating a financial impact on the project."
    ),
    "16.1": (
        "The contractor is considering suspending the works because the employer "
        "has failed to make a required payment or maintain the necessary financial arrangements."
    ),
    "17.1": (
        "Responsibility for loss, damage, or other contractual consequences "
        "arising during execution of the works needs to be determined between the parties."
    ),
    "17.2": (
        "The contractor has caused loss or damage through its actions or failure "
        "to meet its contractual responsibilities."
    ),
    "17.3": (
        "The employer's use of project documents or other protected material "
        "creates a risk of an intellectual property or third-party rights issue."
    ),
    "18.1": (
        "An extraordinary event outside the parties' reasonable control has "
        "occurred and may prevent normal contractual performance."
    ),
    "18.2": (
        "An extraordinary event has occurred and the affected party needs to "
        "formally notify the other party of the event and its effects."
    ),
    "18.3": (
        "An exceptional event is disrupting the works, and reasonable measures "
        "are needed to minimise the resulting delay."
    ),
    "18.4": (
        "An exceptional event is preventing normal performance and may give rise "
        "to additional cost, delay, or an extension of time."
    ),
    "18.5": (
        "An exceptional event continues to prevent performance of the contract, "
        "creating a potential need to terminate the contract."
    ),
    "18.6": (
        "The parties need to address the contractual consequences of an exceptional "
        "event affecting continued performance of the works."
    ),
    "19.1": (
        "The project requires appropriate insurance arrangements to cover specified "
        "risks and protect the parties against insured losses."
    ),
    "20.1": (
        "A contractor believes that an event or circumstance has created an "
        "entitlement to additional payment, an extension of time, or another "
        "contractual remedy and intends to submit a claim."
    ),
    "20.2": (
        "A contractor needs to comply with the contractual procedure and deadlines "
        "for submitting and supporting a claim for additional entitlement."
    ),
    "21.1": (
        "The parties disagree about a contractual matter and need an independent "
        "dispute-resolution mechanism to determine the disagreement."
    ),
    "21.2": (
        "The parties have failed to appoint the required dispute-board member, "
        "creating a procedural problem in the dispute-resolution process."
    ),
    "21.3": (
        "A contractual disagreement has not been resolved between the parties and "
        "needs to be referred through the agreed dispute-resolution procedure."
    ),
    "21.4": (
        "A dispute remains unresolved and may need to proceed to the next formal "
        "stage of contractual dispute resolution."
    ),
}


# Ground-truth categories aligned with the risk represented by each scenario.
TARGET_CLAUSES = {
    "1.9": "TIME_DELAY",
    "2.1": "EMPLOYER",
    "2.5": "EMPLOYER",
    "4.1": "CONTRACTOR",
    "4.2": "CONTRACTOR",
    "4.12": "SITE_PHYSICAL",
    "4.18": "ENVIRONMENTAL",
    "6.7": "HEALTH_SAFETY",
    "7.1": "QUALITY_DEFECTS",
    "7.3": "QUALITY_DEFECTS",
    "7.4": "QUALITY_DEFECTS",
    "8.4": "TIME_DELAY",
    "8.5": "TIME_DELAY",
    "8.7": "TIME_DELAY",
    "8.8": "TIME_DELAY",
    "10.1": "TIME_DELAY",
    "11.1": "QUALITY_DEFECTS",
    "11.2": "QUALITY_DEFECTS",
    "13.7": "COST_FINANCIAL",
    "14.2": "COST_FINANCIAL",
    "14.3": "COST_FINANCIAL",
    "14.6": "COST_FINANCIAL",
    "14.7": "COST_FINANCIAL",
    "16.1": "COST_FINANCIAL",
    "17.1": "LEGAL_CONTRACTUAL",
    "17.2": "CONTRACTOR",
    "17.3": "LEGAL_CONTRACTUAL",
    "18.1": "EXCEPTIONAL_EVENTS",
    "18.2": "EXCEPTIONAL_EVENTS",
    "18.3": "EXCEPTIONAL_EVENTS",
    "18.4": "EXCEPTIONAL_EVENTS",
    "18.5": "EXCEPTIONAL_EVENTS",
    "18.6": "EXCEPTIONAL_EVENTS",
    "19.1": "LEGAL_CONTRACTUAL",
    "20.1": "CLAIMS_DISPUTES",
    "20.2": "CLAIMS_DISPUTES",
    "21.1": "CLAIMS_DISPUTES",
    "21.2": "CLAIMS_DISPUTES",
    "21.3": "CLAIMS_DISPUTES",
    "21.4": "CLAIMS_DISPUTES",
}


def normalize_clause_number(value):
    """Normalize clause numbers such as 1.90 -> 1.9 and 4.20 -> 4.2."""
    value = str(value).strip()
    try:
        return str(float(value))
    except ValueError:
        return value


def main():
    print("=" * 70)
    print("CREATING FIDIC EVALUATION DATASET")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"Input clauses: {len(df)}")

    required_columns = {"contract", "clause_number"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["clause_key"] = df["clause_number"].apply(normalize_clause_number)

    selected = df[df["clause_key"].isin(TARGET_CLAUSES.keys())].copy()

    selected = (
        selected
        .drop_duplicates(subset=["contract", "clause_key"])
        .reset_index(drop=True)
    )

    selected["expected_category"] = selected["clause_key"].map(TARGET_CLAUSES)
    selected["risk_description"] = selected["clause_key"].map(SCENARIOS)

    missing_scenarios = sorted(
        set(selected["clause_key"]) - set(SCENARIOS.keys())
    )
    if missing_scenarios:
        raise ValueError(
            f"Missing independent scenarios for clauses: {missing_scenarios}"
        )

    valid_categories = set(RISK_TAXONOMY.keys())
    invalid_categories = sorted(
        set(selected["expected_category"]) - valid_categories
    )

    if invalid_categories:
        raise ValueError(
            f"Invalid taxonomy categories: {invalid_categories}"
        )

    selected.insert(0, "id", range(1, len(selected) + 1))

    result = selected[
        [
            "id",
            "risk_description",
            "contract",
            "expected_category",
            "clause_number",
        ]
    ].copy()

    result.rename(
        columns={"clause_number": "expected_clause"},
        inplace=True,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("EVALUATION DATASET CREATED")
    print("=" * 70)
    print(f"Total cases: {len(result)}")
    print(f"Contracts: {result['contract'].nunique()}")
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("Category distribution:")
    print(result["expected_category"].value_counts().to_string())

    print()
    print("Contract distribution:")
    print(result["contract"].value_counts().to_string())

    print()
    print("Expected clause distribution:")
    print(result["expected_clause"].value_counts().sort_index().to_string())

    print()
    print("Sample:")
    print(
        result[
            ["id", "risk_description", "contract",
             "expected_category", "expected_clause"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # Leakage checks.
    leaked_titles = 0
    if "clause_title" in df.columns:
        title_map = (
            df.groupby("clause_key")["clause_title"]
            .first()
            .dropna()
            .astype(str)
            .to_dict()
        )
        for _, row in result.iterrows():
            title = title_map.get(row["expected_clause"], "")
            if title and title.lower() in row["risk_description"].lower():
                leaked_titles += 1

    if leaked_titles:
        raise ValueError(
            f"Potential clause-title leakage detected in {leaked_titles} cases."
        )

    if result["risk_description"].str.len().min() < 30:
        raise ValueError("One or more scenarios are suspiciously short.")

    print()
    print("=" * 70)
    print("VALIDATION PASSED")
    print("Independent scenario text: PASS")
    print("Clause title/raw clause text leakage: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
