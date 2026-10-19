import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.risk_classifier import _fast_classify
from src.retrieval.retrieve import concept_aware_retrieve


EVALUATION_FILE = Path(
    "data/annotations/risk_evaluation.csv"
)

RESULTS_DIR = Path(
    "results/evaluation"
)

PREDICTIONS_FILE = (
    RESULTS_DIR / "evaluation_predictions.csv"
)

METRICS_FILE = (
    RESULTS_DIR / "evaluation_metrics.json"
)

CONFUSION_FILE = (
    RESULTS_DIR / "confusion_matrix.csv"
)


def normalize_clause_number(value):
    """
    Normalize clause numbers.

    Examples:
        1.90 -> 1.9
        2.10 -> 2.1
        4.20 -> 4.2
        4.12 -> 4.12
    """

    if value is None:
        return ""

    value = str(value).strip()

    try:
        return str(float(value))
    except ValueError:
        return value


def extract_retrieval_rows(results):
    """
    Convert concept_aware_retrieve() output
    into a list of dictionaries.

    concept_aware_retrieve() returns a pandas DataFrame.
    """

    if results is None:
        return []

    if isinstance(results, pd.DataFrame):
        return results.to_dict(
            orient="records"
        )

    if isinstance(results, pd.Series):
        return [
            results.to_dict()
        ]

    if isinstance(results, dict):
        return [results]

    if isinstance(results, list):
        return [
            item
            for item in results
            if isinstance(item, dict)
        ]

    return []


def evaluate_classification(df):
    """
    Evaluate the FAST keyword-based classifier only.

    We deliberately do NOT call Gemini here.

    This makes evaluation:
    - deterministic
    - fast
    - independent of API quota
    - reproducible
    """

    print("\n" + "=" * 70)
    print("RISK CLASSIFICATION EVALUATION")
    print("=" * 70)

    y_true = []
    y_pred = []

    total = len(df)

    for index, row in df.iterrows():

        risk_description = str(
            row["risk_description"]
        ).strip()

        expected = str(
            row["expected_category"]
        ).strip()

        print(
            f"[{index + 1}/{total}] "
            f"Classifying risk..."
        )

        try:

            result = _fast_classify(
                risk_description
            )

            if result is None:
                predicted = "UNCLASSIFIED"
            else:
                predicted = result.get(
                    "primary_category",
                    "UNCLASSIFIED"
                )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            predicted = "ERROR"

        y_true.append(expected)
        y_pred.append(predicted)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    confusion_df.index.name = "Actual"
    confusion_df.columns.name = "Predicted"

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
    }

    return (
        y_pred,
        metrics,
        confusion_df,
    )


def evaluate_retrieval(df):
    """
    Evaluate FIDIC clause retrieval.

    Metrics:
    - Top-1 clause accuracy
    - Top-3 clause accuracy
    - Contract filtering accuracy
    """

    print("\n" + "=" * 70)
    print("FIDIC CLAUSE RETRIEVAL EVALUATION")
    print("=" * 70)

    top1_correct = 0
    top3_correct = 0
    contract_correct = 0

    retrieval_rows = []

    total = len(df)

    for index, row in df.iterrows():

        risk_description = str(
            row["risk_description"]
        ).strip()

        expected_clause = normalize_clause_number(
            row["expected_clause"]
        )

        expected_contract = str(
            row["contract"]
        ).strip()

        print(
            f"[{index + 1}/{total}] "
            f"Retrieving clauses..."
        )

        try:

            raw_results = concept_aware_retrieve(
                query=risk_description,
                contract=expected_contract,
                top_k=3,
            )

            results = extract_retrieval_rows(
                raw_results
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            results = []

        retrieved_clauses = []

        retrieved_contracts = []

        for result in results[:3]:

            clause = normalize_clause_number(
                result.get(
                    "clause_number",
                    ""
                )
            )

            contract = str(
                result.get(
                    "contract",
                    ""
                )
            ).strip()

            if clause:
                retrieved_clauses.append(
                    clause
                )

            if contract:
                retrieved_contracts.append(
                    contract
                )

        # -----------------------------------------------------------
        # Top-1
        # -----------------------------------------------------------

        if retrieved_clauses:

            top1_clause = retrieved_clauses[0]

        else:

            top1_clause = ""

        top1_match = (
            top1_clause == expected_clause
        )

        # -----------------------------------------------------------
        # Top-3
        # -----------------------------------------------------------

        top3_match = (
            expected_clause
            in retrieved_clauses
        )

        # -----------------------------------------------------------
        # Contract filtering
        # -----------------------------------------------------------

        contract_match = all(
            contract == expected_contract
            for contract in retrieved_contracts
        )

        # If retrieval returned nothing,
        # it should not count as correct.
        if not retrieved_contracts:
            contract_match = False

        if top1_match:
            top1_correct += 1

        if top3_match:
            top3_correct += 1

        if contract_match:
            contract_correct += 1

        retrieval_rows.append(
            {
                "id": row["id"],
                "contract": expected_contract,
                "expected_clause": expected_clause,
                "top1_clause": top1_clause,
                "top3_clauses": " | ".join(
                    retrieved_clauses
                ),
                "top1_match": top1_match,
                "top3_match": top3_match,
                "contract_match": contract_match,
            }
        )

    top1_accuracy = (
        top1_correct / total
        if total
        else 0
    )

    top3_accuracy = (
        top3_correct / total
        if total
        else 0
    )

    contract_accuracy = (
        contract_correct / total
        if total
        else 0
    )

    metrics = {
        "top1_clause_accuracy": float(
            top1_accuracy
        ),
        "top3_clause_accuracy": float(
            top3_accuracy
        ),
        "contract_filtering_accuracy": float(
            contract_accuracy
        ),
        "total_cases": total,
    }

    return (
        retrieval_rows,
        metrics,
    )


def main():

    print("=" * 70)
    print("FIDIC RISK ANALYSIS SYSTEM EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load evaluation dataset
    # ---------------------------------------------------------------

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{EVALUATION_FILE}"
        )

    df = pd.read_csv(
        EVALUATION_FILE
    )

    required_columns = {
        "id",
        "risk_description",
        "contract",
        "expected_category",
        "expected_clause",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:

        raise ValueError(
            "Evaluation dataset is empty."
        )

    print()
    print(
        f"Evaluation cases: {len(df)}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # 1. Classification evaluation
    # ---------------------------------------------------------------

    (
        predictions,
        classification_metrics,
        confusion_df,
    ) = evaluate_classification(df)

    # ---------------------------------------------------------------
    # 2. Retrieval evaluation
    # ---------------------------------------------------------------

    (
        retrieval_rows,
        retrieval_metrics,
    ) = evaluate_retrieval(df)

    # ---------------------------------------------------------------
    # 3. Combine results
    # ---------------------------------------------------------------

    prediction_df = df[
        [
            "id",
            "risk_description",
            "contract",
            "expected_category",
            "expected_clause",
        ]
    ].copy()

    prediction_df[
        "predicted_category"
    ] = predictions

    retrieval_df = pd.DataFrame(
        retrieval_rows
    )

    if not retrieval_df.empty:

        prediction_df = prediction_df.merge(
            retrieval_df[
                [
                    "id",
                    "top1_clause",
                    "top3_clauses",
                    "top1_match",
                    "top3_match",
                    "contract_match",
                ]
            ],
            on="id",
            how="left",
        )

    # ---------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Save confusion matrix
    # ---------------------------------------------------------------

    confusion_df.to_csv(
        CONFUSION_FILE,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------

    metrics = {
        "dataset": {
            "total_cases": len(df),
            "contracts": int(
                df["contract"].nunique()
            ),
            "categories": int(
                df["expected_category"].nunique()
            ),
        },
        "classification": classification_metrics,
        "retrieval": retrieval_metrics,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\nRisk Classification:")

    print(
        f"Accuracy          : "
        f"{classification_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{classification_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{classification_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{classification_metrics['macro_f1']:.4f}"
    )

    print("\nClause Retrieval:")

    print(
        f"Top-1 Accuracy    : "
        f"{retrieval_metrics['top1_clause_accuracy']:.4f}"
    )

    print(
        f"Top-3 Accuracy    : "
        f"{retrieval_metrics['top3_clause_accuracy']:.4f}"
    )

    print(
        f"Contract Accuracy : "
        f"{retrieval_metrics['contract_filtering_accuracy']:.4f}"
    )

    print("\nOutput files:")

    print(
        f"Predictions       : "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Metrics            : "
        f"{METRICS_FILE}"
    )

    print(
        f"Confusion Matrix   : "
        f"{CONFUSION_FILE}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()