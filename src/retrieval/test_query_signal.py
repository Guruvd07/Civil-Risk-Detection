import re
import pandas as pd


CACHE_FILE = "results/evaluation/signal_weight_candidates_batched.pkl"


# ============================================================
# QUERY-AWARE INTENT SIGNALS
# ============================================================

QUERY_SIGNAL_RULES = {
    # --------------------------------------------------------
    # 2017 TIME / PROGRESS
    # --------------------------------------------------------

    "8.5": [
        "extension of time",
        "extend the time",
        "time for completion",
        "completion date",
        "entitled to an extension",
    ],

    "8.7": [
        "rate of progress",
        "progress is too slow",
        "progress too slow",
        "behind programme",
        "behind schedule",
        "progress is behind",
        "progress is falling behind",
        "insufficient progress",
        "slow progress",
        "disruption affecting progress",
        "affecting progress",
    ],

    "8.8": [
        "delay damages",
        "damages for delay",
        "delay damage",
    ],

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    "14.2": [
        "advance payment",
        "advance payments",
        "payment in advance",
    ],

    "14.3": [
        "application for interim payment",
        "application for payment",
        "payment application",
        "interim payment application",
    ],

    "14.6": [
        "interim payment certificate",
        "payment certificate",
        "certificate for payment",
        "certificate cannot be issued",
        "certificate cannot be issued",
        "certificate is delayed",
        "payment certificate delayed",
    ],

    "14.7": [
        "payment is not made",
        "payment not made",
        "payment is overdue",
        "payment overdue",
        "amounts due",
        "amount due",
        "failure to pay",
        "failed to pay",
    ],

    # --------------------------------------------------------
    # CLAIMS
    # --------------------------------------------------------

    "20.1": [
        "contractor claim",
        "contractor's claim",
        "claim procedure",
        "notice of claim",
        "claim notice",
    ],

    "20.2": [
        "claim submission",
        "submission of claim",
        "detailed claim",
        "fully detailed claim",
        "claim for payment",
        "claim for extension of time",
    ],

    # --------------------------------------------------------
    # DISPUTE / DAAB
    # --------------------------------------------------------

    "21.1": [
        "constitution of daab",
        "appointment of daab",
        "daab constitution",
    ],

    "21.2": [
        "failure to appoint daab",
        "daab not appointed",
        "unable to appoint daab",
    ],

    "21.3": [
        "failure to agree",
        "disagreement",
        "daab decision",
        "dispute remains unresolved",
    ],

    "21.4": [
        "refer the dispute",
        "referral to daab",
        "dispute to daab",
        "obtaining a daab decision",
    ],
}


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def query_signal_score(query, clause_number):
    """
    Query-aware signal.

    Returns:
        0.0 = no query-specific signal
        1.0 = strong query-specific signal
    """

    clause = str(clause_number).strip()

    rules = QUERY_SIGNAL_RULES.get(clause, [])

    if not rules:
        return 0.0

    query = normalize(query)

    matches = 0

    for phrase in rules:
        if normalize(phrase) in query:
            matches += 1

    if matches == 0:
        return 0.0

    # Cap at 1.0
    return min(
        matches / 2.0,
        1.0,
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(row, query, signal_weight=0.02):

    query_signal = query_signal_score(
        query,
        row["clause_number"],
    )

    # Keep the same total weight structure.
    return (
        0.50 * float(row["cross_encoder_score"])
        + 0.23 * float(row["semantic_score"])
        + 0.15 * float(row["concept_score"])
        + 0.10 * float(row["title_score"])
        + signal_weight * query_signal
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("QUERY-AWARE SIGNAL EXPERIMENT")
    print("=" * 80)

    cache = pd.read_pickle(CACHE_FILE)

    evaluation = pd.read_csv(
        "data/annotations/risk_evaluation.csv"
    )

    query_map = dict(
        zip(
            evaluation["id"],
            evaluation["risk_description"],
        )
    )

    improved = []
    worsened = []
    baseline_correct = 0
    query_signal_correct = 0

    for item in cache:

        results = item["results"]

        if results is None or results.empty:
            continue

        query = str(
            query_map[item["id"]]
        )

        # Cache does not contain query, so recover it from
        # evaluation dataset below.
        if not query:
            continue

        expected = str(
            item["expected_clause"]
        ).strip()

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline = results.copy()

        baseline["score"] = (
            0.50 * baseline["cross_encoder_score"]
            + 0.25 * baseline["semantic_score"]
            + 0.15 * baseline["concept_score"]
            + 0.10 * baseline["title_score"]
        )

        baseline = baseline.sort_values(
            "score",
            ascending=False,
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Query-aware signal
        # ----------------------------------------------------

        enhanced = results.copy()

        enhanced["query_signal"] = enhanced.apply(
            lambda row: query_signal_score(
                query,
                row["clause_number"],
            ),
            axis=1,
        )

        enhanced["score"] = enhanced.apply(
            lambda row: calculate_score(
                row,
                query,
                signal_weight=0.02,
            ),
            axis=1,
        )

        enhanced = enhanced.sort_values(
            "score",
            ascending=False,
        ).reset_index(drop=True)

        baseline_top1 = str(
            baseline.iloc[0]["clause_number"]
        ).strip()

        enhanced_top1 = str(
            enhanced.iloc[0]["clause_number"]
        ).strip()

        if baseline_top1 == expected:
            baseline_correct += 1

        if enhanced_top1 == expected:
            query_signal_correct += 1

        if (
            enhanced_top1 == expected
            and baseline_top1 != expected
        ):
            improved.append(
                {
                    "id": item["id"],
                    "expected": expected,
                    "baseline": baseline_top1,
                    "query_signal": enhanced_top1,
                }
            )

        elif (
            baseline_top1 == expected
            and enhanced_top1 != expected
        ):
            worsened.append(
                {
                    "id": item["id"],
                    "expected": expected,
                    "baseline": baseline_top1,
                    "query_signal": enhanced_top1,
                }
            )

    # ========================================================
    # RESULTS
    # ========================================================

    total = len(cache)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Baseline Top-1      : "
        f"{baseline_correct / total:.4f}"
    )

    print(
        f"Query-Signal Top-1  : "
        f"{query_signal_correct / total:.4f}"
    )

    print(
        f"Improved cases      : {len(improved)}"
    )

    print(
        f"Worsened cases      : {len(worsened)}"
    )

    print("\n" + "=" * 80)
    print("IMPROVED")
    print("=" * 80)

    if improved:
        print(
            pd.DataFrame(improved).to_string(
                index=False
            )
        )
    else:
        print("None")

    print("\n" + "=" * 80)
    print("WORSENED")
    print("=" * 80)

    if worsened:
        print(
            pd.DataFrame(worsened).to_string(
                index=False
            )
        )
    else:
        print("None")


if __name__ == "__main__":
    main()