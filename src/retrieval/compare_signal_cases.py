import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


CACHE_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "signal_weight_candidates_batched.pkl"
)


WEIGHTS = {
    "baseline": {
        "cross": 0.50,
        "semantic": 0.25,
        "concept": 0.15,
        "title": 0.10,
        "signal": 0.00,
    },
    "signal_02": {
        "cross": 0.50,
        "semantic": 0.23,
        "concept": 0.15,
        "title": 0.10,
        "signal": 0.02,
    },
}


def score(row, weights):
    return (
        weights["cross"] * float(row["cross_encoder_score"])
        + weights["semantic"] * float(row["semantic_score"])
        + weights["concept"] * float(row["concept_score"])
        + weights["title"] * float(row["title_score"])
        + weights["signal"] * float(row["clause_signal_score"])
    )


def rank_case(results, weights):
    results = results.copy()

    results["score"] = results.apply(
        lambda row: score(row, weights),
        axis=1,
    )

    results = results.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    return results


def main():

    print("=" * 80)
    print("BASELINE vs SIGNAL_02 — CASE LEVEL COMPARISON")
    print("=" * 80)

    cache = pd.read_pickle(CACHE_FILE)

    improved = []
    worsened = []
    unchanged = []

    for item in cache:

        results = item["results"]

        if results is None or results.empty:
            continue

        expected = str(item["expected_clause"]).strip()

        baseline = rank_case(
            results,
            WEIGHTS["baseline"],
        )

        signal = rank_case(
            results,
            WEIGHTS["signal_02"],
        )

        baseline_top1 = str(
            baseline.iloc[0]["clause_number"]
        ).strip()

        signal_top1 = str(
            signal.iloc[0]["clause_number"]
        ).strip()

        baseline_top3 = [
            str(x).strip()
            for x in baseline.head(3)["clause_number"]
        ]

        signal_top3 = [
            str(x).strip()
            for x in signal.head(3)["clause_number"]
        ]

        baseline_correct = baseline_top1 == expected
        signal_correct = signal_top1 == expected

        baseline_top3_correct = expected in baseline_top3
        signal_top3_correct = expected in signal_top3

        row = {
            "id": item["id"],
            "contract": item["contract"],
            "expected": expected,
            "baseline_top1": baseline_top1,
            "signal_top1": signal_top1,
            "baseline_top3": " | ".join(baseline_top3),
            "signal_top3": " | ".join(signal_top3),
        }

        if signal_correct and not baseline_correct:
            improved.append(row)

        elif baseline_correct and not signal_correct:
            worsened.append(row)

        else:
            unchanged.append(row)

    # --------------------------------------------------------
    # PRINT IMPROVED
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("IMPROVED CASES")
    print("=" * 80)

    print(f"Count: {len(improved)}")

    if improved:
        print(
            pd.DataFrame(improved).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # PRINT WORSE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("WORSENED CASES")
    print("=" * 80)

    print(f"Count: {len(worsened)}")

    if worsened:
        print(
            pd.DataFrame(worsened).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Improved Top-1 : {len(improved)}")
    print(f"Worsened Top-1 : {len(worsened)}")
    print(f"Other cases    : {len(unchanged)}")

    print("\nDone.")


if __name__ == "__main__":
    main()