import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.retrieve import (
    detect_concepts,
    expand_query,
    retrieve_batch,
    concept_match_score,
    title_relevance,
    clause_signal_score,
    cross_encoder_rerank,
)


# ============================================================
# PATHS
# ============================================================

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "risk_evaluation.csv"
)

CACHE_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "signal_weight_candidates_batched.pkl"
)


# ============================================================
# WEIGHT CONFIGURATIONS
# ============================================================

WEIGHT_CONFIGS = {
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

    "signal_05": {
        "cross": 0.50,
        "semantic": 0.20,
        "concept": 0.15,
        "title": 0.10,
        "signal": 0.05,
    },

    "signal_08": {
        "cross": 0.50,
        "semantic": 0.17,
        "concept": 0.15,
        "title": 0.10,
        "signal": 0.08,
    },
}


# ============================================================
# SCORE CALCULATION
# ============================================================

def calculate_score(row, weights):
    return (
        weights["cross"]
        * float(row["cross_encoder_score"])
        +
        weights["semantic"]
        * float(row["semantic_score"])
        +
        weights["concept"]
        * float(row["concept_score"])
        +
        weights["title"]
        * float(row["title_score"])
        +
        weights["signal"]
        * float(row["clause_signal_score"])
    )


# ============================================================
# BUILD SEMANTIC CANDIDATES
# ============================================================

def build_case_candidates(case):

    query = str(case["risk_description"])

    primary_category = (
        None
        if pd.isna(case.get("primary_category"))
        else str(case["primary_category"])
    )

    secondary_category = (
        None
        if pd.isna(case.get("secondary_category"))
        else str(case["secondary_category"])
    )

    contract = str(case["contract"])

    # --------------------------------------------------------
    # Detect concepts
    # --------------------------------------------------------

    concepts = detect_concepts(
        query=query,
        primary_category=primary_category,
        secondary_category=secondary_category,
    )

    # --------------------------------------------------------
    # Expand query
    # --------------------------------------------------------

    expanded_queries = expand_query(
        query=query,
        primary_category=primary_category,
        secondary_category=secondary_category,
    )

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    batch_results = retrieve_batch(
        expanded_queries,
        top_k=40,
    )

    # --------------------------------------------------------
    # Deduplicate chunks
    # --------------------------------------------------------

    candidate_rows = {}

    for results in batch_results:

        for row in results:

            chunk_id = row["chunk_id"]

            semantic_score = float(
                row["semantic_score"]
            )

            if (
                chunk_id not in candidate_rows
                or semantic_score
                > candidate_rows[chunk_id]["semantic_score"]
            ):
                candidate_rows[chunk_id] = row.to_dict()

    if not candidate_rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(
        list(candidate_rows.values())
    )

    # --------------------------------------------------------
    # Contract filtering
    # --------------------------------------------------------

    candidates = candidates[
        candidates["contract"].astype(str)
        == contract
    ].copy()

    if candidates.empty:
        return candidates

    # --------------------------------------------------------
    # Concept score
    # --------------------------------------------------------

    candidates["concept_score"] = candidates.apply(
        lambda row: concept_match_score(
            clause_title=row["clause_title"],
            clause_text=row["chunk_text"],
            concepts=concepts,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Title score
    # --------------------------------------------------------

    candidates["title_score"] = candidates.apply(
        lambda row: title_relevance(
            clause_title=row["clause_title"],
            concepts=concepts,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Clause signal score
    # --------------------------------------------------------

    candidates["clause_signal_score"] = candidates.apply(
        lambda row: clause_signal_score(
            clause_number=row["clause_number"],
            clause_title=row["clause_title"],
            clause_text=row["chunk_text"],
        ),
        axis=1,
    )

    # Store case information for later batching
    candidates["_case_id"] = case["id"]
    candidates["_expected_clause"] = str(
        case["expected_clause"]
    )
    candidates["_contract"] = contract

    return candidates


# ============================================================
# BATCH CROSS-ENCODER
# ============================================================

def batch_cross_encoder(all_candidates):

    if not all_candidates:
        return []

    print("\nPreparing CrossEncoder batch...")

    combined = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    print(
        f"Total candidate chunks for CrossEncoder: "
        f"{len(combined)}"
    )

    # --------------------------------------------------------
    # Run CrossEncoder ONCE over all candidate pairs
    # --------------------------------------------------------

    queries = []

    for _, row in combined.iterrows():

        case_id = row["_case_id"]

        # Find original query from stored mapping
        queries.append(
            CASE_QUERY_MAP[case_id]
        )

    clause_contexts = (
        combined["clause_title"].astype(str)
        + ". "
        + combined["chunk_text"].astype(str)
    ).str[:5000].tolist()

    pairs = list(
        zip(
            queries,
            clause_contexts,
        )
    )

    print(
        "\nRunning batched CrossEncoder..."
    )
    print(
        "This is the only expensive model pass."
    )

    # Import model directly from retrieve.py
    from src.retrieval.retrieve import RERANKER

    raw_scores = RERANKER.predict(
        pairs,
        batch_size=64,
        show_progress_bar=True,
    )

    raw_scores = np.asarray(
        raw_scores,
        dtype=np.float32,
    )

    raw_scores = np.clip(
        raw_scores,
        -50.0,
        50.0,
    )

    rerank_scores = (
        1.0
        /
        (
            1.0
            +
            np.exp(-raw_scores)
        )
    )

    combined["cross_encoder_score"] = (
        rerank_scores
    )

    return combined


# ============================================================
# CLAUSE LEVEL AGGREGATION
# ============================================================

def aggregate_clauses(candidates):

    if candidates.empty:
        return candidates

    candidates = candidates.copy()

    candidates["clause_key"] = (
        candidates["contract"].astype(str)
        + "||"
        + candidates["clause_number"].astype(str)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # concept_aware_retrieve() ultimately uses the maximum
    # component score across chunks belonging to the clause.
    #
    # We therefore preserve those maxima directly.
    # --------------------------------------------------------

    aggregated = (
        candidates
        .groupby("clause_key", as_index=False)
        .agg(
            {
                "contract": "first",
                "clause_number": "first",
                "clause_title": "first",
                "semantic_score": "max",
                "concept_score": "max",
                "title_score": "max",
                "clause_signal_score": "max",
                "cross_encoder_score": "max",
                "_case_id": "first",
                "_expected_clause": "first",
                "_contract": "first",
            }
        )
    )

    return aggregated


# ============================================================
# EVALUATE CONFIGURATION
# ============================================================

def evaluate_config(cache, weights):

    top1_correct = 0
    top3_correct = 0
    contract_correct = 0

    total = len(cache)

    for item in cache:

        results = item["results"]

        if results is None or results.empty:
            continue

        results = results.copy()

        results["experiment_score"] = (
            results.apply(
                lambda row: calculate_score(
                    row,
                    weights,
                ),
                axis=1,
            )
        )

        results = (
            results
            .sort_values(
                "experiment_score",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        expected_clause = str(
            item["expected_clause"]
        ).strip()

        expected_contract = str(
            item["contract"]
        ).strip()

        top1_clause = str(
            results.iloc[0]["clause_number"]
        ).strip()

        top3_clauses = [
            str(x).strip()
            for x in results.head(3)[
                "clause_number"
            ].tolist()
        ]

        top1_contract = str(
            results.iloc[0]["contract"]
        ).strip()

        if top1_clause == expected_clause:
            top1_correct += 1

        if expected_clause in top3_clauses:
            top3_correct += 1

        if top1_contract == expected_contract:
            contract_correct += 1

    return {
        "Top-1 Accuracy": top1_correct / total,
        "Top-3 Accuracy": top3_correct / total,
        "Contract Accuracy": contract_correct / total,
    }


# ============================================================
# BUILD CACHE
# ============================================================

def build_cache(df):

    print("\nBuilding semantic candidate pool...")
    print(
        "CrossEncoder will NOT run once per case."
    )

    global CASE_QUERY_MAP

    CASE_QUERY_MAP = {
        case["id"]: str(case["risk_description"])
        for _, case in df.iterrows()
    }

    all_candidates = []

    total = len(df)

    for i, (_, case) in enumerate(
        df.iterrows(),
        start=1,
    ):

        print(
            f"[{i:03d}/{total}] "
            f"Case {case['id']} "
            f"| {case['contract']} "
            f"| expected {case['expected_clause']}"
        )

        candidates = build_case_candidates(
            case
        )

        if not candidates.empty:
            all_candidates.append(
                candidates
            )

    print(
        "\nSemantic candidate generation complete."
    )

    # --------------------------------------------------------
    # ONE CrossEncoder pass
    # --------------------------------------------------------

    combined = batch_cross_encoder(
        all_candidates
    )

    # --------------------------------------------------------
    # Split back by evaluation case
    # --------------------------------------------------------

    cache = []

    for _, case in df.iterrows():

        case_id = case["id"]

        case_candidates = combined[
            combined["_case_id"] == case_id
        ].copy()

        case_candidates = aggregate_clauses(
            case_candidates
        )

        # Keep only fields needed by experiment
        keep_columns = [
            "contract",
            "clause_number",
            "clause_title",
            "semantic_score",
            "concept_score",
            "title_score",
            "clause_signal_score",
            "cross_encoder_score",
        ]

        case_candidates = case_candidates[
            keep_columns
        ].copy()

        cache.append(
            {
                "id": case_id,
                "contract": str(
                    case["contract"]
                ),
                "expected_clause": str(
                    case["expected_clause"]
                ),
                "results": case_candidates,
            }
        )

    CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.to_pickle(
        cache,
        CACHE_FILE,
    )

    print(
        "\nCache saved:"
    )
    print(CACHE_FILE)

    return cache


# ============================================================
# LOAD OR BUILD CACHE
# ============================================================

def get_cache(df):

    if CACHE_FILE.exists():

        print(
            "\nFound existing batched cache:"
        )
        print(CACHE_FILE)

        print(
            "\nLoading cache..."
        )

        return pd.read_pickle(
            CACHE_FILE
        )

    return build_cache(df)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CLAUSE SIGNAL WEIGHT A/B EXPERIMENT"
    )
    print("=" * 80)

    df = pd.read_csv(
        EVALUATION_FILE
    )

    print(
        f"\nEvaluation cases: {len(df)}"
    )

    # --------------------------------------------------------
    # Build/load cache
    # --------------------------------------------------------

    cache = get_cache(df)

    # --------------------------------------------------------
    # Test configurations
    # --------------------------------------------------------

    experiment_results = []

    for name, weights in WEIGHT_CONFIGS.items():

        print(
            "\n"
            + "-"
            * 80
        )

        print(
            f"Testing: {name}"
        )

        print(
            f"Weights: {weights}"
        )

        print(
            "-"
            * 80
        )

        metrics = evaluate_config(
            cache,
            weights,
        )

        print(
            f"Top-1 Accuracy    : "
            f"{metrics['Top-1 Accuracy']:.4f}"
        )

        print(
            f"Top-3 Accuracy    : "
            f"{metrics['Top-3 Accuracy']:.4f}"
        )

        print(
            f"Contract Accuracy : "
            f"{metrics['Contract Accuracy']:.4f}"
        )

        experiment_results.append(
            {
                "configuration": name,
                **metrics,
            }
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        experiment_results
    )

    print("\n")
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        result_df.to_string(
            index=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    CASE_QUERY_MAP = {}
    main()