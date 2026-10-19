import pandas as pd

cache = pd.read_pickle(
    "results/evaluation/signal_weight_candidates_batched.pkl"
)

ids = [14, 54, 62]

columns = [
    "clause_number",
    "semantic_score",
    "concept_score",
    "title_score",
    "clause_signal_score",
    "cross_encoder_score",
]

for item in cache:
    if item["id"] not in ids:
        continue

    print("\n" + "=" * 80)
    print(
        f"CASE {item['id']} | "
        f"EXPECTED {item['expected_clause']} | "
        f"{item['contract']}"
    )
    print("=" * 80)

    results = item["results"].copy()

    print(
        results[columns]
        .head(10)
        .to_string(index=False)
    )