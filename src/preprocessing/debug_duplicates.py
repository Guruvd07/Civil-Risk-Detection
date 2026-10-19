import pandas as pd


CSV_PATH = "data/processed/all_fidic_clauses_clean.csv"


def main():

    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig",
        dtype={"clause_number": str}
    )

    duplicates = df[
        df.duplicated(
            subset=["contract", "clause_number"],
            keep=False
        )
    ].copy()

    print("\n" + "=" * 70)
    print("FIDIC DUPLICATE CLAUSE DIAGNOSIS")
    print("=" * 70)

    print(f"\nTotal rows: {len(df)}")
    print(f"Duplicate rows: {len(duplicates)}")

    groups = (
        duplicates
        .groupby(["contract", "clause_number"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"Duplicate groups: {len(groups)}")

    print("\nTop 30 duplicate groups:\n")

    print(
        groups
        .head(30)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("DETAILED EXAMPLES")
    print("=" * 70)

    for _, row in groups.head(10).iterrows():

        contract = row["contract"]
        clause = row["clause_number"]

        print(
            f"\n--- {contract} | Clause {clause} ---"
        )

        matches = df[
            (df["contract"] == contract)
            & (df["clause_number"] == clause)
        ]

        print(
            matches[
                [
                    "clause_id",
                    "page",
                    "clause_number",
                    "clause_title"
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()