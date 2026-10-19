import pandas as pd
from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")

INPUT_FILES = [
    "fidic_red_book_1_clauses.csv",
    "fidicsilver2017-5_clauses.csv",
    "fidic-silver-book-2_clauses.csv",
]

OUTPUT_FILE = (
    PROCESSED_DIR /
    "all_fidic_clauses_clean.csv"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove soft hyphen
    text = text.replace("\u00ad", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# ============================================================
# TITLE CLEANING
# ============================================================

def clean_title(title):

    if pd.isna(title):
        return ""

    title = str(title)

    title = clean_text(title)

    return title


# ============================================================
# LOAD ONE DATASET
# ============================================================

def load_dataset(filename):

    path = PROCESSED_DIR / filename

    print(f"\nReading: {filename}")

    if not path.exists():

        print("ERROR: File not found")

        return None

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "clause_number": str
        }
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    return df


# ============================================================
# STANDARDIZE DATASET
# ============================================================

def standardize_dataset(df):

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "clause_id",
        "contract",
        "category",
        "page",
        "clause_number",
        "clause_title",
        "clause_text"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean text fields
    # --------------------------------------------------------

    df["contract"] = (
        df["contract"]
        .apply(clean_text)
    )

    df["category"] = (
        df["category"]
        .apply(clean_text)
    )

    df["clause_number"] = (
        df["clause_number"]
        .astype(str)
        .str.strip()
    )

    df["clause_title"] = (
        df["clause_title"]
        .apply(clean_title)
    )

    df["clause_text"] = (
        df["clause_text"]
        .apply(clean_text)
    )

    # --------------------------------------------------------
    # Convert page to integer
    # --------------------------------------------------------

    df["page"] = pd.to_numeric(
        df["page"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove completely empty clause text
    # --------------------------------------------------------

    before = len(df)

    df = df[
        df["clause_text"].str.len() > 0
    ].copy()

    removed = before - len(df)

    if removed:

        print(
            f"Empty clauses removed: {removed}"
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    print(
        f"Total clauses: {len(df)}"
    )

    # --------------------------------------------------------
    # Contract distribution
    # --------------------------------------------------------

    print("\nClauses by contract:")

    print(
        df["contract"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Empty values
    # --------------------------------------------------------

    print("\nEmpty values:")

    for column in [
        "contract",
        "category",
        "clause_number",
        "clause_title",
        "clause_text"
    ]:

        count = (
            df[column]
            .isna()
            .sum()
        )

        print(
            f"{column:15} : {count}"
        )

    # --------------------------------------------------------
    # Duplicate clauses
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "contract",
            "clause_number"
        ],
        keep=False
    )

    duplicate_rows = df[duplicates]

    duplicate_count = (
        duplicate_rows
        .groupby(
            ["contract", "clause_number"]
        )
        .ngroups
    )

    print(
        f"\nDuplicate contract + clause groups: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        print("\nDuplicate examples:")

        print(
            duplicate_rows[
                [
                    "contract",
                    "clause_number",
                    "clause_title"
                ]
            ]
            .drop_duplicates(
                subset=[
                    "contract",
                    "clause_number"
                ]
            )
            .head(20)
            .to_string(index=False)
        )

        print(
            f"\nDuplicate contract + clause: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Short text
    # --------------------------------------------------------

    short_text = (
        df["clause_text"]
        .str.len()
        .lt(30)
        .sum()
    )

    print(
        f"Very short clause text (<30 chars): "
        f"{short_text}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("FIDIC CLAUSE PREPROCESSING")
    print("=" * 60)

    datasets = []

    # --------------------------------------------------------
    # Load all datasets
    # --------------------------------------------------------

    for filename in INPUT_FILES:

        df = load_dataset(filename)

        if df is not None:

            df = standardize_dataset(df)

            datasets.append(df)

    if not datasets:

        print(
            "\nERROR: No datasets found."
        )

        return

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    master_df = pd.concat(
        datasets,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove duplicate rows only
    #
    # Do NOT remove same clause numbers across
    # different FIDIC contracts.
    # --------------------------------------------------------

    before = len(master_df)

    master_df = (
        master_df
        .drop_duplicates(
            subset=[
                "contract",
                "clause_number",
                "clause_text"
            ]
        )
        .reset_index(drop=True)
    )

    removed = before - len(master_df)

    print(
        f"\nDuplicate rows removed: {removed}"
    )

    # --------------------------------------------------------
    # Re-create global IDs
    # --------------------------------------------------------

    master_df["clause_id"] = [
        f"FIDIC_{i:05d}"
        for i in range(
            1,
            len(master_df) + 1
        )
    ]

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    master_df = master_df[
        [
            "clause_id",
            "contract",
            "category",
            "page",
            "clause_number",
            "clause_title",
            "clause_text"
        ]
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_dataset(
        master_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    master_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nClean dataset saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"\nFinal rows: {len(master_df)}"
    )

    print(
        "\nFirst 10 rows:"
    )

    print(
        master_df[
            [
                "clause_id",
                "contract",
                "clause_number",
                "clause_title"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()