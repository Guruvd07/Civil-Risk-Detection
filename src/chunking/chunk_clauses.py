import pandas as pd
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/all_fidic_clauses_clean.csv"
OUTPUT_PATH = "data/processed/all_fidic_chunks.csv"

# Target chunk size in characters.
# We keep chunks reasonably large so contractual context is preserved.
CHUNK_SIZE = 2500

# Small overlap helps preserve context when a clause is split.
CHUNK_OVERLAP = 300


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize whitespace while preserving the actual clause content.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text):
    """
    Lightweight sentence splitter.

    We intentionally avoid aggressive NLP sentence splitting because
    contractual text contains abbreviations, numbered references,
    brackets and legal cross-references.
    """

    text = clean_text(text)

    if not text:
        return []

    # Split after sentence-ending punctuation when followed by
    # whitespace and an uppercase character/number/opening quote.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])",
        text
    )

    return [part.strip() for part in parts if part.strip()]


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Create semantic-ish chunks while trying to keep sentence boundaries.

    If a sentence is larger than chunk_size, it is kept intact rather
    than being arbitrarily destroyed.
    """

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current = ""

    for sentence in sentences:

        # If current chunk is empty
        if not current:
            current = sentence
            continue

        candidate = current + " " + sentence

        # Keep adding sentences until target size is reached
        if len(candidate) <= chunk_size:
            current = candidate

        else:
            chunks.append(current.strip())

            # ------------------------------------------------
            # Create overlap from the end of previous chunk
            # ------------------------------------------------

            overlap_text = current[-overlap:].strip()

            # Avoid starting in the middle of a word
            if " " in overlap_text:
                overlap_text = overlap_text[
                    overlap_text.find(" ") + 1:
                ]

            current = overlap_text + " " + sentence

            current = current.strip()

    # Add final chunk
    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# PROCESS CLAUSES
# ============================================================

def process_clauses(df):

    all_chunks = []

    for _, row in df.iterrows():

        clause_text = clean_text(row["clause_text"])

        chunks = create_chunks(clause_text)

        for chunk_index, chunk_text in enumerate(chunks, start=1):

            chunk_id = (
                f"{row['clause_id']}_CHUNK_{chunk_index:03d}"
            )

            all_chunks.append({
                "chunk_id": chunk_id,
                "clause_id": row["clause_id"],
                "contract": row["contract"],
                "category": row["category"],
                "page": row["page"],
                "clause_number": row["clause_number"],
                "clause_title": row["clause_title"],
                "chunk_index": chunk_index,
                "chunk_text": chunk_text
            })

    return pd.DataFrame(all_chunks)


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(df):

    print("\n" + "=" * 60)
    print("CHUNK DATASET VALIDATION")
    print("=" * 60)

    print(f"Total chunks: {len(df)}")

    print("\nChunks by contract:")

    print(
        df.groupby("contract")
        .size()
        .to_string()
    )

    print("\nEmpty values:")

    for column in [
        "chunk_id",
        "clause_id",
        "contract",
        "category",
        "clause_number",
        "clause_title",
        "chunk_text"
    ]:

        print(
            f"{column:15s}: "
            f"{df[column].isna().sum()}"
        )

    duplicate_chunks = df["chunk_id"].duplicated().sum()

    print(
        f"\nDuplicate chunk IDs: {duplicate_chunks}"
    )

    short_chunks = (
        df["chunk_text"]
        .str.len()
        .lt(100)
        .sum()
    )

    print(
        f"Very short chunks (<100 chars): {short_chunks}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FIDIC SEMANTIC CHUNKING")
    print("=" * 60)

    print(f"\nInput : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    print(f"\nChunk size : {CHUNK_SIZE} characters")
    print(f"Overlap    : {CHUNK_OVERLAP} characters")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nReading cleaned clause dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Clauses loaded: {len(df)}")

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print("\nCreating chunks...")

    chunks_df = process_clauses(df)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_chunks(chunks_df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\nFirst 10 chunks:\n")

    print(
        chunks_df[
            [
                "chunk_id",
                "contract",
                "clause_number",
                "clause_title",
                "chunk_index"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()