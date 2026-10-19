import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/all_fidic_chunks.csv"

OUTPUT_DIR = Path("data/processed/embeddings")

EMBEDDINGS_PATH = OUTPUT_DIR / "fidic_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "fidic_embedding_metadata.csv"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks():

    print("Reading chunk dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Chunks loaded: {len(df)}")

    required_columns = [
        "chunk_id",
        "clause_id",
        "contract",
        "category",
        "clause_number",
        "clause_title",
        "chunk_index",
        "chunk_text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(model, texts):

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype("float32")


# ============================================================
# VALIDATION
# ============================================================

def validate_embeddings(df, embeddings):

    print("\n" + "=" * 60)
    print("EMBEDDING VALIDATION")
    print("=" * 60)

    print(f"Chunks                : {len(df)}")
    print(f"Embedding rows        : {embeddings.shape[0]}")
    print(f"Embedding dimensions  : {embeddings.shape[1]}")
    print(f"Data type             : {embeddings.dtype}")

    # Row count must match
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            "Number of embeddings does not match number of chunks."
        )

    # Check NaN / infinity
    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # Check vector norms
    norms = np.linalg.norm(embeddings, axis=1)

    print(
        f"Vector norm range     : "
        f"{norms.min():.4f} -> {norms.max():.4f}"
    )

    # Normalized embeddings should have norm ~1
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Embeddings are not properly normalized."
        )

    print("NaN / Infinity       : 0")
    print("Normalization        : PASS")
    print("Row alignment        : PASS")

    print("=" * 60)


# ============================================================
# SAVE
# ============================================================

def save_embeddings(df, embeddings):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving embeddings...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    # Metadata stays aligned with embedding row index.
    metadata_columns = [
        "chunk_id",
        "clause_id",
        "contract",
        "category",
        "page",
        "clause_number",
        "clause_title",
        "chunk_index",
        "chunk_text",
    ]

    available_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    metadata = df[available_columns].copy()

    # Explicit vector row mapping
    metadata.insert(
        0,
        "embedding_index",
        range(len(metadata))
    )

    metadata.to_csv(
        METADATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Embeddings saved to : {EMBEDDINGS_PATH}")
    print(f"Metadata saved to   : {METADATA_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FIDIC EMBEDDING GENERATION")
    print("=" * 60)

    print(f"\nInput : {INPUT_PATH}")
    print(f"Model : {MODEL_NAME}")

    # Load chunks
    df = load_chunks()

    # Remove accidental empty text
    df["chunk_text"] = (
        df["chunk_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_text = (df["chunk_text"] == "").sum()

    if empty_text > 0:
        raise ValueError(
            f"Found {empty_text} chunks with empty text."
        )

    # Load model
    model = load_model()

    # Generate embeddings
    embeddings = create_embeddings(
        model,
        df["chunk_text"].tolist()
    )

    # Validate
    validate_embeddings(
        df,
        embeddings
    )

    # Save
    save_embeddings(
        df,
        embeddings
    )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()