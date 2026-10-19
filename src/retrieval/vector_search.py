import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_PATH = "data/processed/embeddings/fidic_embeddings.npy"
METADATA_PATH = "data/processed/embeddings/fidic_embedding_metadata.csv"

INDEX_DIR = Path("data/processed/vector_store")
INDEX_PATH = INDEX_DIR / "fidic_faiss.index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5


# ============================================================
# LOAD RESOURCES
# ============================================================

def load_resources():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    print(f"Embeddings shape: {embeddings.shape}")

    print("\nLoading metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Metadata rows: {len(metadata)}")

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding count does not match metadata count."
        )

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    return embeddings, metadata, model


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(embeddings):

    dimension = embeddings.shape[1]

    print("\nBuilding FAISS index...")
    print(f"Vector dimension: {dimension}")

    # Inner Product works as cosine similarity because
    # embeddings were normalized during generation.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"Vectors indexed: {index.ntotal}")

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(index):

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    print(f"\nFAISS index saved to:")
    print(INDEX_PATH)


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    model,
    index,
    metadata,
    top_k=TOP_K
):

    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Similarity search
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        row = metadata.iloc[idx]

        results.append({
            "rank": rank,
            "score": float(score),
            "contract": row["contract"],
            "clause_number": row["clause_number"],
            "clause_title": row["clause_title"],
            "chunk_id": row["chunk_id"],
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],
        })

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(query, results):

    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH")
    print("=" * 70)

    print(f"\nQuery:")
    print(query)

    for result in results:

        print("\n" + "-" * 70)

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"Similarity : {result['score']:.4f}"
        )

        print(
            f"Contract   : {result['contract']}"
        )

        print(
            f"Clause     : {result['clause_number']} "
            f"- {result['clause_title']}"
        )

        print(
            f"Chunk      : {result['chunk_id']}"
        )

        print("\nText:")

        print(
            result["chunk_text"][:1000]
        )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FIDIC VECTOR SEARCH")
    print("=" * 70)

    embeddings, metadata, model = load_resources()

    index = build_index(embeddings)

    save_index(index)

    # --------------------------------------------------------
    # Test queries
    # --------------------------------------------------------

    test_queries = [
        "What happens if the contractor is delayed?",
        "How can the contractor claim an extension of time?",
        "What are the contractor's obligations regarding variations?",
    ]

    for query in test_queries:

        results = search(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=TOP_K
        )

        display_results(
            query,
            results
        )

    print("\nVECTOR SEARCH COMPLETE")


if __name__ == "__main__":
    main()