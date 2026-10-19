from pathlib import Path

import faiss
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    BASE_DIR / "data" / "processed" / "embeddings" / "fidic_embeddings.npy"
)

METADATA_PATH = (
    BASE_DIR / "data" / "processed" / "embeddings" / "fidic_embedding_metadata.csv"
)

INDEX_DIR = BASE_DIR / "models"
INDEX_PATH = INDEX_DIR / "fidic_faiss.index"
INDEX_METADATA_PATH = INDEX_DIR / "fidic_faiss_metadata.csv"


# --------------------------------------------------
# Load embeddings and metadata
# --------------------------------------------------

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
metadata = pd.read_csv(METADATA_PATH)

print(f"Embeddings shape : {embeddings.shape}")
print(f"Metadata rows    : {len(metadata)}")


# --------------------------------------------------
# Validation
# --------------------------------------------------

if len(embeddings) != len(metadata):
    raise ValueError(
        f"Row mismatch: {len(embeddings)} embeddings "
        f"vs {len(metadata)} metadata rows"
    )

if embeddings.ndim != 2:
    raise ValueError("Embeddings must be a 2D matrix")

dimension = embeddings.shape[1]

if dimension != 384:
    raise ValueError(
        f"Expected 384 dimensions, found {dimension}"
    )


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------
# Embeddings are normalized.
# Therefore Inner Product = Cosine Similarity.

print("\nBuilding FAISS index...")

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


# --------------------------------------------------
# Save index + metadata
# --------------------------------------------------

INDEX_DIR.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(INDEX_PATH))
metadata.to_csv(INDEX_METADATA_PATH, index=False)


# --------------------------------------------------
# Final validation
# --------------------------------------------------

print("\n" + "=" * 50)
print("FAISS INDEX CREATED")
print("=" * 50)

print(f"Vectors indexed : {index.ntotal}")
print(f"Dimension       : {dimension}")
print(f"Index type      : {type(index).__name__}")
print(f"Index path      : {INDEX_PATH}")
print(f"Metadata path   : {INDEX_METADATA_PATH}")

if index.ntotal != len(metadata):
    raise ValueError("FAISS index and metadata are misaligned")

print("\nValidation: PASS")