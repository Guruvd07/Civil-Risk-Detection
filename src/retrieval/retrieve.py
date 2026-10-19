"""
FIDIC Concept-Expansion Retrieval

Pipeline:

    Risk Description
          ↓
    Original semantic retrieval
          +
    FIDIC contractual concept expansion
          ↓
    FAISS candidate generation
          ↓
    Concept-aware reranking
          +
    Clause-specific contractual signals
          ↓
    Clause-level deduplication
          ↓
    Full clause context assembly
          ↓
    Explicit clause-reference retrieval
          ↓
    Relevant FIDIC clauses
"""

from pathlib import Path
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "embeddings"
    / "fidic_embeddings.npy"
)

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "embeddings"
    / "fidic_embedding_metadata.csv"
)

FAISS_INDEX_PATH = (
    BASE_DIR
    / "models"
    / "fidic_faiss.index"
)


# ============================================================
# LOAD MODEL / INDEX / METADATA
# ============================================================

print("Loading embedding model...")

MODEL = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Loading cross-encoder reranker...")

RERANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


print("Loading FAISS index...")

INDEX = faiss.read_index(
    str(FAISS_INDEX_PATH)
)

METADATA = pd.read_csv(
    METADATA_PATH
)


# ============================================================
# VALIDATION
# ============================================================

if INDEX.ntotal != len(METADATA):
    raise ValueError(
        "FAISS index and metadata row count do not match."
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for concept matching.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_clause_number(value):
    """
    Normalize clause numbers so values such as:

        8.50
        8.5

    are treated consistently.
    """

    value = str(value).strip()

    try:
        return str(float(value))
    except ValueError:
        return value


# ============================================================
# FIDIC CONTRACTUAL CONCEPTS
# ============================================================

FIDIC_CONCEPTS = {

    "physical_conditions": {
        "canonical": "Unforeseeable Physical Conditions",

        "phrases": [
            "unforeseeable physical conditions",
            "unforeseen physical conditions",
            "physical conditions",
            "unforeseeable conditions",
            "ground conditions",
            "subsurface conditions",
            "geological conditions",
            "site conditions",
        ],
    },

    "cost": {
        "canonical": "Additional Cost",

        "phrases": [
            "additional cost",
            "additional costs",
            "extra cost",
            "extra costs",
            "cost incurred",
            "costs incurred",
            "additional expenditure",
            "financial consequences",
        ],
    },

    "time": {
        "canonical": "Extension of Time for Completion",

        "phrases": [
            "extension of time",
            "extension of time for completion",
            "delay to completion",
            "delay in completion",
            "delayed completion",
        ],
    },

    "claims": {
        "canonical": "Claim and Entitlement",

        "phrases": [
            "claim",
            "claims",
            "notice",
            "entitlement",
            "contractor claim",
        ],
    },

    "responsibility": {
        "canonical": "Contractor Responsibility",

        "phrases": [
            "contractor responsibility",
            "contractor obligations",
            "contractor risk",
            "employer responsibility",
            "employer risk",
        ],
    },

    "archaeological": {
        "canonical": "Archaeological and Geological Findings",

        "phrases": [
            "archaeological",
            "archaeological findings",
            "fossils",
            "antiquity",
            "geological findings",
        ],
    },
}


# ============================================================
# CATEGORY → CONTRACTUAL CONCEPTS
# ============================================================

CATEGORY_CONCEPTS = {

    "SITE_PHYSICAL": [
        "physical_conditions",
    ],

    "COST_FINANCIAL": [
        "cost",
    ],

    "TIME_DELAY": [
        "time",
    ],

    "CLAIMS_DISPUTES": [
        "claims",
    ],

    "CONTRACTOR": [
        "responsibility",
    ],

    "EMPLOYER": [
        "responsibility",
    ],
}


# ============================================================
# CLAUSE-SPECIFIC RETRIEVAL SIGNALS
# ============================================================

CLAUSE_SIGNALS = {
    
    "4.12": [
        "unforeseeable physical conditions",
        "unforeseen physical conditions",
        "unexpected geological conditions",
        "underground rock formations",
        "subsurface conditions",
        "physical obstructions at the site",
    ],

    "8.5": [
        "extension of time",
        "extension of time for completion",
        "time for completion",
        "delay to completion",
    ],

    "8.7": [
        "delay damages",
        "delay damages for failure to complete",
        "damages for delay",
    ],

    "8.8": [
        "suspension of work",
        "suspension",
        "suspend work",
    ],

    "10.1": [
        "taking over",
        "taking-over",
        "taking over the works",
        "taking over of the works",
    ],

    "14.2": [
        "advance payment",
        "advance payment guarantee",
    ],

    "14.3": [
        "application for interim payment",
        "interim payment",
        "payment statement",
    ],

    "14.6": [
        "issue of interim payment certificate",
        "interim payment certificate",
        "payment certificate",
    ],

    "14.7": [
        "payment",
        "time for payment",
        "amounts due",
    ],

    "16.1": [
        "suspension by contractor",
        "contractor suspend",
        "failure to pay",
        "payment not made",
    ],

    "18.1": [
        "exceptional event",
        "exceptional events",
        "beyond reasonable control",
    ],

    "18.2": [
        "notice of an exceptional event",
        "notice of exceptional event",
    ],

    "18.3": [
        "duty to minimise delay",
        "minimise delay",
        "minimise the delay",
    ],

    "18.4": [
        "consequences of an exceptional event",
        "consequences of exceptional event",
    ],

    "18.5": [
        "optional termination",
        "termination due to exceptional event",
    ],

    "18.6": [
        "release from performance",
        "release from performance under the contract",
    ],

    "20.1": [
        "claims",
        "claim",
        "contractor's claims",
        "contractor claims",
    ],

    "20.2": [
        "claims for payment",
        "claims for payment and/or extension of time",
        "claim submission",
    ],

    "21.1": [
        "constitution of daab",
        "daab",
        "dispute avoidance adjudication board",
    ],

    "21.2": [
        "failure to appoint daab",
        "appoint daab member",
        "daab member",
    ],

    "21.3": [
        "failure to agree",
        "agreement of daab",
        "daab decision",
    ],

    "21.4": [
        "obtaining daab decision",
        "daab decision",
        "dispute to daab",
    ],
}


# ============================================================
# DETECT CONCEPTS
# ============================================================

def detect_concepts(
    query: str,
    primary_category: str | None = None,
    secondary_category: str | None = None,
):
    """
    Detect contractual concepts from the risk description
    and taxonomy categories.
    """

    normalized_query = normalize_text(
        query
    )

    detected = []

    # --------------------------------------------------------
    # Text-based detection
    # --------------------------------------------------------

    for concept_name, concept_data in FIDIC_CONCEPTS.items():

        for phrase in concept_data["phrases"]:

            normalized_phrase = normalize_text(
                phrase
            )

            if normalized_phrase in normalized_query:

                detected.append(
                    concept_name
                )

                break

    # --------------------------------------------------------
    # Category-based detection
    # --------------------------------------------------------

    for category in [
        primary_category,
        secondary_category,
    ]:

        if not category:
            continue

        for concept_name in CATEGORY_CONCEPTS.get(
            category,
            [],
        ):

            if concept_name not in detected:

                detected.append(
                    concept_name
                )

    return list(
        dict.fromkeys(
            detected
        )
    )


# ============================================================
# CONCEPT EXPANSION
# ============================================================

def expand_query(
    query: str,
    primary_category: str | None = None,
    secondary_category: str | None = None,
):
    """
    Expand the original risk into FIDIC contractual
    terminology.
    """

    concepts = detect_concepts(
        query=query,
        primary_category=primary_category,
        secondary_category=secondary_category,
    )

    expanded_queries = [
        query
    ]

    for concept_name in concepts:

        concept_data = FIDIC_CONCEPTS[
            concept_name
        ]

        # ----------------------------------------------------
        # Canonical FIDIC terminology
        # ----------------------------------------------------

        expanded_queries.append(
            concept_data["canonical"]
        )

        # ----------------------------------------------------
        # Individual contractual phrases
        # ----------------------------------------------------

        for phrase in concept_data["phrases"]:

            expanded_queries.append(
                phrase
            )

    return list(
        dict.fromkeys(
            expanded_queries
        )
    )


# ============================================================
# BASIC SEMANTIC RETRIEVAL
# ============================================================

def retrieve_batch(
    queries,
    top_k: int = 40,
):
    """
    Batch semantic retrieval for multiple queries.

    Encodes all queries in one SentenceTransformer call
    and searches FAISS in one batch.
    """

    if not queries:
        return []

    clean_queries = [
        str(query).strip()
        for query in queries
        if str(query).strip()
    ]

    if not clean_queries:
        return []

    query_embeddings = MODEL.encode(
        clean_queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(
        np.float32
    )

    scores, indices = INDEX.search(
        query_embeddings,
        top_k,
    )

    all_results = []

    for query_index, query in enumerate(
        clean_queries
    ):

        results = []

        for rank, (
            index,
            score,
        ) in enumerate(
            zip(
                indices[query_index],
                scores[query_index],
            ),
            start=1,
        ):

            if index < 0:
                continue

            row = METADATA.iloc[
                int(index)
            ].copy()

            row["semantic_score"] = float(
                score
            )

            row["rank"] = rank

            results.append(
                row
            )

        all_results.append(
            results
        )

    return all_results


def retrieve(
    query: str,
    top_k: int = 5,
):
    """
    Basic semantic FAISS retrieval.
    """

    if not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    query_embedding = MODEL.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    scores, indices = INDEX.search(
        query_embedding,
        top_k,
    )

    results = []

    for rank, (
        index,
        score,
    ) in enumerate(
        zip(
            indices[0],
            scores[0],
        ),
        start=1,
    ):

        if index < 0:
            continue

        row = METADATA.iloc[
            int(index)
        ].copy()

        row["semantic_score"] = float(
            score
        )

        row["rank"] = rank

        results.append(
            row
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# CONCEPT MATCH SCORE
# ============================================================

def concept_match_score(
    clause_title: str,
    clause_text: str,
    concepts,
):
    """
    Score how strongly a clause matches the detected
    FIDIC concepts.

    Title matches are stronger than body matches.
    """

    title = normalize_text(
        clause_title
    )

    text = normalize_text(
        clause_text
    )

    if not concepts:
        return 0.0

    best_scores = []

    for concept_name in concepts:

        concept_data = FIDIC_CONCEPTS[
            concept_name
        ]

        best = 0.0

        canonical = normalize_text(
            concept_data["canonical"]
        )

        # ----------------------------------------------------
        # Canonical phrase in title
        # ----------------------------------------------------

        if canonical in title:

            best = max(
                best,
                1.00,
            )

        # ----------------------------------------------------
        # Canonical phrase in body
        # ----------------------------------------------------

        elif canonical in text:

            best = max(
                best,
                0.75,
            )

        # ----------------------------------------------------
        # Other contractual phrases
        # ----------------------------------------------------

        for phrase in concept_data["phrases"]:

            normalized_phrase = normalize_text(
                phrase
            )

            if normalized_phrase in title:

                best = max(
                    best,
                    0.90,
                )

            elif normalized_phrase in text:

                best = max(
                    best,
                    0.60,
                )

        best_scores.append(
            best
        )

    return max(
        best_scores
    )


# ============================================================
# TITLE RELEVANCE
# ============================================================

def title_relevance(
    clause_title: str,
    concepts,
):
    """
    Additional title-level contractual relevance.
    """

    title = normalize_text(
        clause_title
    )

    if not concepts:
        return 0.0

    score = 0.0

    for concept_name in concepts:

        concept_data = FIDIC_CONCEPTS[
            concept_name
        ]

        canonical = normalize_text(
            concept_data["canonical"]
        )

        if canonical in title:

            score = max(
                score,
                1.0,
            )

        for phrase in concept_data["phrases"]:

            normalized_phrase = normalize_text(
                phrase
            )

            if normalized_phrase in title:

                score = max(
                    score,
                    0.85,
                )

    return score


# ============================================================
# FULL CLAUSE CONTEXT ASSEMBLY
# ============================================================


def assemble_full_clause_text(
    contract: str,
    clause_number: str,
):
    """
    Assemble a complete clause while removing duplicated text
    caused by overlapping chunks.

    Returns:
        chunk_text: Reconstructed clause text
        source_chunk_ids: IDs of all source chunks
        chunk_count: Number of source chunks
    """

    matching = METADATA[
        (METADATA["contract"].astype(str) == str(contract))
        & (
            METADATA["clause_number"].astype(str)
            == str(clause_number)
        )
    ].copy()

    if matching.empty:
        return {
            "chunk_text": "",
            "source_chunk_ids": [],
            "chunk_count": 0,
        }

    # Preserve the original chunk order.
    if "chunk_id" in matching.columns:
        matching = matching.sort_values("chunk_id")

    texts = (
        matching["chunk_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    chunk_ids = (
        matching["chunk_id"].astype(str).tolist()
        if "chunk_id" in matching.columns
        else []
    )

    # Merge consecutive chunks by removing their shared overlap.
    combined_text = ""

    for text in texts:
        text = text.strip()

        if not text:
            continue

        if not combined_text:
            combined_text = text
            continue

        # Find the largest exact suffix-prefix overlap.
        max_overlap = min(
            len(combined_text),
            len(text),
            1500,
        )

        overlap = 0

        for size in range(max_overlap, 19, -1):
            if combined_text[-size:] == text[:size]:
                overlap = size
                break

        if overlap:
            combined_text += text[overlap:]
        else:
            # No exact overlap found; preserve the new content.
            combined_text += "\n\n" + text

    return {
        "chunk_text": combined_text.strip(),
        "source_chunk_ids": chunk_ids,
        "chunk_count": len(chunk_ids),
    }

    # --------------------------------------------------------
    # Preserve original chunk/document order
    # --------------------------------------------------------

    if "chunk_id" in matching.columns:

        matching = matching.sort_values(
            "chunk_id"
        )

    # --------------------------------------------------------
    # Combine every chunk
    # --------------------------------------------------------

    texts = (
        matching["chunk_text"]
        .astype(str)
        .tolist()
    )

    chunk_ids = (
        matching["chunk_id"]
        .astype(str)
        .tolist()
    )

    combined_text = "\n\n".join(
        texts
    )

    return {
        "chunk_text": combined_text,
        "source_chunk_ids": chunk_ids,
        "chunk_count": len(chunk_ids),
    }


# ============================================================
# EXTRACT CLAUSE REFERENCES
# ============================================================

def extract_clause_references(
    text: str,
) -> list[str]:
    """
    Extract FIDIC-style clause references.

    Examples:

        20.2
        13.3.1
        4.12.4
        8.5

    Nested references are normalized:

        13.3.1 -> 13.3
        4.12.4 -> 4.12
    """

    if not text:
        return []

    pattern = (
        r"(?<!\d)"
        r"(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)"
        r"(?!\d)"
    )

    matches = re.findall(
        pattern,
        str(text),
    )

    references = []

    for ref in matches:

        parts = ref.split(".")

        if len(parts) < 2:
            continue

        main_clause = (
            f"{parts[0]}.{parts[1]}"
        )

        if main_clause not in references:

            references.append(
                main_clause
            )

    return references


# ============================================================
# RETRIEVE REFERENCED CLAUSES
# ============================================================

def retrieve_referenced_clauses(
    contract: str,
    clause_number: str,
    clause_text: str,
) -> list[dict]:
    """
    Retrieve clauses explicitly referenced by a selected clause.

    Only clauses belonging to the same contract are returned.
    """

    references = extract_clause_references(
        clause_text
    )

    # --------------------------------------------------------
    # Remove the current clause itself
    # --------------------------------------------------------

    references = [
        ref
        for ref in references
        if str(ref) != str(clause_number)
    ]

    referenced_results = []

    # --------------------------------------------------------
    # Retrieve every referenced clause
    # --------------------------------------------------------

    for ref in references:

        matching = METADATA[
            (
                METADATA["contract"].astype(str)
                == str(contract)
            )
            &
            (
                METADATA["clause_number"].astype(str)
                == str(ref)
            )
        ].copy()

        if matching.empty:
            continue

        # ----------------------------------------------------
        # Assemble COMPLETE referenced clause
        # ----------------------------------------------------

        full_clause = assemble_full_clause_text(
            contract=contract,
            clause_number=ref,
        )

        if not full_clause["chunk_text"]:
            continue

        # ----------------------------------------------------
        # Use first metadata row
        # ----------------------------------------------------

        row = matching.iloc[0].copy()

        row["chunk_text"] = (
            full_clause["chunk_text"]
        )

        row["source_chunk_ids"] = (
            " | ".join(
                full_clause[
                    "source_chunk_ids"
                ]
            )
        )

        row["chunk_count"] = (
            full_clause["chunk_count"]
        )

        row["reference_source"] = (
            clause_number
        )

        referenced_results.append(
            row.to_dict()
        )

    return referenced_results


# ============================================================
# CLAUSE-SPECIFIC SIGNAL SCORE
# ============================================================

def clause_signal_score(
    clause_number,
    clause_title,
    clause_text,
):
    """
    Score distinctive contractual terminology for a specific
    clause.

    Returns a normalized score between 0 and 1.
    """

    clause_number = normalize_clause_number(
        clause_number
    )

    signals = CLAUSE_SIGNALS.get(
        clause_number,
        [],
    )

    if not signals:
        return 0.0

    text = normalize_text(
        f"{clause_title} {clause_text}"
    )

    matched = 0

    for signal in signals:

        if normalize_text(signal) in text:

            matched += 1

    return min(
        matched / max(
            len(signals),
            1,
        ),
        1.0,
    )


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def cross_encoder_rerank(
    query: str,
    candidates: pd.DataFrame,
    batch_size: int = 16,
):
    """
    Rerank semantic candidates using the original risk description
    and each candidate clause/chunk as a query-document pair.
    """

    if candidates.empty:
        return candidates

    pairs = []

    for _, row in candidates.iterrows():

        clause_title = str(
            row.get("clause_title", "")
        )

        clause_text = str(
            row.get("chunk_text", "")
        )

        clause_context = (
            f"{clause_title}. {clause_text}"
        )[:5000]

        pairs.append(
            (
                str(query),
                clause_context,
            )
        )

    raw_scores = RERANKER.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
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

    candidates = candidates.copy()

    candidates[
        "cross_encoder_score"
    ] = rerank_scores

    return candidates

# ============================================================
# CONCEPT-AWARE RETRIEVAL
# ============================================================

def concept_aware_retrieve(
    query: str,
    primary_category: str | None = None,
    secondary_category: str | None = None,
    top_k: int = 5,
    candidate_k: int = 40,
    contract: str | None = None,
):
    """
    Main retrieval function.

    Candidates come from:

        1. Original risk query
        2. Canonical FIDIC concepts
        3. FIDIC contractual phrases

    Final results are deduplicated at:

        CONTRACT + CLAUSE

    After selecting a clause, ALL chunks belonging to
    that clause are assembled.

    Explicitly referenced clauses are also retrieved and
    stored inside referenced_clauses.
    """

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
    # Retrieve candidates from every expanded query
    # --------------------------------------------------------

    candidate_rows = {}

    batch_results = retrieve_batch(
        expanded_queries,
        top_k=candidate_k,
    )

    for results in batch_results:

        for row in results:

            chunk_id = row["chunk_id"]

            semantic_score = float(
                row["semantic_score"]
            )

            if (
                chunk_id not in candidate_rows
                or semantic_score
                > candidate_rows[
                    chunk_id
                ]["semantic_score"]
            ):

                candidate_rows[
                    chunk_id
                ] = row.to_dict()

    # --------------------------------------------------------
    # No candidates
    # --------------------------------------------------------

    if not candidate_rows:

        return pd.DataFrame()

    candidates = pd.DataFrame(
        list(
            candidate_rows.values()
        )
    )

    # --------------------------------------------------------
    # Contract filtering
    # --------------------------------------------------------

    if contract:

        candidates = candidates[
            candidates["contract"].astype(str)
            == contract
        ].copy()

    if candidates.empty:

        return pd.DataFrame()

    # ========================================================
    # CONCEPT SCORE
    # ========================================================

    candidates[
        "concept_score"
    ] = candidates.apply(
        lambda row: concept_match_score(
            clause_title=row["clause_title"],
            clause_text=row["chunk_text"],
            concepts=concepts,
        ),
        axis=1,
    )

    # ========================================================
    # TITLE SCORE
    # ========================================================

    candidates[
        "title_score"
    ] = candidates.apply(
        lambda row: title_relevance(
            clause_title=row["clause_title"],
            concepts=concepts,
        ),
        axis=1,
    )

    # ========================================================
    # CLAUSE-SPECIFIC SIGNAL SCORE
    # ========================================================

    candidates[
        "clause_signal_score"
    ] = candidates.apply(
        lambda row: clause_signal_score(
            clause_number=row["clause_number"],
            clause_title=row["clause_title"],
            clause_text=row["chunk_text"],
        ),
        axis=1,
    )

    # ========================================================
    # CROSS-ENCODER RERANKING
    # ========================================================

    candidates = cross_encoder_rerank(
        query=query,
        candidates=candidates,
        batch_size=16,
    )

    # ========================================================
    # PER-CHUNK SCORE (used only to pick the best chunk
    # within each clause during deduplication below)
    # ========================================================

    candidates["final_score"] = (
        0.45 * candidates["cross_encoder_score"]
        + 0.25 * candidates["semantic_score"]
        + 0.15 * candidates["concept_score"]
        + 0.05 * candidates["title_score"]
        + 0.10 * candidates["clause_signal_score"]
    )

    # ========================================================
    # CLAUSE-LEVEL DEDUPLICATION
    # ========================================================

    candidates[
        "clause_key"
    ] = (
        candidates["contract"].astype(str)
        + "||"
        + candidates["clause_number"].astype(str)
    )

    clause_results = []

    for _, group in candidates.groupby(
        "clause_key"
    ):

        # ----------------------------------------------------
        # Pick strongest chunk
        # ----------------------------------------------------

        best_index = group[
            "final_score"
        ].idxmax()

        best_row = group.loc[
            best_index
        ].copy()

        contract_value = str(
            best_row["contract"]
        )

        clause_number = str(
            best_row["clause_number"]
        )

        # ----------------------------------------------------
        # Assemble COMPLETE clause
        # ----------------------------------------------------

        full_clause = assemble_full_clause_text(
            contract=contract_value,
            clause_number=clause_number,
        )

        # ----------------------------------------------------
        # Replace chunk text with complete clause
        # ----------------------------------------------------

        best_row[
            "chunk_text"
        ] = full_clause[
            "chunk_text"
        ]

        # ----------------------------------------------------
        # Preserve all source chunks
        # ----------------------------------------------------

        best_row[
            "source_chunk_ids"
        ] = " | ".join(
            full_clause[
                "source_chunk_ids"
            ]
        )

        best_row[
            "chunk_count"
        ] = full_clause[
            "chunk_count"
        ]

        # ----------------------------------------------------
        # Best scores across retrieved chunks
        #
        # NOTE: all four component scores (including
        # cross_encoder_score) MUST be updated to their
        # group-max BEFORE final_score is recalculated below,
        # otherwise final_score is computed against a stale
        # cross_encoder_score and the two fields disagree.
        # ----------------------------------------------------

        best_row[
            "semantic_score"
        ] = group[
            "semantic_score"
        ].max()

        best_row[
            "concept_score"
        ] = group[
            "concept_score"
        ].max()

        best_row[
            "title_score"
        ] = group[
            "title_score"
        ].max()

        best_row[
            "clause_signal_score"
        ] = group[
            "clause_signal_score"
        ].max()

        best_row[
            "cross_encoder_score"
        ] = group[
            "cross_encoder_score"
        ].max()

        # ----------------------------------------------------
        # Recalculate FINAL CLAUSE SCORE
        # (now uses the updated group-max cross_encoder_score)
        # ----------------------------------------------------

        best_row[
            "final_score"
        ] = (
            0.50
            * best_row["cross_encoder_score"]
            +
            0.25
            * best_row["semantic_score"]
            +
            0.15
            * best_row["concept_score"]
            +
            0.10
            * best_row["title_score"]
            + 
            0.10 
            * best_row["clause_signal_score"]
        )

        # ----------------------------------------------------
        # Retrieve explicitly referenced clauses
        # ----------------------------------------------------

        referenced_clauses = (
            retrieve_referenced_clauses(
                contract=contract_value,
                clause_number=clause_number,
                clause_text=best_row[
                    "chunk_text"
                ],
            )
        )

        best_row[
            "referenced_clauses"
        ] = referenced_clauses

        clause_results.append(
            best_row
        )

    # ========================================================
    # CONVERT BACK TO DATAFRAME
    # ========================================================

    final_results = pd.DataFrame(
        clause_results
    )

    if final_results.empty:

        return final_results

    # ========================================================
    # FINAL CLAUSE RANKING
    # ========================================================

    final_results = (
        final_results
        .sort_values(
            "final_score",
            ascending=False,
        )
        .head(top_k)
        .copy()
    )

    final_results = (
        final_results.reset_index(
            drop=True
        )
    )

    final_results[
        "rank"
    ] = range(
        1,
        len(final_results) + 1,
    )

    # --------------------------------------------------------
    # Compatibility with risk_analyzer.py
    # --------------------------------------------------------

    final_results[
        "similarity_score"
    ] = final_results[
        "final_score"
    ]

    return final_results


# ============================================================
# MULTI-QUERY RETRIEVAL
# ============================================================

def multi_query_retrieve(
    risk_description: str,
    primary_category: str = None,
    secondary_category: str = None,
    contract: str = None,
    top_k: int = 5,
):
    """
    Retrieve relevant FIDIC clauses using multiple
    query perspectives.

    Perspectives:

        1. Original risk
        2. Contractual treatment

    Results are deduplicated at CONTRACT + CLAUSE level.

    Each selected clause contains:

        - Complete clause text
        - Source chunk IDs
        - Chunk count
        - Referenced clauses
    """

    perspectives = [

        risk_description,

        (
            "FIDIC contractual treatment, entitlement, "
            "notice, claims, obligations, cost and delay: "
            f"{risk_description}"
        ),
    ]

    results = []

    # ========================================================
    # RUN EVERY PERSPECTIVE
    # ========================================================

    for query in perspectives:

        retrieved = concept_aware_retrieve(
            query=query,
            primary_category=primary_category,
            secondary_category=secondary_category,
            contract=contract,
            top_k=top_k,
        )

        if (
            retrieved is not None
            and not retrieved.empty
        ):

            results.append(
                retrieved
            )

    # ========================================================
    # FALLBACK
    # ========================================================

    if not results:

        fallback = retrieve(
            risk_description,
            top_k=40,
        )

        if contract:

            fallback = fallback[
                fallback["contract"].astype(str)
                == contract
            ].head(
                top_k
            )

        return fallback

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    combined[
        "clause_key"
    ] = (
        combined["contract"].astype(str)
        + "||"
        + combined["clause_number"].astype(str)
    )

    combined = (
        combined
        .sort_values(
            "similarity_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "clause_key"
            ],
            keep="first",
        )
    )

    # ========================================================
    # FINAL RANKING
    # ========================================================

    combined = (
        combined
        .sort_values(
            "similarity_score",
            ascending=False,
        )
        .head(
            top_k
        )
        .copy()
    )

    combined = (
        combined.reset_index(
            drop=True
        )
    )

    combined[
        "rank"
    ] = range(
        1,
        len(combined) + 1,
    )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    combined = combined.drop(
        columns=[
            "clause_key"
        ],
        errors="ignore",
    )

    return combined


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "Unexpected geological conditions are "
        "causing additional excavation costs "
        "and delaying the work."
    )

    print("\n")
    print("=" * 80)
    print("FIDIC CONCEPT-EXPANSION RETRIEVAL")
    print("=" * 80)

    # ========================================================
    # DETECT CONCEPTS
    # ========================================================

    concepts = detect_concepts(
        query=query,
        primary_category="SITE_PHYSICAL",
        secondary_category="COST_FINANCIAL",
    )

    print("\nDetected concepts:")

    for concept in concepts:

        print(
            f"  - {concept}: "
            f"{FIDIC_CONCEPTS[concept]['canonical']}"
        )

    # ========================================================
    # EXPANDED QUERIES
    # ========================================================

    expanded = expand_query(
        query=query,
        primary_category="SITE_PHYSICAL",
        secondary_category="COST_FINANCIAL",
    )

    print("\nExpanded contractual queries:")

    for item in expanded:

        print(
            f"  - {item}"
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    print("\nRESULTS:")

    results = concept_aware_retrieve(
        query=query,
        primary_category="SITE_PHYSICAL",
        secondary_category="COST_FINANCIAL",
        top_k=10,
        candidate_k=40,
    )

    if results.empty:

        print(
            "\nNo relevant clauses found."
        )

    else:

        for _, row in results.iterrows():

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Rank {row['rank']}"
                f" | Semantic: "
                f"{row['semantic_score']:.4f}"
                f" | Concept: "
                f"{row['concept_score']:.4f}"
                f" | Title: "
                f"{row['title_score']:.4f}"
                f" | Clause Signal: "
                f"{row['clause_signal_score']:.4f}"
                f" | Cross-Encoder: "
                f"{row['cross_encoder_score']:.4f}"
                f" | Final: "
                f"{row['final_score']:.4f}"
            )

            print(
                f"Contract : "
                f"{row['contract']}"
            )

            print(
                f"Clause   : "
                f"{row['clause_number']}"
            )

            print(
                f"Title    : "
                f"{row['clause_title']}"
            )

            print(
                f"Chunk Count : "
                f"{row['chunk_count']}"
            )

            print(
                f"Source Chunks : "
                f"{row['source_chunk_ids']}"
            )

            print(
                f"Full Clause Text Length : "
                f"{len(str(row['chunk_text']))} characters"
            )

            # ------------------------------------------------
            # Referenced clauses
            # ------------------------------------------------

            references = row.get(
                "referenced_clauses",
                [],
            )

            print(
                "\nReferenced Clauses:"
            )

            if not references:

                print(
                    "  None found."
                )

            else:

                for ref in references:

                    print(
                        f"  - "
                        f"{ref.get('clause_number', 'N/A')}"
                        f" | "
                        f"{ref.get('clause_title', 'N/A')}"
                        f" | "
                        f"Source: "
                        f"{ref.get('reference_source', 'N/A')}"
                    )