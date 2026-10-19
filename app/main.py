import os
from fastapi import FastAPI, HTTPException
from app.schemas import RiskRequest, RiskAnalysisResponse
from src.analysis.risk_analyzer import analyze_risk
from app.response_builder import build_api_response
from src.retrieval.retrieve import assemble_full_clause_text
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="FIDIC Risk Analyzer API",
    description="AI-powered FIDIC contract risk analysis system",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_URLS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "FIDIC Risk Analyzer API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/clauses/{contract}/{clause_number}")
def get_clause(
    contract: str,
    clause_number: str,
):
    """
    Return the complete text of a specific FIDIC clause.
    """

    clause = assemble_full_clause_text(
        contract=contract,
        clause_number=clause_number,
    )

    if clause["chunk_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Clause {clause_number} not found "
                f"for contract '{contract}'."
            ),
        )

    return {
        "contract": contract,
        "clause_number": clause_number,
        "clause_text": clause["chunk_text"],
        "source_chunk_ids": clause["source_chunk_ids"],
        "chunk_count": clause["chunk_count"],
    }


@app.get("/status")
def status():
    return {
        "api": "running",
        "embedding_model": "loaded",
        "faiss_index": "loaded",
        "retrieval": "ready",
        "contract_analysis": "ready",
    }


@app.post("/analyze", response_model=RiskAnalysisResponse)
def analyze(request: RiskRequest):

    try:
        result = analyze_risk(
            risk_description=request.risk_description,
            probability=request.probability,
            impact=request.impact,
            contract=request.contract,
            top_k=request.top_k,
        )

        return build_api_response(result)


    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. "
                    "Please wait and try again later."
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {error_message}",
        )