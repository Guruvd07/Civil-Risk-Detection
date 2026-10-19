from typing import Any, Optional

from pydantic import BaseModel, Field


# =========================================================
# REQUEST
# =========================================================

class RiskRequest(BaseModel):
    risk_description: str = Field(
        ...,
        min_length=1,
        description="Description of the project or contractual risk",
    )

    probability: int = Field(
        ...,
        ge=1,
        le=5,
        description="Probability score from 1 to 5",
    )

    impact: int = Field(
        ...,
        ge=1,
        le=5,
        description="Impact score from 1 to 5",
    )

    contract: str = Field(
        default="FIDIC Red Book 2017",
        description="FIDIC contract edition",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant clauses to retrieve",
    )


# =========================================================
# PRIMARY CLAUSE
# =========================================================

class PrimaryClause(BaseModel):
    contract: str
    clause_number: str
    clause_title: str
    reason: Optional[str] = None


# =========================================================
# ENTITLEMENT
# =========================================================

class Entitlement(BaseModel):
    status: Optional[str] = None
    explanation: Optional[str] = None


# =========================================================
# CLAIM
# =========================================================

class ClaimInfo(BaseModel):
    required: Optional[bool] = None
    notice_required: Optional[bool] = None
    notice_deadline_days: Optional[int] = None
    detailed_claim_deadline_days: Optional[int] = None
    requirements: Optional[str] = None


# =========================================================
# ACTIONS
# =========================================================

class Actions(BaseModel):
    contractor: list[str] = []
    employer: list[str] = []
    recommended_next_steps: list[str] = []


# =========================================================
# CONFIDENCE
# =========================================================

class Confidence(BaseModel):
    level: Optional[str] = None
    evidence_count: Optional[int] = None
    reason: Optional[str] = None


# =========================================================
# EVIDENCE
# =========================================================

class ClauseEvidence(BaseModel):
    contract: str
    clause_number: str
    clause_title: str
    supports: Optional[str] = None
    evidence: Optional[str] = None


# =========================================================
# RELEVANT CLAUSE
# =========================================================

class RelevantClause(BaseModel):
    rank: Optional[int] = None
    similarity_score: Optional[float] = None
    contract: Optional[str] = None
    clause_number: Optional[str] = None
    clause_title: Optional[str] = None
    clause_type: Optional[str] = None
    chunk_count: Optional[int] = None


# =========================================================
# SUPPORTING CLAUSE
# =========================================================

class SupportingClause(BaseModel):
    contract: Optional[str] = None
    clause_number: Optional[str] = None
    clause_title: Optional[str] = None
    clause_type: Optional[str] = None
    reference_source: Optional[str] = None


# =========================================================
# RISK
# =========================================================

class RiskInfo(BaseModel):
    description: str
    score: Optional[int] = None
    level: Optional[str] = None
    probability: Optional[int] = None
    impact: Optional[int] = None


# =========================================================
# CLASSIFICATION
# =========================================================

class ClassificationInfo(BaseModel):
    primary_category: Optional[str] = None
    secondary_category: Optional[str] = None
    risk_summary: Optional[str] = None


# =========================================================
# API RESPONSE
# =========================================================

class RiskAnalysisResponse(BaseModel):
    risk: RiskInfo

    classification: ClassificationInfo

    primary_clause: Optional[PrimaryClause] = None

    entitlement: dict[str, Optional[Entitlement]]

    claim: ClaimInfo

    contractual_mechanism: Optional[str] = None

    actions: Actions

    uncertainties: list[str] = []

    confidence: Optional[Confidence] = None

    evidence: list[ClauseEvidence] = []

    relevant_clauses: list[RelevantClause] = []

    supporting_clauses: list[SupportingClause] = []