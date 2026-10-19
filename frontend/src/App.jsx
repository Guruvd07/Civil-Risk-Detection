import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");


function App() {
  const [riskDescription, setRiskDescription] = useState("");
  const [probability, setProbability] = useState(3);
  const [impact, setImpact] = useState(3);
  const [contract, setContract] = useState("FIDIC Red Book 2017");

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const resultRef = useRef(null);

  const [selectedClause, setSelectedClause] = useState(null);
  const [clauseLoading, setClauseLoading] = useState(false);
  const [clauseError, setClauseError] = useState("");

  // --------------------------------------------------
  // ANALYZE RISK
  // --------------------------------------------------

  const handleAnalyze = async () => {
    if (!riskDescription.trim()) {
      setError("Please enter a risk description.");
      return;
    }

    setError("");
    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, {
        risk_description: riskDescription,
        probability,
        impact,
        contract,
        top_k: 5,
      });

      setResult(response.data);

      setTimeout(() => {
        resultRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      if (err.response?.status === 429) {
        setError(
          "Gemini API quota exceeded. Please try again later."
        );
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError(
          "Unable to connect to the risk analysis API."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------
  // LOAD FULL CLAUSE
  // --------------------------------------------------

  const handleClauseClick = async (clause) => {
    setClauseLoading(true);
    setClauseError("");
    setSelectedClause(null);

    try {
      const response = await axios.get(
        `${API_BASE_URL}/clauses/${encodeURIComponent(
          clause.contract
        )}/${encodeURIComponent(clause.clause_number)}`
      );

      setSelectedClause(response.data);
    } catch (err) {
      setClauseError(
        err.response?.data?.detail ||
          "Unable to load clause details."
      );
    } finally {
      setClauseLoading(false);
    }
  };

  // --------------------------------------------------
  // CLOSE CLAUSE MODAL
  // --------------------------------------------------

  const closeClause = () => {
    setSelectedClause(null);
    setClauseError("");
  };

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div className="app">
      <div className="container">

        {/* ==================================================
            HEADER
        ================================================== */}

        <header className="header">
          <div className="header-badge">
            AI CONTRACT INTELLIGENCE
          </div>

          <h1>FIDIC Risk Analyzer</h1>

          <p>
            Identify contractual risks, discover relevant FIDIC
            clauses, and generate AI-powered contractual analysis.
          </p>
        </header>

        {/* ==================================================
            INPUT SECTION
        ================================================== */}

        <section className="card analysis-input-card">

          <div className="section-heading">
            <div>
              <span className="eyebrow">
                RISK ASSESSMENT
              </span>

              <h2>Analyze a Contractual Risk</h2>

              <p>
                Provide the risk details below to identify the
                relevant FIDIC contractual provisions.
              </p>
            </div>
          </div>

          {/* RISK DESCRIPTION */}

          <div className="form-group">
            <label htmlFor="riskDescription">
              Risk Description
            </label>

            <textarea
              id="riskDescription"
              value={riskDescription}
              onChange={(e) =>
                setRiskDescription(e.target.value)
              }
              placeholder="Example: Unexpected geological conditions are causing additional excavation costs and delaying the work..."
              rows={6}
            />

            <span className="input-hint">
              Describe the event, its potential impact, and any
              contractual consequences.
            </span>
          </div>

          {/* FORM GRID */}

          <div className="form-grid">

            {/* PROBABILITY */}

            <div className="form-group">
              <label htmlFor="probability">
                Probability
              </label>

              <select
                id="probability"
                value={probability}
                onChange={(e) =>
                  setProbability(Number(e.target.value))
                }
              >
                <option value={1}>
                  1 — Very Low
                </option>

                <option value={2}>
                  2 — Low
                </option>

                <option value={3}>
                  3 — Medium
                </option>

                <option value={4}>
                  4 — High
                </option>

                <option value={5}>
                  5 — Very High
                </option>
              </select>
            </div>

            {/* IMPACT */}

            <div className="form-group">
              <label htmlFor="impact">
                Impact
              </label>

              <select
                id="impact"
                value={impact}
                onChange={(e) =>
                  setImpact(Number(e.target.value))
                }
              >
                <option value={1}>
                  1 — Very Low
                </option>

                <option value={2}>
                  2 — Low
                </option>

                <option value={3}>
                  3 — Medium
                </option>

                <option value={4}>
                  4 — High
                </option>

                <option value={5}>
                  5 — Very High
                </option>
              </select>
            </div>

            {/* CONTRACT */}

            <div className="form-group">
              <label htmlFor="contract">
                FIDIC Contract
              </label>

              <select
                id="contract"
                value={contract}
                onChange={(e) =>
                  setContract(e.target.value)
                }
              >
                <option value="FIDIC Red Book 2017">
                  FIDIC Red Book 2017
                </option>

                <option value="FIDIC Silver Book 2017">
                  FIDIC Silver Book 2017
                </option>

                <option value="FIDIC Silver Book 1999">
                  FIDIC Silver Book 1999
                </option>
              </select>
            </div>
          </div>

          {/* ANALYZE ACTION */}

          <div className="analysis-action">

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading
                ? "Analyzing Contract..."
                : "Analyze Risk"}
            </button>

            <span className="analysis-note">
              Powered by FIDIC clause retrieval + AI reasoning
            </span>
          </div>

          {/* ERROR */}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        {/* ==================================================
            RESULTS
        ================================================== */}

        {result && (
          <section
            className="results"
            ref={resultRef}
          >

            {/* ==================================================
                RESULT HEADER
            ================================================== */}

            <div className="result-header card">

              <div>
                <span className="eyebrow">
                  ANALYSIS RESULT
                </span>

                <h2>Risk Assessment</h2>
              </div>

              <span className="risk-badge">
                {result.risk?.level || "N/A"}
              </span>
            </div>

            {/* ==================================================
                SUMMARY
            ================================================== */}

              <div className="summary-grid">

              {/* RISK SCORE */}

              <div className="summary-card risk-score-card">
                <span>RISK SCORE</span>

                <strong className="risk-score-value">
                  {result.risk?.score ?? "-"}
                </strong>

                <small>
                  Probability × Impact
                </small>
              </div>

              {/* RISK LEVEL */}

              <div className="summary-card risk-level-card">
                <span>RISK LEVEL</span>

                <strong className="risk-level-value">
                  {result.risk?.level || "-"}
                </strong>

                <small>
                  Overall assessment
                </small>
              </div>

              {/* PROBABILITY */}

              <div className="summary-card">
                <span>PROBABILITY</span>

                <strong>
                  {result.risk?.probability ?? "-"}
                  <small className="scale-text"> / 5</small>
                </strong>

                <small>
                  Likelihood of occurrence
                </small>
              </div>

              {/* IMPACT */}

              <div className="summary-card">
                <span>IMPACT</span>

                <strong>
                  {result.risk?.impact ?? "-"}
                  <small className="scale-text"> / 5</small>
                </strong>

                <small>
                  Potential consequence
                </small>
              </div>

              </div>

            {/* ==================================================
                PRIMARY CLAUSE
            ================================================== */}

            {result.primary_clause && (
              <div className="card">

                <span className="eyebrow">
                  PRIMARY CONTRACTUAL BASIS
                </span>

                <h2>
                  Clause{" "}
                  {result.primary_clause.clause_number}
                </h2>

                <h3>
                  {result.primary_clause.clause_title}
                </h3>

                <p>
                  {result.primary_clause.reason || "-"}
                </p>

                <button
                  className="clause-button"
                  onClick={() =>
                    handleClauseClick(
                      result.primary_clause
                    )
                  }
                >
                  View Full Clause
                </button>
              </div>
            )}

            {/* ==================================================
                ENTITLEMENT
            ================================================== */}

            <div className="card">

              <span className="eyebrow">
                ENTITLEMENT
              </span>

              <div className="two-column">

                <div>
                  <span className="field-label">
                    Cost Entitlement
                  </span>

                  <strong>
                    {result.entitlement?.cost?.status ||
                      "-"}
                  </strong>
                </div>

                <div>
                  <span className="field-label">
                    Time Entitlement
                  </span>

                  <strong>
                    {result.entitlement?.time?.status ||
                      "-"}
                  </strong>
                </div>

              </div>
            </div>

            {/* ==================================================
                CLAIM & NOTICE
            ================================================== */}

            <div className="card">

              <span className="eyebrow">
                CLAIM & NOTICE
              </span>

              <div className="four-column">

                <div>
                  <span>
                    Claim Required
                  </span>

                  <strong>
                    {String(
                      result.claim?.required ?? false
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Notice Required
                  </span>

                  <strong>
                    {String(
                      result.claim?.notice_required ??
                        false
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Notice Deadline
                  </span>

                  <strong>
                    {result.claim
                      ?.notice_deadline_days
                      ? `${result.claim.notice_deadline_days} days`
                      : "-"}
                  </strong>
                </div>

                <div>
                  <span>
                    Detailed Claim Deadline
                  </span>

                  <strong>
                    {result.claim
                      ?.detailed_claim_deadline_days
                      ? `${result.claim.detailed_claim_deadline_days} days`
                      : "-"}
                  </strong>
                </div>

              </div>
            </div>

            {/* ==================================================
                CONTRACTUAL MECHANISM
            ================================================== */}

            <div className="card">

              <span className="eyebrow">
                CONTRACTUAL MECHANISM
              </span>

              <p className="mechanism-text">
                {result.contractual_mechanism || "-"}
              </p>
            </div>

            {/* ==================================================
                ACTIONS
            ================================================== */}

            {result.actions && (
              <div className="action-grid">

                {/* CONTRACTOR */}

                <div className="card">

                  <span className="eyebrow">
                    CONTRACTOR ACTIONS
                  </span>

                  <ul>
                    {result.actions.contractor?.map(
                      (action, index) => (
                        <li key={index}>
                          {action}
                        </li>
                      )
                    )}
                  </ul>
                </div>

                {/* EMPLOYER */}

                <div className="card">

                  <span className="eyebrow">
                    EMPLOYER ACTIONS
                  </span>

                  <ul>
                    {result.actions.employer?.map(
                      (action, index) => (
                        <li key={index}>
                          {action}
                        </li>
                      )
                    )}
                  </ul>
                </div>

              </div>
            )}

            {/* ==================================================
                EVIDENCE
            ================================================== */}

            {result.evidence?.length > 0 && (
              <div className="card">

                <span className="eyebrow">
                  CONTRACTUAL EVIDENCE
                </span>

                <div className="evidence-list">

                  {result.evidence.map(
                    (item, index) => (
                      <div
                        className="evidence-item"
                        key={index}
                      >
                        <strong>
                          Clause {item.clause_number}
                        </strong>

                        <span>
                          {item.clause_title}
                        </span>

                        <p>
                          {item.supports}
                        </p>
                      </div>
                    )
                  )}

                </div>
              </div>
            )}

            {/* ==================================================
                CONFIDENCE
            ================================================== */}

            {result.confidence && (
              <div className="card confidence-card">

                <div>

                  <span className="eyebrow">
                    ANALYSIS CONFIDENCE
                  </span>

                  <h2>
                    {result.confidence.level}
                  </h2>

                  <p>
                    {result.confidence.reason}
                  </p>
                </div>

                <div className="confidence-count">

                  <strong>
                    {result.confidence.evidence_count}
                  </strong>

                  <span>
                    Evidence Items
                  </span>

                </div>
              </div>
            )}

            {/* ==================================================
                UNCERTAINTIES
            ================================================== */}

            {result.uncertainties?.length > 0 && (
              <div className="card uncertainty-card">

                <span className="eyebrow">
                  UNCERTAINTIES
                </span>

                <ul>
                  {result.uncertainties.map(
                    (item, index) => (
                      <li key={index}>
                        {item}
                      </li>
                    )
                  )}
                </ul>
              </div>
            )}

            {/* ==================================================
                RELEVANT CLAUSES
            ================================================== */}

            {result.relevant_clauses?.length > 0 && (
              <div className="card">

                <span className="eyebrow">
                  RELEVANT CLAUSES
                </span>

                <div className="clause-list">

                  {result.relevant_clauses.map(
                    (clause, index) => (
                      <button
                        className="clause-row"
                        key={index}
                        onClick={() =>
                          handleClauseClick(clause)
                        }
                      >
                        <strong>
                          {clause.clause_number}
                        </strong>

                        <span>
                          {clause.clause_title}
                        </span>

                        <small>
                          Similarity:{" "}
                          {clause.similarity_score?.toFixed(
                            3
                          )}
                        </small>
                      </button>
                    )
                  )}

                </div>
              </div>
            )}

          </section>
        )}

        {/* ==================================================
            CLAUSE MODAL
        ================================================== */}

        {(selectedClause ||
          clauseLoading ||
          clauseError) && (
          <div
            className="modal-overlay"
            onClick={closeClause}
          >

            <div
              className="clause-modal"
              onClick={(e) =>
                e.stopPropagation()
              }
            >

              {/* CLOSE */}

              <button
                className="close-button"
                onClick={closeClause}
              >
                ×
              </button>

              {/* LOADING */}

              {clauseLoading && (
                <div className="loading">
                  Loading clause...
                </div>
              )}

              {/* ERROR */}

              {clauseError && (
                <div className="error-message">
                  {clauseError}
                </div>
              )}

              {/* CLAUSE CONTENT */}

              {selectedClause && (
                <>

                  <span className="eyebrow">
                    FIDIC CLAUSE
                  </span>

                  <h2>
                    Clause{" "}
                    {selectedClause.clause_number}
                  </h2>

                  <p className="contract-name">
                    {selectedClause.contract}
                  </p>

                  <div className="clause-text">
                    {selectedClause.clause_text}
                  </div>

                  <div className="source-info">

                    <strong>
                      Source Chunks:
                    </strong>

                    <span>
                      {selectedClause.chunk_count}
                    </span>

                    <div>
                      {selectedClause.source_chunk_ids?.map(
                        (id) => (
                          <code key={id}>
                            {id}
                          </code>
                        )
                      )}
                    </div>

                  </div>
                </>
              )}

            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default App;