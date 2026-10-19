
import pytest

from src.retrieval.retrieve import assemble_full_clause_text
from src.analysis.risk_scoring import calculate_risk_score


@pytest.mark.parametrize(
    "contract, clause_number",
    [
        ("FIDIC Red Book 2017", "4.12"),
        ("FIDIC Red Book 2017", "8.5"),
    ],
)
def test_clause_reconstruction_returns_content(contract, clause_number):
    result = assemble_full_clause_text(contract, clause_number)

    assert result["chunk_count"] > 0
    assert len(result["chunk_text"].strip()) > 0
    assert result["source_chunk_ids"]


def test_physical_conditions_clause_contains_expected_content():
    result = assemble_full_clause_text(
        "FIDIC Red Book 2017", "4.12"
    )

    text = result["chunk_text"].lower()

    assert "physical conditions" in text
    assert "unforeseeable" in text
    assert "cost" in text


def test_extension_of_time_clause_contains_expected_content():
    result = assemble_full_clause_text(
        "FIDIC Red Book 2017", "8.5"
    )

    text = result["chunk_text"].lower()

    assert len(text) > 100
    assert "extension of time" in text or "completion" in text



def test_risk_score_for_probability_4_impact_4():
    result = calculate_risk_score(4, 4)

    assert isinstance(result, dict)
    assert result["probability"] == 4
    assert result["impact"] == 4

    # Verify the calculated score without assuming its field name.
    score_fields = [
        key for key in result
        if "score" in key.lower()
    ]

    assert score_fields, f"No score field found in: {result}"
    assert any(result[key] == 16 for key in score_fields)

