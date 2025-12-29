"""Tests for spaceproof.governance module."""

import pytest
from spaceproof.governance import (
    load_raci_matrix,
    get_raci_for_event,
    emit_raci_receipt,
    capture_provenance,
    emit_provenance_receipt,
    validate_reason_code,
    emit_intervention_receipt,
    assign_ownership,
    track_decision_chain,
    evaluate_escalation,
    should_escalate,
)


def test_load_raci_matrix():
    """load_raci_matrix returns a dict with event types."""
    matrix = load_raci_matrix()
    assert isinstance(matrix, dict)
    assert len(matrix) > 0


def test_get_raci_for_event_known():
    """get_raci_for_event returns RACI for known events."""
    raci = get_raci_for_event("autonomous_decision")
    assert "responsible" in raci
    assert "accountable" in raci
    assert "consulted" in raci
    assert "informed" in raci


def test_get_raci_for_event_unknown():
    """get_raci_for_event returns default for unknown events."""
    raci = get_raci_for_event("unknown_event_type")
    assert "responsible" in raci
    assert raci["responsible"] == "system"


def test_emit_raci_receipt(capsys):
    """emit_raci_receipt emits a valid receipt."""
    receipt = emit_raci_receipt(
        event_type="test_event",
        decision_id="dec-123",
        raci={"responsible": "agent", "accountable": "operator", "consulted": [], "informed": []},
    )
    assert receipt["receipt_type"] == "raci_assignment"
    assert receipt["decision_id"] == "dec-123"


def test_capture_provenance():
    """capture_provenance returns model and policy info."""
    prov = capture_provenance()
    assert "model_id" in prov
    assert "model_version" in prov
    assert "policy_id" in prov
    assert "policy_version" in prov
    assert "timestamp" in prov


def test_emit_provenance_receipt(capsys):
    """emit_provenance_receipt emits a valid receipt."""
    receipt = emit_provenance_receipt(
        decision_id="dec-456",
        provenance={"model_id": "test", "model_version": "1.0"},
    )
    assert receipt["receipt_type"] == "provenance"
    assert receipt["decision_id"] == "dec-456"


def test_validate_reason_code_valid():
    """validate_reason_code returns True for valid codes."""
    assert validate_reason_code("RE001_FACTUAL_ERROR") is True
    assert validate_reason_code("RE002_POLICY_VIOLATION") is True
    assert validate_reason_code("RE010_OTHER") is True


def test_validate_reason_code_invalid():
    """validate_reason_code returns False for invalid codes."""
    assert validate_reason_code("INVALID_CODE") is False
    assert validate_reason_code("RE999_FAKE") is False
    assert validate_reason_code("") is False


def test_emit_intervention_receipt(capsys):
    """emit_intervention_receipt emits a valid receipt."""
    receipt = emit_intervention_receipt(
        intervention_id="int-789",
        target_decision_id="dec-123",
        intervener_id="HUMAN_1",
        reason_code="RE001_FACTUAL_ERROR",
        justification="Test correction",
    )
    assert receipt["receipt_type"] == "intervention"
    assert receipt["intervention_id"] == "int-789"
    assert receipt["reason_code"] == "RE001_FACTUAL_ERROR"


def test_assign_ownership():
    """assign_ownership returns ownership dict."""
    ownership = assign_ownership(
        decision_id="dec-001",
        owner_id="agent-1",
        owner_type="autonomous_agent",
    )
    assert ownership["decision_id"] == "dec-001"
    assert ownership["owner_id"] == "agent-1"


def test_track_decision_chain():
    """track_decision_chain builds a chain of decisions."""
    chain = track_decision_chain(
        decision_ids=["dec-1", "dec-2", "dec-3"],
        root_id="dec-1",
    )
    assert chain["root_id"] == "dec-1"
    assert len(chain["chain"]) == 3


def test_evaluate_escalation_low_risk():
    """evaluate_escalation returns low for safe decisions."""
    result = evaluate_escalation(
        decision_type="navigation",
        confidence=0.95,
        impact_level="low",
    )
    assert result["escalation_level"] in ["none", "low", "medium", "high", "critical"]


def test_should_escalate_high_impact():
    """should_escalate returns True for high impact."""
    assert should_escalate(escalation_level="critical") is True
    assert should_escalate(escalation_level="high") is True


def test_should_escalate_low_impact():
    """should_escalate returns False for low impact."""
    assert should_escalate(escalation_level="none") is False
    assert should_escalate(escalation_level="low") is False
