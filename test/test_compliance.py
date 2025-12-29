"""Tests for spaceproof.compliance module."""

import pytest
from spaceproof.compliance import (
    generate_audit_trail,
    get_audit_trail,
    emit_audit_receipt,
    generate_raci_report,
    get_raci_coverage,
    emit_raci_report_receipt,
    generate_intervention_report,
    get_intervention_metrics,
    emit_intervention_report_receipt,
    generate_provenance_report,
    get_provenance_history,
    emit_provenance_report_receipt,
)


def test_generate_audit_trail():
    """generate_audit_trail creates trail within SLO."""
    import time
    start = time.time()
    trail = generate_audit_trail(
        tenant_id="test-tenant",
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-02T00:00:00Z",
    )
    elapsed_ms = (time.time() - start) * 1000

    assert trail["tenant_id"] == "test-tenant"
    assert "entries" in trail
    assert elapsed_ms < 5000  # 5 second SLO


def test_get_audit_trail():
    """get_audit_trail retrieves existing trail."""
    trail = get_audit_trail(trail_id="trail-001")
    assert isinstance(trail, dict)


def test_emit_audit_receipt(capsys):
    """emit_audit_receipt emits valid receipt."""
    receipt = emit_audit_receipt(
        trail_id="trail-001",
        entry_count=100,
        generation_time_ms=1500,
    )
    assert receipt["receipt_type"] == "audit_trail"
    assert receipt["entry_count"] == 100


def test_generate_raci_report():
    """generate_raci_report creates accountability report."""
    report = generate_raci_report(
        tenant_id="test-tenant",
        period="2024-01",
    )
    assert report["tenant_id"] == "test-tenant"
    assert "coverage" in report or "decisions" in report


def test_get_raci_coverage():
    """get_raci_coverage returns coverage percentage."""
    coverage = get_raci_coverage(tenant_id="test-tenant")
    assert isinstance(coverage, dict)
    assert "percentage" in coverage or isinstance(coverage.get("coverage"), (int, float))


def test_emit_raci_report_receipt(capsys):
    """emit_raci_report_receipt emits valid receipt."""
    receipt = emit_raci_report_receipt(
        report_id="raci-001",
        coverage_pct=98.5,
    )
    assert receipt["receipt_type"] == "raci_report"


def test_generate_intervention_report():
    """generate_intervention_report creates intervention metrics."""
    report = generate_intervention_report(
        tenant_id="test-tenant",
        period="2024-01",
    )
    assert report["tenant_id"] == "test-tenant"
    assert "interventions" in report or "count" in report


def test_get_intervention_metrics():
    """get_intervention_metrics returns intervention stats."""
    metrics = get_intervention_metrics(tenant_id="test-tenant")
    assert isinstance(metrics, dict)


def test_emit_intervention_report_receipt(capsys):
    """emit_intervention_report_receipt emits valid receipt."""
    receipt = emit_intervention_report_receipt(
        report_id="int-report-001",
        intervention_count=15,
        by_reason_code={"RE001": 5, "RE002": 10},
    )
    assert receipt["receipt_type"] == "intervention_report"
    assert receipt["intervention_count"] == 15


def test_generate_provenance_report():
    """generate_provenance_report creates version history."""
    report = generate_provenance_report(
        tenant_id="test-tenant",
        period="2024-01",
    )
    assert report["tenant_id"] == "test-tenant"
    assert "versions" in report or "history" in report


def test_get_provenance_history():
    """get_provenance_history returns model/policy versions."""
    history = get_provenance_history(
        tenant_id="test-tenant",
        artifact_type="model",
    )
    assert isinstance(history, (list, dict))


def test_emit_provenance_report_receipt(capsys):
    """emit_provenance_report_receipt emits valid receipt."""
    receipt = emit_provenance_report_receipt(
        report_id="prov-report-001",
        model_versions=3,
        policy_versions=2,
    )
    assert receipt["receipt_type"] == "provenance_report"
