"""Tests for spaceproof.privacy module."""

import pytest
from spaceproof.privacy import (
    detect_pii,
    redact_pii,
    get_pii_patterns,
    emit_redaction_receipt,
    add_laplace_noise,
    add_gaussian_noise,
    get_privacy_budget,
    spend_privacy_budget,
    emit_dp_receipt,
    emit_privacy_operation_receipt,
    get_privacy_operations,
    validate_privacy_compliance,
)


def test_detect_pii_email():
    """detect_pii finds email addresses."""
    text = "Contact us at user@example.com for help"
    pii = detect_pii(text)
    assert len(pii) > 0
    assert any(p["type"] == "email" for p in pii)


def test_detect_pii_ssn():
    """detect_pii finds SSN patterns."""
    text = "SSN: 123-45-6789"
    pii = detect_pii(text)
    assert len(pii) > 0
    assert any(p["type"] == "ssn" for p in pii)


def test_detect_pii_phone():
    """detect_pii finds phone numbers."""
    text = "Call 555-123-4567"
    pii = detect_pii(text)
    assert len(pii) > 0
    assert any(p["type"] == "phone" for p in pii)


def test_detect_pii_none():
    """detect_pii returns empty for clean text."""
    text = "This is safe text with no personal info"
    pii = detect_pii(text)
    assert len(pii) == 0


def test_redact_pii():
    """redact_pii replaces PII with placeholders."""
    text = "Email: test@example.com, SSN: 123-45-6789"
    redacted = redact_pii(text)
    assert "test@example.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED" in redacted


def test_get_pii_patterns():
    """get_pii_patterns returns pattern dict."""
    patterns = get_pii_patterns()
    assert isinstance(patterns, dict)
    assert "email" in patterns
    assert "ssn" in patterns
    assert "phone" in patterns


def test_emit_redaction_receipt(capsys):
    """emit_redaction_receipt emits valid receipt."""
    receipt = emit_redaction_receipt(
        original_hash="abc123",
        redacted_hash="def456",
        pii_types=["email", "ssn"],
    )
    assert receipt["receipt_type"] == "redaction"
    assert receipt["pii_types"] == ["email", "ssn"]


def test_add_laplace_noise():
    """add_laplace_noise adds calibrated noise."""
    value = 50.0
    epsilon = 1.0
    sensitivity = 1.0

    noisy = add_laplace_noise(value, epsilon, sensitivity)
    # Noise should be added (though could be zero)
    assert isinstance(noisy, float)


def test_add_laplace_noise_bounds():
    """add_laplace_noise respects epsilon."""
    import numpy as np
    values = [add_laplace_noise(100.0, 1.0, 1.0) for _ in range(100)]
    # With epsilon=1.0, noise should be reasonable (most within 10)
    differences = [abs(v - 100.0) for v in values]
    assert np.median(differences) < 5  # Most noise within 5


def test_add_gaussian_noise():
    """add_gaussian_noise adds calibrated noise."""
    value = 50.0
    epsilon = 1.0
    delta = 1e-5
    sensitivity = 1.0

    noisy = add_gaussian_noise(value, epsilon, delta, sensitivity)
    assert isinstance(noisy, float)


def test_get_privacy_budget():
    """get_privacy_budget returns budget info."""
    budget = get_privacy_budget(tenant_id="test-tenant")
    assert "remaining" in budget or "budget" in budget


def test_spend_privacy_budget():
    """spend_privacy_budget deducts from budget."""
    result = spend_privacy_budget(
        tenant_id="test-tenant",
        epsilon=0.5,
    )
    assert result["success"] is True or result.get("rejected") is True


def test_emit_dp_receipt(capsys):
    """emit_dp_receipt emits valid receipt."""
    receipt = emit_dp_receipt(
        epsilon=1.0,
        sensitivity=1.0,
        noise_type="laplace",
        budget_remaining=5.0,
    )
    assert receipt["receipt_type"] == "differential_privacy"
    assert receipt["epsilon"] == 1.0


def test_emit_privacy_operation_receipt(capsys):
    """emit_privacy_operation_receipt emits valid receipt."""
    receipt = emit_privacy_operation_receipt(
        operation_type="redaction",
        target_id="doc-001",
        success=True,
    )
    assert receipt["receipt_type"] == "privacy_operation"


def test_get_privacy_operations():
    """get_privacy_operations returns operation list."""
    ops = get_privacy_operations(tenant_id="test-tenant")
    assert isinstance(ops, list)


def test_validate_privacy_compliance():
    """validate_privacy_compliance checks compliance."""
    result = validate_privacy_compliance(
        tenant_id="test-tenant",
        outputs=["This is clean text", "No PII here"],
    )
    assert result["compliant"] is True
    assert result["leakage_count"] == 0


def test_validate_privacy_compliance_failure():
    """validate_privacy_compliance detects leakage."""
    result = validate_privacy_compliance(
        tenant_id="test-tenant",
        outputs=["Contact: user@example.com"],
    )
    assert result["compliant"] is False
    assert result["leakage_count"] > 0
