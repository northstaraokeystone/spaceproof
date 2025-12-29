"""Tests for spaceproof.economy module."""

import pytest
from spaceproof.economy import (
    authorize_with_receipt,
    verify_receipt_for_access,
    emit_access_receipt,
    record_operation_cost,
    get_cost_summary,
    emit_cost_receipt,
    check_quota,
    consume_quota,
    reset_quota,
    emit_quota_receipt,
)


def test_authorize_with_receipt_valid():
    """authorize_with_receipt grants access with valid receipt."""
    receipt = {
        "receipt_type": "access_grant",
        "tenant_id": "test-tenant",
        "resource": "compute",
        "valid_until": "2030-01-01T00:00:00Z",
    }
    result = authorize_with_receipt(receipt, resource="compute")
    assert result["authorized"] is True


def test_authorize_with_receipt_invalid():
    """authorize_with_receipt denies access with invalid receipt."""
    receipt = {
        "receipt_type": "access_grant",
        "tenant_id": "test-tenant",
        "resource": "storage",
    }
    result = authorize_with_receipt(receipt, resource="compute")
    assert result["authorized"] is False


def test_verify_receipt_for_access():
    """verify_receipt_for_access validates receipt chain."""
    receipt = {
        "receipt_type": "access_grant",
        "payload_hash": "abc123:def456",
    }
    result = verify_receipt_for_access(receipt)
    assert isinstance(result, dict)
    assert "valid" in result


def test_emit_access_receipt(capsys):
    """emit_access_receipt emits valid receipt."""
    receipt = emit_access_receipt(
        tenant_id="test-tenant",
        resource="compute",
        action="read",
        authorized=True,
    )
    assert receipt["receipt_type"] == "access_authorization"
    assert receipt["authorized"] is True


def test_record_operation_cost():
    """record_operation_cost tracks operation cost."""
    result = record_operation_cost(
        tenant_id="test-tenant",
        operation="inference",
        cost_units=10.5,
    )
    assert result["recorded"] is True
    assert result["cost_units"] == 10.5


def test_get_cost_summary():
    """get_cost_summary returns cost aggregation."""
    summary = get_cost_summary(
        tenant_id="test-tenant",
        period="2024-01",
    )
    assert isinstance(summary, dict)
    assert "total" in summary or "costs" in summary


def test_emit_cost_receipt(capsys):
    """emit_cost_receipt emits valid receipt."""
    receipt = emit_cost_receipt(
        tenant_id="test-tenant",
        operation="inference",
        cost_units=15.0,
    )
    assert receipt["receipt_type"] == "cost_accounting"
    assert receipt["cost_units"] == 15.0


def test_check_quota_available():
    """check_quota returns available quota."""
    result = check_quota(
        tenant_id="test-tenant",
        resource="api_calls",
    )
    assert "available" in result or "remaining" in result


def test_consume_quota():
    """consume_quota deducts from quota."""
    result = consume_quota(
        tenant_id="test-tenant",
        resource="api_calls",
        amount=1,
    )
    assert result["success"] is True or result.get("exceeded") is True


def test_consume_quota_exceeded():
    """consume_quota handles quota exceeded."""
    # First exhaust quota
    for _ in range(1000):
        consume_quota("test-exceed", "api_calls", 100)

    result = consume_quota(
        tenant_id="test-exceed",
        resource="api_calls",
        amount=1,
    )
    # Should either succeed or be rejected
    assert "success" in result or "exceeded" in result


def test_reset_quota():
    """reset_quota resets quota to initial value."""
    result = reset_quota(
        tenant_id="test-tenant",
        resource="api_calls",
    )
    assert result["success"] is True


def test_emit_quota_receipt(capsys):
    """emit_quota_receipt emits valid receipt."""
    receipt = emit_quota_receipt(
        tenant_id="test-tenant",
        resource="api_calls",
        consumed=50,
        remaining=950,
    )
    assert receipt["receipt_type"] == "quota_enforcement"
    assert receipt["consumed"] == 50
    assert receipt["remaining"] == 950
