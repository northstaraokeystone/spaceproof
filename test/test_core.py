"""Tests for spaceproof.core module."""

import pytest
from spaceproof.core import dual_hash, emit_receipt, merkle, StopRule


def test_dual_hash_bytes():
    """dual_hash returns SHA256:BLAKE3 format for bytes."""
    result = dual_hash(b"test data")
    assert ":" in result
    parts = result.split(":")
    assert len(parts) == 2
    assert len(parts[0]) == 64  # SHA256 hex
    assert len(parts[1]) == 64  # BLAKE3 hex (or duplicate SHA256)


def test_dual_hash_string():
    """dual_hash accepts string input."""
    result = dual_hash("test string")
    assert ":" in result


def test_dual_hash_deterministic():
    """dual_hash is deterministic."""
    h1 = dual_hash(b"same data")
    h2 = dual_hash(b"same data")
    assert h1 == h2


def test_emit_receipt_returns_dict(capsys):
    """emit_receipt returns a receipt dict with required fields."""
    receipt = emit_receipt("test", {"key": "value"})

    assert receipt["receipt_type"] == "test"
    assert "ts" in receipt
    assert "tenant_id" in receipt
    assert "payload_hash" in receipt
    assert receipt["key"] == "value"

    # Verify it printed JSON to stdout
    captured = capsys.readouterr()
    assert "test" in captured.out


def test_merkle_empty():
    """merkle of empty list returns hash of 'empty'."""
    result = merkle([])
    assert ":" in result


def test_merkle_single():
    """merkle of single item returns hash of that item."""
    result = merkle([{"a": 1}])
    assert ":" in result


def test_merkle_multiple():
    """merkle of multiple items returns valid root."""
    items = [{"a": 1}, {"b": 2}, {"c": 3}]
    result = merkle(items)
    assert ":" in result


def test_merkle_deterministic():
    """merkle is deterministic for same input."""
    items = [{"x": 1}, {"y": 2}]
    r1 = merkle(items)
    r2 = merkle(items)
    assert r1 == r2


def test_stoprule_exception():
    """StopRule can be raised with message."""
    with pytest.raises(StopRule) as exc_info:
        raise StopRule("test error")
    assert "test error" in str(exc_info.value)
