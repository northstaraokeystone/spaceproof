"""Tests for spaceproof.offline module."""

import pytest
from spaceproof.offline import (
    sync_offline_receipts,
    calculate_sync_delay,
    emit_sync_receipt,
    resolve_conflict,
    detect_conflicts,
    MergeStrategy,
    emit_conflict_receipt,
    create_offline_ledger,
    append_offline,
    merge_offline_ledger,
    emit_offline_ledger_receipt,
)


def test_sync_offline_receipts():
    """sync_offline_receipts merges receipts from offline nodes."""
    node_receipts = {
        "node_a": [{"receipt_id": "r1", "data": "a"}],
        "node_b": [{"receipt_id": "r2", "data": "b"}],
    }
    result = sync_offline_receipts(node_receipts)
    assert result["success"] is True
    assert result["merged_count"] == 2


def test_calculate_sync_delay():
    """calculate_sync_delay returns delay based on distance."""
    delay = calculate_sync_delay(distance_km=225_000_000)  # Mars at opposition
    # Light delay should be ~12.5 minutes
    assert delay > 700  # At least 700 seconds
    assert delay < 1500  # Less than 25 minutes


def test_emit_sync_receipt(capsys):
    """emit_sync_receipt emits valid receipt."""
    receipt = emit_sync_receipt(
        sync_id="sync-001",
        node_count=3,
        merged_count=150,
        sync_time_ms=5000,
    )
    assert receipt["receipt_type"] == "offline_sync"
    assert receipt["merged_count"] == 150


def test_resolve_conflict_timestamp():
    """resolve_conflict uses timestamp strategy."""
    versions = [
        {"receipt_id": "r1", "timestamp": "2024-01-01T10:00:00Z", "data": "old"},
        {"receipt_id": "r1", "timestamp": "2024-01-01T12:00:00Z", "data": "new"},
    ]
    winner = resolve_conflict(versions, strategy=MergeStrategy.TIMESTAMP)
    assert winner["data"] == "new"


def test_resolve_conflict_hash():
    """resolve_conflict uses hash-based deterministic strategy."""
    versions = [
        {"receipt_id": "r1", "data": "version_a"},
        {"receipt_id": "r1", "data": "version_b"},
    ]
    winner = resolve_conflict(versions, strategy=MergeStrategy.HASH_ORDER)
    assert winner is not None


def test_resolve_conflict_priority():
    """resolve_conflict uses priority strategy."""
    versions = [
        {"receipt_id": "r1", "node_id": "earth", "data": "earth_data"},
        {"receipt_id": "r1", "node_id": "mars", "data": "mars_data"},
    ]
    winner = resolve_conflict(versions, strategy=MergeStrategy.PRIORITY, priority_order=["earth", "mars"])
    assert winner["data"] == "earth_data"


def test_detect_conflicts():
    """detect_conflicts finds conflicting receipts."""
    receipts = [
        {"receipt_id": "r1", "node_id": "a", "data": "v1"},
        {"receipt_id": "r1", "node_id": "b", "data": "v2"},
        {"receipt_id": "r2", "node_id": "a", "data": "v3"},
    ]
    conflicts = detect_conflicts(receipts)
    assert len(conflicts) == 1
    assert "r1" in conflicts


def test_emit_conflict_receipt(capsys):
    """emit_conflict_receipt emits valid receipt."""
    receipt = emit_conflict_receipt(
        receipt_id="r1",
        version_count=2,
        resolution_strategy="hash_order",
        winner_node="node_a",
    )
    assert receipt["receipt_type"] == "conflict_resolution"
    assert receipt["version_count"] == 2


def test_create_offline_ledger():
    """create_offline_ledger creates local ledger."""
    ledger = create_offline_ledger(node_id="colony_1")
    assert ledger["node_id"] == "colony_1"
    assert "entries" in ledger
    assert len(ledger["entries"]) == 0


def test_append_offline():
    """append_offline adds receipt to offline ledger."""
    ledger = create_offline_ledger(node_id="colony_1")
    receipt = {"receipt_id": "r1", "data": "test"}
    updated = append_offline(ledger, receipt)
    assert len(updated["entries"]) == 1


def test_merge_offline_ledger():
    """merge_offline_ledger combines ledgers."""
    ledger_a = {
        "node_id": "colony_1",
        "entries": [{"receipt_id": "r1", "data": "a"}],
    }
    ledger_b = {
        "node_id": "colony_2",
        "entries": [{"receipt_id": "r2", "data": "b"}],
    }
    merged = merge_offline_ledger([ledger_a, ledger_b])
    assert len(merged["entries"]) == 2


def test_emit_offline_ledger_receipt(capsys):
    """emit_offline_ledger_receipt emits valid receipt."""
    receipt = emit_offline_ledger_receipt(
        node_id="colony_1",
        entry_count=50,
        merkle_root="abc123:def456",
    )
    assert receipt["receipt_type"] == "offline_ledger"
    assert receipt["entry_count"] == 50


def test_merge_strategy_enum():
    """MergeStrategy enum has expected values."""
    assert MergeStrategy.TIMESTAMP is not None
    assert MergeStrategy.HASH_ORDER is not None
    assert MergeStrategy.PRIORITY is not None
