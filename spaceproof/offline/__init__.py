"""Offline module - Light-delay tolerant synchronization.

Mars-scale synchronization with Byzantine-resilient conflict resolution.
"""

from .sync import (
    sync_ledger,
    queue_for_sync,
    get_sync_status,
    emit_sync_receipt,
    SyncResult,
    SyncQueue,
)

from .conflict_resolution import (
    resolve_conflicts,
    detect_conflicts,
    merge_receipts,
    emit_conflict_receipt,
    ConflictResult,
    MergeStrategy,
)

from .offline_ledger import (
    create_offline_ledger,
    append_offline,
    get_offline_entries,
    prepare_for_sync,
    OfflineLedger,
    OfflineEntry,
)

__all__ = [
    # Sync
    "sync_ledger",
    "queue_for_sync",
    "get_sync_status",
    "emit_sync_receipt",
    "SyncResult",
    "SyncQueue",
    # Conflict resolution
    "resolve_conflicts",
    "detect_conflicts",
    "merge_receipts",
    "emit_conflict_receipt",
    "ConflictResult",
    "MergeStrategy",
    # Offline ledger
    "create_offline_ledger",
    "append_offline",
    "get_offline_entries",
    "prepare_for_sync",
    "OfflineLedger",
    "OfflineEntry",
]
