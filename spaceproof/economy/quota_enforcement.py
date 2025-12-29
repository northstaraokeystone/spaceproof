"""quota_enforcement.py - Rate limiting via receipts.

Enforce quotas with receipt-based tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

ECONOMY_TENANT = "spaceproof-economy"


@dataclass
class QuotaConfig:
    """Quota configuration."""

    quota_id: str
    resource_type: str
    limit: int
    window_hours: int
    actor_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quota_id": self.quota_id,
            "resource_type": self.resource_type,
            "limit": self.limit,
            "window_hours": self.window_hours,
            "actor_id": self.actor_id,
            "created_at": self.created_at,
        }


@dataclass
class QuotaUsage:
    """Usage record for quota tracking."""

    usage_id: str
    quota_id: str
    amount: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "usage_id": self.usage_id,
            "quota_id": self.quota_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
        }


@dataclass
class QuotaStatus:
    """Current quota status."""

    quota_id: str
    resource_type: str
    limit: int
    used: int
    remaining: int
    window_start: str
    window_end: str
    is_exceeded: bool
    utilization_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quota_id": self.quota_id,
            "resource_type": self.resource_type,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "is_exceeded": self.is_exceeded,
            "utilization_pct": self.utilization_pct,
        }


# Storage
_quotas: Dict[str, QuotaConfig] = {}
_usage: Dict[str, List[QuotaUsage]] = {}


def create_quota(
    resource_type: str,
    limit: int,
    actor_id: str,
    window_hours: int = 24,
) -> QuotaConfig:
    """Create a quota configuration.

    Args:
        resource_type: Type of resource
        limit: Maximum allowed in window
        actor_id: Actor the quota applies to
        window_hours: Time window in hours

    Returns:
        QuotaConfig
    """
    quota = QuotaConfig(
        quota_id=str(uuid.uuid4()),
        resource_type=resource_type,
        limit=limit,
        window_hours=window_hours,
        actor_id=actor_id,
    )

    _quotas[quota.quota_id] = quota
    _usage[quota.quota_id] = []

    return quota


def get_quota(
    actor_id: str,
    resource_type: str,
) -> Optional[QuotaConfig]:
    """Get quota for actor and resource type.

    Args:
        actor_id: Actor identifier
        resource_type: Resource type

    Returns:
        QuotaConfig or None
    """
    for quota in _quotas.values():
        if quota.actor_id == actor_id and quota.resource_type == resource_type:
            return quota
    return None


def _get_window_usage(quota: QuotaConfig) -> int:
    """Get usage within current window.

    Args:
        quota: QuotaConfig

    Returns:
        Total usage in window
    """
    window_start = (datetime.utcnow() - timedelta(hours=quota.window_hours)).isoformat() + "Z"

    usage_list = _usage.get(quota.quota_id, [])
    return sum(u.amount for u in usage_list if u.timestamp >= window_start)


def check_quota(
    actor_id: str,
    resource_type: str,
    amount: int = 1,
) -> QuotaStatus:
    """Check quota status.

    Args:
        actor_id: Actor identifier
        resource_type: Resource type
        amount: Amount to check against

    Returns:
        QuotaStatus
    """
    quota = get_quota(actor_id, resource_type)

    if not quota:
        # No quota configured - allow unlimited
        return QuotaStatus(
            quota_id="unlimited",
            resource_type=resource_type,
            limit=-1,
            used=0,
            remaining=-1,
            window_start="",
            window_end="",
            is_exceeded=False,
            utilization_pct=0.0,
        )

    used = _get_window_usage(quota)
    remaining = max(0, quota.limit - used)
    is_exceeded = used + amount > quota.limit

    now = datetime.utcnow()
    window_start = (now - timedelta(hours=quota.window_hours)).isoformat() + "Z"
    window_end = now.isoformat() + "Z"

    return QuotaStatus(
        quota_id=quota.quota_id,
        resource_type=resource_type,
        limit=quota.limit,
        used=used,
        remaining=remaining,
        window_start=window_start,
        window_end=window_end,
        is_exceeded=is_exceeded,
        utilization_pct=(used / quota.limit * 100) if quota.limit > 0 else 0,
    )


def consume_quota(
    actor_id: str,
    resource_type: str,
    amount: int = 1,
) -> tuple[bool, QuotaStatus]:
    """Consume quota if available.

    Args:
        actor_id: Actor identifier
        resource_type: Resource type
        amount: Amount to consume

    Returns:
        Tuple of (success, QuotaStatus)
    """
    status = check_quota(actor_id, resource_type, amount)

    if status.is_exceeded:
        return False, status

    quota = get_quota(actor_id, resource_type)
    if quota:
        usage = QuotaUsage(
            usage_id=str(uuid.uuid4()),
            quota_id=quota.quota_id,
            amount=amount,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        _usage[quota.quota_id].append(usage)

        # Update status
        status.used += amount
        status.remaining = max(0, status.limit - status.used)
        status.utilization_pct = (status.used / status.limit * 100) if status.limit > 0 else 0

    return True, status


def reset_quota(
    actor_id: str,
    resource_type: str,
) -> bool:
    """Reset quota usage.

    Args:
        actor_id: Actor identifier
        resource_type: Resource type

    Returns:
        True if reset successful
    """
    quota = get_quota(actor_id, resource_type)
    if quota:
        _usage[quota.quota_id] = []
        return True
    return False


def emit_quota_receipt(status: QuotaStatus, action: str = "check") -> Dict[str, Any]:
    """Emit quota enforcement receipt.

    Args:
        status: QuotaStatus to emit
        action: Action type (check, consume, reset)

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "quota_enforcement",
        {
            "tenant_id": ECONOMY_TENANT,
            "action": action,
            **status.to_dict(),
        },
    )


def clear_quota_data() -> None:
    """Clear all quota data (for testing)."""
    global _quotas, _usage
    _quotas = {}
    _usage = {}
