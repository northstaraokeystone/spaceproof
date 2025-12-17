"""stoprules.py - Centralized StopRule Definitions

All stoprule logic lives here. Modules import and use these functions
rather than defining their own patterns.

PATTERN:
1. Check condition
2. Emit anomaly receipt if triggered
3. Raise StopRule with context

Source: CLAUDEME S4.7 pattern
"""

from typing import NoReturn, Optional
from .core import emit_receipt, StopRule
from .constants import (
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
    RETENTION_FACTOR_MIN,
    RETENTION_FACTOR_STOPRULE_MAX,
    OVERFLOW_CAPACITY_PCT,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_PROOF_PATHS_RETAINED,
    MIN_QUORUM_FRACTION,
    OVER_PRUNE_STOPRULE_THRESHOLD,
    BLACKOUT_MAX_UNREALISTIC,
)


# =============================================================================
# ALPHA STOPRULES
# =============================================================================

def stoprule_invalid_retention(factor: float, tenant_id: str = "axiom-alpha") -> None:
    """StopRule if retention factor is unphysical.

    Args:
        factor: Retention factor to validate
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If factor < 0.95 or > 1.15
    """
    if factor < RETENTION_FACTOR_MIN or factor > RETENTION_FACTOR_STOPRULE_MAX:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "retention_factor",
            "baseline": 1.0,
            "delta": factor - 1.0,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Invalid retention factor {factor:.4f}: "
            f"must be in range [{RETENTION_FACTOR_MIN}, {RETENTION_FACTOR_STOPRULE_MAX}]"
        )


def stoprule_alpha_below_floor(alpha: float, tenant_id: str = "axiom-alpha") -> None:
    """StopRule if alpha drops below Shannon floor.

    Args:
        alpha: Computed alpha to validate
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If alpha < 2.70
    """
    if alpha < 2.70:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "computed_alpha",
            "baseline": SHANNON_FLOOR_ALPHA,
            "delta": alpha - SHANNON_FLOOR_ALPHA,
            "classification": "deviation",
            "action": "investigate"
        })
        raise StopRule(
            f"Alpha {alpha:.4f} below Shannon floor {SHANNON_FLOOR_ALPHA:.4f}"
        )


def stoprule_alpha_above_ceiling(alpha: float, tenant_id: str = "axiom-alpha") -> None:
    """StopRule if alpha exceeds theoretical ceiling.

    Args:
        alpha: Computed alpha to validate
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If alpha > 3.1
    """
    ceiling_plus_margin = ALPHA_CEILING_TARGET + 0.1
    if alpha > ceiling_plus_margin:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "computed_alpha",
            "baseline": ALPHA_CEILING_TARGET,
            "delta": alpha - ALPHA_CEILING_TARGET,
            "classification": "deviation",
            "action": "investigate"
        })
        raise StopRule(
            f"Alpha {alpha:.4f} exceeds ceiling {ALPHA_CEILING_TARGET:.1f} + margin"
        )


# =============================================================================
# CACHE STOPRULES
# =============================================================================

def stoprule_cache_overflow(
    utilization: float,
    days: int,
    tenant_id: str = "axiom-cache"
) -> None:
    """StopRule if cache utilization exceeds threshold.

    Args:
        utilization: Current cache utilization (0-1)
        days: Current simulation day
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If utilization > OVERFLOW_CAPACITY_PCT
    """
    if utilization > OVERFLOW_CAPACITY_PCT:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "cache_overflow",
            "baseline": OVERFLOW_CAPACITY_PCT,
            "delta": utilization - OVERFLOW_CAPACITY_PCT,
            "days": days,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(f"Cache overflow at day {days}: {utilization:.2%} > {OVERFLOW_CAPACITY_PCT:.0%}")


def stoprule_cache_break(days: int, tenant_id: str = "axiom-cache") -> NoReturn:
    """StopRule for cache break condition.

    Args:
        days: Day when cache break occurred
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: Always raises
    """
    emit_receipt("anomaly", {
        "tenant_id": tenant_id,
        "metric": "cache_break",
        "baseline": 0,
        "delta": days,
        "classification": "violation",
        "action": "halt"
    })
    raise StopRule(f"Cache break at day {days}")


# =============================================================================
# PRUNING STOPRULES
# =============================================================================

def stoprule_over_prune(trim_factor: float, tenant_id: str = "axiom-pruning") -> None:
    """StopRule if pruning is too aggressive.

    Args:
        trim_factor: Current trim factor
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If trim_factor > OVER_PRUNE_STOPRULE_THRESHOLD
    """
    if trim_factor > OVER_PRUNE_STOPRULE_THRESHOLD:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "over_prune",
            "baseline": OVER_PRUNE_STOPRULE_THRESHOLD,
            "delta": trim_factor - OVER_PRUNE_STOPRULE_THRESHOLD,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Over-prune: trim_factor {trim_factor:.2f} > threshold {OVER_PRUNE_STOPRULE_THRESHOLD}"
        )


def stoprule_low_confidence(
    confidence: float,
    tenant_id: str = "axiom-pruning"
) -> None:
    """StopRule if GNN confidence is too low.

    Args:
        confidence: GNN prediction confidence
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If confidence < MIN_CONFIDENCE_THRESHOLD
    """
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "low_confidence",
            "baseline": MIN_CONFIDENCE_THRESHOLD,
            "delta": confidence - MIN_CONFIDENCE_THRESHOLD,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Low confidence {confidence:.3f} < threshold {MIN_CONFIDENCE_THRESHOLD}"
        )


def stoprule_chain_broken(
    proof_paths: int,
    has_root: bool,
    tenant_id: str = "axiom-pruning"
) -> None:
    """StopRule if chain integrity is compromised.

    Args:
        proof_paths: Number of retained proof paths
        has_root: Whether root hash exists
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If proof_paths < MIN_PROOF_PATHS_RETAINED or no root
    """
    if proof_paths < MIN_PROOF_PATHS_RETAINED:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "chain_broken",
            "baseline": MIN_PROOF_PATHS_RETAINED,
            "delta": proof_paths - MIN_PROOF_PATHS_RETAINED,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Chain broken: only {proof_paths} proof paths (need {MIN_PROOF_PATHS_RETAINED})"
        )

    if not has_root:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "chain_broken",
            "baseline": 1,
            "delta": -1,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule("Chain broken: missing root hash")


# =============================================================================
# QUORUM STOPRULES
# =============================================================================

def stoprule_quorum_failed(
    nodes_surviving: int,
    quorum_min: int,
    tenant_id: str = "axiom-partition"
) -> None:
    """StopRule if quorum is not maintained.

    Args:
        nodes_surviving: Number of surviving nodes
        quorum_min: Minimum required for quorum
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If nodes_surviving < quorum_min
    """
    if nodes_surviving < quorum_min:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "quorum_failed",
            "baseline": quorum_min,
            "delta": nodes_surviving - quorum_min,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Quorum failed: {nodes_surviving} nodes surviving < {quorum_min} threshold"
        )


# =============================================================================
# BLACKOUT STOPRULES
# =============================================================================

def stoprule_unrealistic_blackout(
    blackout_days: int,
    tenant_id: str = "axiom-blackout"
) -> None:
    """StopRule if blackout duration is unrealistic.

    Args:
        blackout_days: Blackout duration in days
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If blackout_days > BLACKOUT_MAX_UNREALISTIC
    """
    if blackout_days > BLACKOUT_MAX_UNREALISTIC:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "unrealistic_blackout",
            "baseline": BLACKOUT_MAX_UNREALISTIC,
            "delta": blackout_days - BLACKOUT_MAX_UNREALISTIC,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Blackout duration {blackout_days}d > {BLACKOUT_MAX_UNREALISTIC}d unrealistic limit"
        )


# =============================================================================
# HASH STOPRULES
# =============================================================================

def stoprule_hash_mismatch(
    expected: str,
    actual: str,
    context: str = "",
    tenant_id: str = "axiom-core"
) -> NoReturn:
    """StopRule for hash mismatch.

    Args:
        expected: Expected hash value
        actual: Actual hash value
        context: Additional context
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: Always raises
    """
    emit_receipt("anomaly", {
        "tenant_id": tenant_id,
        "metric": "hash_mismatch",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "expected": expected,
        "actual": actual,
        "context": context
    })
    raise StopRule(f"Hash mismatch{' (' + context + ')' if context else ''}: expected {expected}, got {actual}")


def stoprule_missing_hash(
    field_name: str,
    file_path: str,
    tenant_id: str = "axiom-core"
) -> NoReturn:
    """StopRule for missing hash field.

    Args:
        field_name: Name of missing field
        file_path: File where field was expected
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: Always raises
    """
    emit_receipt("anomaly", {
        "tenant_id": tenant_id,
        "metric": "missing_hash",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "field": field_name,
        "file": file_path
    })
    raise StopRule(f"{file_path} missing {field_name} field")


# =============================================================================
# DISPARITY STOPRULES
# =============================================================================

def stoprule_disparity_halt(
    disparity: float,
    threshold: float,
    metric_a: str,
    metric_b: str,
    tenant_id: str = "axiom-provenance"
) -> None:
    """StopRule if disparity exceeds threshold.

    Args:
        disparity: Computed disparity value
        threshold: Maximum allowed disparity
        metric_a: First metric name
        metric_b: Second metric name
        tenant_id: Tenant ID for receipt

    Raises:
        StopRule: If disparity > threshold
    """
    if disparity > threshold:
        emit_receipt("anomaly", {
            "tenant_id": tenant_id,
            "metric": "disparity_halt",
            "baseline": threshold,
            "delta": disparity - threshold,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Disparity {disparity:.4f} between {metric_a} and {metric_b} > {threshold} threshold"
        )
