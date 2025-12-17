"""alpha_compute.py - Explicit α Formula Computation Module

THE PHYSICS (from Grok clarification):
    - α and H measure different things
    - H ≤ e·ln(n) is Shannon entropy bound (invariant)
    - α = (min_eff / baseline) * retention_factor is engineered resilience
    - e is FLOOR (baseline), not ceiling
    - Ceiling is ~3.0 with ML optimization

KEY FORMULA:
    α = (min_eff / baseline) * retention_factor

    Where:
    - min_eff: minimum effective redundancy (from GNN asymptote, typically ~2.718)
    - baseline: normalized to 1.0
    - retention_factor: product of all layer contributions (GNN, pruning, etc.)
    - retention_factor > 1.0 means engineering adds resilience above Shannon floor

Source: Grok - "α and H measure different things... retention_factor >1 from GNN boosts"
"""

import json
import os
from functools import reduce
from typing import Dict, Any, List

from .core import emit_receipt, dual_hash

# Import all constants from centralized location
from .constants import (
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
    RETENTION_FACTOR_MAX,
    RETENTION_FACTOR_MIN,
    RETENTION_FACTOR_STOPRULE_MAX,
    RETENTION_FACTOR_GNN_RANGE,
    RETENTION_FACTOR_PRUNE_RANGE,
    ABLATION_MODES,
    ALPHA_FORMULA_VERSION,
    ALPHA_FORMULA_SPEC_PATH,
)

# Import stoprules from centralized location
from .stoprules import (
    stoprule_invalid_retention,
    stoprule_alpha_below_floor,
    stoprule_alpha_above_ceiling,
)


def load_alpha_formula_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify alpha formula specification file.

    Args:
        path: Optional path override (default: ALPHA_FORMULA_SPEC_PATH)

    Returns:
        Dict containing alpha formula specification

    Receipt: alpha_formula_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, ALPHA_FORMULA_SPEC_PATH)

    with open(path, 'r') as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt("alpha_formula_spec_ingest", {
        "tenant_id": "axiom-alpha-compute",
        "file_path": path,
        "formula_version": data["formula_version"],
        "formula": data["formula"],
        "shannon_floor_alpha": data["constants"]["shannon_floor_alpha"],
        "alpha_ceiling_target": data["constants"]["alpha_ceiling_target"],
        "ablation_modes": data["ablation_modes"],
        "payload_hash": content_hash
    })

    return data


def alpha_calc(
    min_eff: float,
    baseline: float,
    retention_factor: float,
    validate: bool = True
) -> Dict[str, Any]:
    """Compute alpha using explicit formula.

    THE FORMULA: α = (min_eff / baseline) * retention_factor

    Args:
        min_eff: Minimum effective redundancy (from GNN asymptote, typically ~2.718)
        baseline: Normalized baseline (typically 1.0)
        retention_factor: Product of all layer contributions (GNN, pruning, etc.)
        validate: Whether to apply stoprules (default: True)

    Returns:
        Dict with computed_alpha, formula_used, components, references, gap_to_ceiling

    Raises:
        StopRule: If retention_factor is invalid or alpha is out of bounds

    Receipt: alpha_formula
    """
    if validate:
        stoprule_invalid_retention(retention_factor, "axiom-alpha-compute")

    # THE FORMULA
    computed_alpha = (min_eff / baseline) * retention_factor
    computed_alpha = round(computed_alpha, 5)

    if validate:
        stoprule_alpha_below_floor(computed_alpha, "axiom-alpha-compute")
        stoprule_alpha_above_ceiling(computed_alpha, "axiom-alpha-compute")

    gap_absolute = ALPHA_CEILING_TARGET - computed_alpha
    gap_pct = (gap_absolute / ALPHA_CEILING_TARGET) * 100

    result = {
        "computed_alpha": computed_alpha,
        "formula_used": f"({min_eff} / {baseline}) * {retention_factor}",
        "components": {
            "min_eff": min_eff,
            "baseline": baseline,
            "retention_factor": retention_factor
        },
        "floor_reference": SHANNON_FLOOR_ALPHA,
        "ceiling_reference": ALPHA_CEILING_TARGET,
        "gap_to_ceiling_absolute": round(gap_absolute, 4),
        "gap_to_ceiling_pct": round(gap_pct, 2),
        "formula_version": ALPHA_FORMULA_VERSION
    }

    emit_receipt("alpha_formula", {
        "receipt_type": "alpha_formula",
        "tenant_id": "axiom-alpha-compute",
        "min_eff": min_eff,
        "baseline": baseline,
        "retention_factor": retention_factor,
        "computed_alpha": computed_alpha,
        "formula_version": ALPHA_FORMULA_VERSION,
        "floor_reference": SHANNON_FLOOR_ALPHA,
        "ceiling_reference": ALPHA_CEILING_TARGET,
        "gap_to_ceiling_pct": round(gap_pct, 2),
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def compound_retention(factors: List[float]) -> float:
    """Compute compound retention factor from multiple layers.

    Multiplicative: Π(factors). E.g., [1.01, 1.02] → 1.0302

    Args:
        factors: List of retention factors from each layer

    Returns:
        Compound retention factor
    """
    if not factors:
        return 1.0
    compound = reduce(lambda x, y: x * y, factors, 1.0)
    return round(compound, 5)


def isolate_layer_contribution(
    full_alpha: float,
    ablated_alpha: float,
    floor: float = SHANNON_FLOOR_ALPHA
) -> float:
    """Compute isolated contribution of a layer.

    Formula: contribution_factor = (full - ablated) / (full - floor)

    Args:
        full_alpha: Alpha with all layers active
        ablated_alpha: Alpha with one layer disabled
        floor: Shannon floor (default: e)

    Returns:
        Contribution factor (0-1 range, where 1 = 100% contribution)
    """
    if full_alpha <= floor:
        return 0.0
    contribution = (full_alpha - ablated_alpha) / (full_alpha - floor)
    return round(max(0.0, min(1.0, contribution)), 4)


def ceiling_gap(
    current_alpha: float,
    ceiling_target: float = ALPHA_CEILING_TARGET
) -> Dict[str, Any]:
    """Track progress toward ceiling target.

    Args:
        current_alpha: Current alpha value
        ceiling_target: Target ceiling (default: 3.0)

    Returns:
        Dict with gap metrics and path description

    Receipt: ceiling_track
    """
    gap_absolute = ceiling_target - current_alpha
    gap_pct = (gap_absolute / ceiling_target) * 100
    retention_current = current_alpha / SHANNON_FLOOR_ALPHA
    retention_needed = ceiling_target / SHANNON_FLOOR_ALPHA
    retention_delta = retention_needed - retention_current

    if gap_pct <= 0:
        path = "Ceiling reached"
    elif gap_pct <= 5:
        path = "Near ceiling - minor optimization needed"
    elif gap_pct <= 10:
        path = "Close to ceiling - moderate optimization needed"
    else:
        path = f"Far from ceiling - {gap_pct:.1f}% optimization needed"

    result = {
        "current_alpha": current_alpha,
        "ceiling_target": ceiling_target,
        "gap_absolute": round(gap_absolute, 4),
        "gap_pct": round(gap_pct, 2),
        "retention_factor_current": round(retention_current, 4),
        "retention_factor_needed": round(retention_needed, 4),
        "retention_factor_delta": round(retention_delta, 4),
        "path_to_ceiling": path
    }

    emit_receipt("ceiling_track", {
        "receipt_type": "ceiling_track",
        "tenant_id": "axiom-alpha-compute",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def validate_formula(
    min_eff: float,
    retention: float,
    expected: float,
    tolerance: float = 0.01
) -> bool:
    """Validate formula correctness within tolerance."""
    result = alpha_calc(min_eff, 1.0, retention, validate=False)
    computed = result["computed_alpha"]
    return abs(computed - expected) <= tolerance


def get_ablation_expected(mode: str) -> Dict[str, Any]:
    """Get expected results for an ablation mode."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}. Valid: {ABLATION_MODES}")

    expected = {
        "baseline": {
            "alpha_range": (2.71, 2.72),
            "retention": 1.0,
            "description": "No engineering - Shannon floor"
        },
        "no_cache": {
            "alpha_range": (2.76, 2.80),
            "retention": (1.015, 1.03),
            "description": "Pruning only - no GNN caching"
        },
        "no_prune": {
            "alpha_range": (2.72, 2.74),
            "retention": (1.003, 1.01),
            "description": "GNN only - no pruning"
        },
        "full": {
            "alpha_range": (2.80, 2.85),
            "retention": (1.03, 1.05),
            "description": "Full stack - GNN + pruning"
        }
    }

    return expected[mode]


def compute_alpha_from_layers(
    gnn_retention: float = 1.0,
    prune_retention: float = 1.0,
    base_min_eff: float = SHANNON_FLOOR_ALPHA,
    ablation_mode: str = "full"
) -> Dict[str, Any]:
    """Compute alpha from individual layer retention factors.

    Args:
        gnn_retention: GNN layer retention factor (default: 1.0)
        prune_retention: Pruning layer retention factor (default: 1.0)
        base_min_eff: Base minimum efficiency (default: e)
        ablation_mode: Ablation mode (default: "full")

    Returns:
        Dict with computed_alpha, layer_contributions, ablation_mode

    Receipt: alpha_from_layers
    """
    if ablation_mode == "baseline":
        gnn_active = 1.0
        prune_active = 1.0
    elif ablation_mode == "no_cache":
        gnn_active = 1.0
        prune_active = prune_retention
    elif ablation_mode == "no_prune":
        gnn_active = gnn_retention
        prune_active = 1.0
    else:  # full
        gnn_active = gnn_retention
        prune_active = prune_retention

    compound = compound_retention([gnn_active, prune_active])
    alpha_result = alpha_calc(base_min_eff, 1.0, compound)

    result = {
        "computed_alpha": alpha_result["computed_alpha"],
        "compound_retention": compound,
        "layer_contributions": {
            "gnn_retention": gnn_retention,
            "gnn_active": gnn_active,
            "prune_retention": prune_retention,
            "prune_active": prune_active
        },
        "ablation_mode": ablation_mode,
        "base_min_eff": base_min_eff,
        "gap_to_ceiling_pct": alpha_result["gap_to_ceiling_pct"]
    }

    emit_receipt("alpha_from_layers", {
        "receipt_type": "alpha_from_layers",
        "tenant_id": "axiom-alpha-compute",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def get_alpha_compute_info() -> Dict[str, Any]:
    """Get alpha compute module configuration info.

    Receipt: alpha_compute_info
    """
    info = {
        "shannon_floor_alpha": SHANNON_FLOOR_ALPHA,
        "alpha_ceiling_target": ALPHA_CEILING_TARGET,
        "retention_factor_max": RETENTION_FACTOR_MAX,
        "retention_factor_min": RETENTION_FACTOR_MIN,
        "retention_factor_gnn_range": RETENTION_FACTOR_GNN_RANGE,
        "retention_factor_prune_range": RETENTION_FACTOR_PRUNE_RANGE,
        "ablation_modes": ABLATION_MODES,
        "alpha_formula_version": ALPHA_FORMULA_VERSION,
        "formula": "alpha = (min_eff / baseline) * retention_factor",
        "physics_clarification": {
            "alpha_vs_entropy": "α and H measure different things",
            "shannon_bound": "H ≤ e·ln(n) is Shannon entropy bound",
            "alpha_definition": "α is engineered resilience metric",
            "floor_is_e": "e is the FLOOR (baseline), not ceiling",
            "ceiling_is_3": "Ceiling is ~3.0 with ML optimization"
        },
        "description": "Explicit α formula computation module."
    }

    emit_receipt("alpha_compute_info", {
        "tenant_id": "axiom-alpha-compute",
        **info,
        "payload_hash": dual_hash(json.dumps(info, sort_keys=True))
    })

    return info
