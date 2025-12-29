"""cost_accounting.py - Operation cost tracking.

Track computational and resource costs with receipt-based accounting.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

ECONOMY_TENANT = "spaceproof-economy"

# Default cost rates
DEFAULT_COST_RATES = {
    "compute": 0.01,  # per operation
    "storage": 0.001,  # per KB
    "network": 0.005,  # per KB transferred
    "inference": 0.10,  # per AI inference
    "encryption": 0.002,  # per operation
}


@dataclass
class OperationCost:
    """Cost record for an operation."""

    cost_id: str
    operation_type: str
    resource_units: float
    unit_cost: float
    total_cost: float
    actor_id: str
    budget_id: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cost_id": self.cost_id,
            "operation_type": self.operation_type,
            "resource_units": self.resource_units,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "actor_id": self.actor_id,
            "budget_id": self.budget_id,
            "timestamp": self.timestamp,
        }


@dataclass
class CostBudget:
    """Budget for cost tracking."""

    budget_id: str
    owner_id: str
    initial_amount: float
    remaining_amount: float
    operations_count: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "budget_id": self.budget_id,
            "owner_id": self.owner_id,
            "initial_amount": self.initial_amount,
            "remaining_amount": self.remaining_amount,
            "spent": self.initial_amount - self.remaining_amount,
            "operations_count": self.operations_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }

    def spend(self, amount: float) -> bool:
        """Spend from budget.

        Args:
            amount: Amount to spend

        Returns:
            True if budget available
        """
        if amount > self.remaining_amount:
            return False
        self.remaining_amount -= amount
        self.operations_count += 1
        self.last_used = datetime.utcnow().isoformat() + "Z"
        return True


# Storage
_budgets: Dict[str, CostBudget] = {}
_cost_history: List[OperationCost] = []


def allocate_budget(
    owner_id: str,
    amount: float,
) -> CostBudget:
    """Allocate a new cost budget.

    Args:
        owner_id: Budget owner
        amount: Initial amount

    Returns:
        New CostBudget
    """
    budget = CostBudget(
        budget_id=str(uuid.uuid4()),
        owner_id=owner_id,
        initial_amount=amount,
        remaining_amount=amount,
        operations_count=0,
    )

    _budgets[budget.budget_id] = budget

    return budget


def get_budget(budget_id: str) -> Optional[CostBudget]:
    """Get budget by ID.

    Args:
        budget_id: Budget identifier

    Returns:
        CostBudget or None
    """
    return _budgets.get(budget_id)


def track_operation_cost(
    operation_type: str,
    resource_units: float,
    actor_id: str,
    budget_id: Optional[str] = None,
    unit_cost: Optional[float] = None,
) -> OperationCost:
    """Track cost of an operation.

    Args:
        operation_type: Type of operation
        resource_units: Number of resource units consumed
        actor_id: Actor performing operation
        budget_id: Optional budget to charge
        unit_cost: Optional custom unit cost

    Returns:
        OperationCost record
    """
    # Get unit cost
    if unit_cost is None:
        unit_cost = DEFAULT_COST_RATES.get(operation_type, 0.01)

    total_cost = resource_units * unit_cost

    # Charge to budget if specified
    if budget_id:
        budget = _budgets.get(budget_id)
        if budget:
            budget.spend(total_cost)

    cost = OperationCost(
        cost_id=str(uuid.uuid4()),
        operation_type=operation_type,
        resource_units=resource_units,
        unit_cost=unit_cost,
        total_cost=total_cost,
        actor_id=actor_id,
        budget_id=budget_id,
    )

    _cost_history.append(cost)

    return cost


def get_cost_summary(
    actor_id: Optional[str] = None,
    budget_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get cost summary.

    Args:
        actor_id: Filter by actor
        budget_id: Filter by budget

    Returns:
        Summary statistics
    """
    costs = _cost_history

    if actor_id:
        costs = [c for c in costs if c.actor_id == actor_id]
    if budget_id:
        costs = [c for c in costs if c.budget_id == budget_id]

    if not costs:
        return {
            "total_cost": 0.0,
            "operation_count": 0,
            "by_type": {},
        }

    total = sum(c.total_cost for c in costs)

    by_type: Dict[str, float] = defaultdict(float)
    for c in costs:
        by_type[c.operation_type] += c.total_cost

    return {
        "total_cost": total,
        "operation_count": len(costs),
        "average_cost": total / len(costs),
        "by_type": dict(by_type),
    }


def emit_cost_receipt(cost: OperationCost) -> Dict[str, Any]:
    """Emit cost accounting receipt.

    Args:
        cost: OperationCost to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "cost_accounting",
        {
            "tenant_id": ECONOMY_TENANT,
            **cost.to_dict(),
        },
    )


def clear_cost_data() -> None:
    """Clear all cost data (for testing)."""
    global _budgets, _cost_history
    _budgets = {}
    _cost_history = []
