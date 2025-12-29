"""escalation.py - Risk-based escalation routing.

Route decisions to appropriate authority based on risk level.
Implements escalation paths for CRITICAL, HIGH, MEDIUM, LOW decisions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt

from .raci import load_raci_matrix

# === CONSTANTS ===

GOVERNANCE_TENANT = "spaceproof-governance"

# Risk level thresholds
RISK_THRESHOLDS = {
    "critical": 0.9,  # >= 90% risk -> immediate escalation
    "high": 0.7,  # >= 70% risk -> team lead escalation
    "medium": 0.4,  # >= 40% risk -> monitoring
    "low": 0.0,  # < 40% risk -> normal operation
}


@dataclass
class EscalationResult:
    """Result of escalation decision."""

    escalation_id: str
    decision_id: str
    risk_level: str
    risk_score: float
    should_escalate: bool
    escalation_path: List[str]
    escalated_to: Optional[str]
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "escalation_id": self.escalation_id,
            "decision_id": self.decision_id,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "should_escalate": self.should_escalate,
            "escalation_path": self.escalation_path,
            "escalated_to": self.escalated_to,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


def calculate_risk_level(risk_score: float) -> str:
    """Calculate risk level from score.

    Args:
        risk_score: Risk score (0.0 - 1.0)

    Returns:
        Risk level string (critical, high, medium, low)
    """
    if risk_score >= RISK_THRESHOLDS["critical"]:
        return "critical"
    elif risk_score >= RISK_THRESHOLDS["high"]:
        return "high"
    elif risk_score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"


def get_escalation_path(risk_level: str) -> List[str]:
    """Get escalation path for risk level.

    Args:
        risk_level: Risk level (critical, high, medium, low)

    Returns:
        List of roles in escalation order
    """
    matrix = load_raci_matrix()
    chains = matrix.get("escalation_chains", {})
    return chains.get(risk_level, [])


def should_escalate(
    risk_score: float,
    event_type: str = "default",
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Determine if decision should be escalated.

    Args:
        risk_score: Computed risk score
        event_type: Type of event
        context: Optional context

    Returns:
        True if escalation required
    """
    risk_level = calculate_risk_level(risk_score)

    # Critical and high always escalate
    if risk_level in ["critical", "high"]:
        return True

    # Medium escalates based on context
    if risk_level == "medium":
        if context:
            # Escalate if any of these conditions
            if context.get("involves_human_safety", False):
                return True
            if context.get("regulatory_impact", False):
                return True
            if context.get("financial_impact", 0) > 100000:
                return True

    return False


def evaluate_escalation(
    decision_id: str,
    risk_score: float,
    event_type: str = "default",
    context: Optional[Dict[str, Any]] = None,
) -> EscalationResult:
    """Evaluate and create escalation decision.

    Args:
        decision_id: Decision identifier
        risk_score: Computed risk score
        event_type: Type of event
        context: Optional context

    Returns:
        EscalationResult with decision
    """
    risk_level = calculate_risk_level(risk_score)
    escalation_path = get_escalation_path(risk_level)
    needs_escalation = should_escalate(risk_score, event_type, context)

    escalated_to = escalation_path[0] if needs_escalation and escalation_path else None

    # Build reason
    if needs_escalation:
        if risk_level == "critical":
            reason = f"Critical risk level ({risk_score:.2f}) requires immediate escalation"
        elif risk_level == "high":
            reason = f"High risk level ({risk_score:.2f}) requires team lead escalation"
        else:
            context_reasons = []
            if context:
                if context.get("involves_human_safety"):
                    context_reasons.append("human safety")
                if context.get("regulatory_impact"):
                    context_reasons.append("regulatory impact")
                if context.get("financial_impact", 0) > 100000:
                    context_reasons.append("financial impact")
            reason = f"Escalation due to: {', '.join(context_reasons)}" if context_reasons else "Policy-based escalation"
    else:
        reason = f"No escalation required for {risk_level} risk level"

    return EscalationResult(
        escalation_id=str(uuid.uuid4()),
        decision_id=decision_id,
        risk_level=risk_level,
        risk_score=risk_score,
        should_escalate=needs_escalation,
        escalation_path=escalation_path,
        escalated_to=escalated_to,
        reason=reason,
    )


def emit_escalation_receipt(
    result: EscalationResult,
    additional_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit escalation receipt.

    Args:
        result: EscalationResult to emit
        additional_context: Optional additional context

    Returns:
        Receipt dict with dual-hash
    """
    receipt_data = {
        "tenant_id": GOVERNANCE_TENANT,
        **result.to_dict(),
    }

    if additional_context:
        receipt_data["additional_context"] = additional_context

    return emit_receipt("escalation", receipt_data)


def escalate_decision(
    decision_id: str,
    risk_score: float,
    event_type: str = "default",
    context: Optional[Dict[str, Any]] = None,
) -> EscalationResult:
    """Full escalation workflow: evaluate and emit receipt.

    Args:
        decision_id: Decision identifier
        risk_score: Computed risk score
        event_type: Type of event
        context: Optional context

    Returns:
        EscalationResult with receipt emitted
    """
    result = evaluate_escalation(decision_id, risk_score, event_type, context)
    emit_escalation_receipt(result, context)
    return result
