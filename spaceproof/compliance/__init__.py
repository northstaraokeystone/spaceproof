"""Compliance module - Audit trails and reports.

Generate compliance reports and audit trails.
Provides accountability documentation for enterprise governance.
"""

from .audit_trail import (
    generate_audit_trail,
    query_audit_trail,
    export_audit_trail,
    AuditTrailEntry,
    AuditTrailReport,
)

from .raci_report import (
    generate_raci_report,
    get_accountability_summary,
    RACIReport,
)

from .intervention_report import (
    generate_intervention_report,
    get_intervention_metrics,
    InterventionReport,
)

from .provenance_report import (
    generate_provenance_report,
    get_model_history,
    get_policy_history,
    ProvenanceReport,
)

__all__ = [
    # Audit trail
    "generate_audit_trail",
    "query_audit_trail",
    "export_audit_trail",
    "AuditTrailEntry",
    "AuditTrailReport",
    # RACI report
    "generate_raci_report",
    "get_accountability_summary",
    "RACIReport",
    # Intervention report
    "generate_intervention_report",
    "get_intervention_metrics",
    "InterventionReport",
    # Provenance report
    "generate_provenance_report",
    "get_model_history",
    "get_policy_history",
    "ProvenanceReport",
]
