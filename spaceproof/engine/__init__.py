"""SpaceProof Engine - Entropy-based validation infrastructure.

The engine modules implement the entropy pump paradigm from the xAI collaboration:
- entropy.py: Shannon entropy, coherence scoring, entropy delta health
- receipts.py: IETF COSE Merkle Tree Proofs, receipt chaining
- gates.py: Validation gate orchestration
- saga.py: Compensation/rollback patterns

The fundamental insight: QED is an entropy pump.
Telemetry enters with high entropy, compression happens, decisions emerge with low entropy.
"""

from spaceproof.engine.entropy import (
    shannon_entropy,
    entropy_delta,
    coherence_score,
    compression_ratio,
    kolmogorov_entropy,
    fitness_score,
    ThompsonState,
    thompson_select,
    COHERENCE_THRESHOLD,
    ENTROPY_DELTA_HEALTHY,
    ENTROPY_DELTA_WARNING,
    ENTROPY_DELTA_CRITICAL,
    COMPRESSION_BASELINE_MIN,
    COMPRESSION_BASELINE_MAX,
    FRAUD_SIGNAL_THRESHOLD,
    RANDOM_SIGNAL_THRESHOLD,
    SURVIVAL_PERCENTILE,
)

from spaceproof.engine.receipts import (
    SpaceProofReceipt,
    DomainConfig,
    ModuleAttestation,
    create_receipt,
    chain_receipts,
    verify_chain,
    build_xai_receipt,
    build_doge_receipt,
    build_nasa_receipt,
    build_defense_receipt,
    build_dot_receipt,
)

from spaceproof.engine.gates import (
    Gate,
    GateStatus,
    GateType,
    GateResult,
    GateContext,
    GateOrchestrator,
    PreValidationGate,
    EntropyMeasurementGate,
    ModuleGate,
    CompressionGate,
    CoherenceGate,
    MerkleGate,
    SignatureGate,
    PostValidationGate,
    create_standard_gate_sequence,
)

from spaceproof.engine.saga import (
    Saga,
    SagaStep,
    SagaStatus,
    SagaResult,
    SagaBuilder,
    FunctionStep,
    create_validation_saga,
    create_entropy_pump_saga,
)

__all__ = [
    # Entropy
    "shannon_entropy",
    "entropy_delta",
    "coherence_score",
    "compression_ratio",
    "kolmogorov_entropy",
    "fitness_score",
    "ThompsonState",
    "thompson_select",
    "COHERENCE_THRESHOLD",
    "ENTROPY_DELTA_HEALTHY",
    "ENTROPY_DELTA_WARNING",
    "ENTROPY_DELTA_CRITICAL",
    "COMPRESSION_BASELINE_MIN",
    "COMPRESSION_BASELINE_MAX",
    "FRAUD_SIGNAL_THRESHOLD",
    "RANDOM_SIGNAL_THRESHOLD",
    "SURVIVAL_PERCENTILE",
    # Receipts
    "SpaceProofReceipt",
    "DomainConfig",
    "ModuleAttestation",
    "create_receipt",
    "chain_receipts",
    "verify_chain",
    "build_xai_receipt",
    "build_doge_receipt",
    "build_nasa_receipt",
    "build_defense_receipt",
    "build_dot_receipt",
    # Gates
    "Gate",
    "GateStatus",
    "GateType",
    "GateResult",
    "GateContext",
    "GateOrchestrator",
    "PreValidationGate",
    "EntropyMeasurementGate",
    "ModuleGate",
    "CompressionGate",
    "CoherenceGate",
    "MerkleGate",
    "SignatureGate",
    "PostValidationGate",
    "create_standard_gate_sequence",
    # Saga
    "Saga",
    "SagaStep",
    "SagaStatus",
    "SagaResult",
    "SagaBuilder",
    "FunctionStep",
    "create_validation_saga",
    "create_entropy_pump_saga",
]
