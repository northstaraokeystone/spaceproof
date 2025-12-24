"""SpaceProof Dimensions - D1-D20 Validation Maturity Framework.

The D-series dimensions structure validation maturity:

Foundation (D1-D5):
    Basic schema validation, bounds checking, simple entropy measurement.
    These emerge naturally from implementing ultimate dimensions.

Intermediate (D6-D10):
    Cross-module validation, pattern recognition, basic coherence.

Advanced (D11-D14):
    Complex pattern validation, multi-step verification, anomaly detection.

Ultimate (D15-D20):
    D15 - Autocatalysis: Self-sustaining receipt patterns (tau >= 0.7)
    D16 - Self-reproduction: System regenerates validation logic
    D17 - Thermodynamic: Energy/entropy conservation per cycle
    D18 - Singularity: Self-referential boundary conditions
    D19 - Decidability: Godel completeness bounds acknowledged
    D20 - Transcendence: Meta-system consistency across all dimensions

SpaceProof back-builds from D15-D20, following "receipts all the way down".
"""

from spaceproof.sim.dimensions.foundation import (
    D1_SchemaValidation,
    D2_BoundsCheck,
    D3_EntropyMeasurement,
    D4_TypeValidation,
    D5_FormatValidation,
)

from spaceproof.sim.dimensions.intermediate import (
    D6_CrossModuleValidation,
    D7_PatternRecognition,
    D8_TemporalConsistency,
    D9_BasicCoherence,
    D10_DependencyValidation,
)

from spaceproof.sim.dimensions.advanced import (
    D11_ComplexPatterns,
    D12_MultiStepVerification,
    D13_AnomalyDetection,
    D14_StatisticalValidation,
)

from spaceproof.sim.dimensions.ultimate import (
    D15_Autocatalysis,
    D16_SelfReproduction,
    D17_Thermodynamic,
    D18_Singularity,
    D19_Decidability,
    D20_Transcendence,
)

__all__ = [
    # Foundation
    "D1_SchemaValidation",
    "D2_BoundsCheck",
    "D3_EntropyMeasurement",
    "D4_TypeValidation",
    "D5_FormatValidation",
    # Intermediate
    "D6_CrossModuleValidation",
    "D7_PatternRecognition",
    "D8_TemporalConsistency",
    "D9_BasicCoherence",
    "D10_DependencyValidation",
    # Advanced
    "D11_ComplexPatterns",
    "D12_MultiStepVerification",
    "D13_AnomalyDetection",
    "D14_StatisticalValidation",
    # Ultimate
    "D15_Autocatalysis",
    "D16_SelfReproduction",
    "D17_Thermodynamic",
    "D18_Singularity",
    "D19_Decidability",
    "D20_Transcendence",
]
