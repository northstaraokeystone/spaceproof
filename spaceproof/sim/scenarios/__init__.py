"""SpaceProof Scenarios - Six-Scenario Validation Framework.

The six scenarios from the xAI collaboration probe different aspects of system behavior:

1. BASELINE: Normal operation with standard probability distributions
2. STRESS: Edge cases at 3-5x normal intensity with heavy-tail distributions
3. GENESIS: System initialization with bootstrap validation
4. SINGULARITY: Self-referential conditions where system audits itself
5. THERMODYNAMIC: Entropy conservation verification per second law
6. GODEL: Completeness bounds and decidability limits

Each scenario implements:
- Specific input distribution generators
- Custom checkpoint frequencies
- Scenario-specific validation criteria
- Receipt patterns appropriate to the scenario
"""

from spaceproof.sim.scenarios.baseline import BaselineScenario
from spaceproof.sim.scenarios.stress import StressScenario
from spaceproof.sim.scenarios.genesis import GenesisScenario
from spaceproof.sim.scenarios.singularity import SingularityScenario
from spaceproof.sim.scenarios.thermodynamic import ThermodynamicScenario
from spaceproof.sim.scenarios.godel import GodelScenario

__all__ = [
    "BaselineScenario",
    "StressScenario",
    "GenesisScenario",
    "SingularityScenario",
    "ThermodynamicScenario",
    "GodelScenario",
]
