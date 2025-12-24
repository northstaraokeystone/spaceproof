"""SpaceProof Simulation - Monte Carlo Validation Framework.

The simulation modules implement multi-domain validation with:
- monte_carlo.py: Core Monte Carlo simulation engine
- scenarios/: Six validation scenarios (baseline, stress, genesis, singularity, thermodynamic, godel)
- dimensions/: D1-D20 validation dimension framework

Each domain configuration (xAI, DOGE, NASA, Defense, DOT) runs as an isolated
simulation that imports the shared engine core, applies its 3-module composition,
and produces domain-specific receipts.
"""

from spaceproof.sim.monte_carlo import (
    MonteCarloEngine,
    SimulationConfig,
    SimulationResult,
    Scenario,
    CheckpointConfig,
    run_simulation,
    run_domain_simulation,
)

__all__ = [
    "MonteCarloEngine",
    "SimulationConfig",
    "SimulationResult",
    "Scenario",
    "CheckpointConfig",
    "run_simulation",
    "run_domain_simulation",
]
