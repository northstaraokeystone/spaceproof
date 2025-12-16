"""system.py - Multi-Body Solar System Simulation (v2.1)

THE xAI LOGISTICS EXTENSION:
    Neuralink boosts decision QUANTITY (100,000x bandwidth).
    xAI boosts decision QUALITY (1.5x decision value).
    Combined, they make small-scale sovereignty viable.

KEY INSIGHT (from Grok):
    The floor isn't technological—it's physical.
    You need 4 humans for 24/7 coverage no matter how augmented they are.
    Below 4, NO amount of tech helps.

Source: CLAUDEME.md (§8), Grok X Reply (Dec 16, 2025)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math

from .core import emit_receipt, TENANT_ID
from .entropy import (
    internal_compression_rate,
    effective_threshold,
    decision_capacity,
    earth_input_rate,
    sovereignty_threshold,
    xAI_LOGISTICS_MULTIPLIER,
    MINIMUM_VIABLE_CREW,
    MARS_RELAY_MBPS,
    LIGHT_DELAY_MAX,
    LIGHT_DELAY_MIN,
)


# === SYSTEM TENANT ===

SYSTEM_TENANT_ID = "axiom-system"
"""Receipt tenant for system simulation."""


# === DATACLASSES ===

@dataclass
class BodyState:
    """State of a single celestial body colony."""
    id: str
    delay_s: float  # Light delay in seconds
    bandwidth_share: float  # Share of relay bandwidth (0-1)
    relay_path: List[str]  # Path to Earth (e.g., ['earth'] or ['mars_relay', 'earth'])
    internal_rate: float  # Internal decision rate (bits/sec)
    external_rate: float  # External input rate from Earth (bits/sec)
    advantage: float  # internal_rate / external_rate
    sovereign: bool  # internal_rate > external_rate
    entropy: float  # Current entropy level
    status: str  # "nominal", "critical", "failed"
    crew: int = 10  # Number of crew
    neuralink_fraction: float = 0.0  # Fraction with Neuralink
    neuralink_bandwidth_mbps: float = 0.0  # Neural bandwidth per user
    compute_flops: float = 0.0  # Available compute


@dataclass
class SystemConfig:
    """Configuration for multi-body simulation.

    New in v2.1: xAI logistics integration.
    """
    duration_sols: int = 365
    bodies_enabled: List[str] = field(default_factory=lambda: ['earth', 'mars'])
    mars_crew: int = 10
    moon_crew: int = 0
    neuralink_enabled: bool = False
    neuralink_fraction: float = 1.0  # Fraction of crew with Neuralink
    neuralink_bandwidth_mbps: float = 1.0  # 1 Mbps per Neuralink user
    xai_enabled: bool = False
    xai_logistics_mode: str = "both"  # "queue" | "decisions" | "both"
    compute_flops: float = 1e15  # 1 PFLOP default
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Result of a multi-body simulation."""
    bodies: Dict[str, BodyState]
    duration_sols: int
    receipts: List[Dict]
    queue: List[Dict]
    final_entropy: float


# === CORE FUNCTIONS ===

def check_minimum_viable(body_state: BodyState) -> bool:
    """Check if body meets minimum viable crew requirement.

    Args:
        body_state: The body state to check.

    Returns:
        False if crew < MINIMUM_VIABLE_CREW, True otherwise.

    Note:
        Below MINIMUM_VIABLE_CREW (4), NO amount of tech helps.
        This is the physics floor for 24/7 coverage.
    """
    return body_state.crew >= MINIMUM_VIABLE_CREW


def optimize_queue_xai(queue: List[Dict], bodies: Dict[str, BodyState]) -> List[Dict]:
    """Optimize mission queue using xAI logistics.

    If xai_enabled, reorder queue for optimal entropy flow:
    1. Prioritize bodies closest to sovereignty threshold
    2. Reduce queue entropy by batching related missions

    Args:
        queue: List of queued missions/tasks.
        bodies: Dict of body states.

    Returns:
        Optimized queue (reordered).

    Emits:
        xai_optimization_receipt when queue is reordered.
    """
    if not queue:
        return queue

    # Sort by sovereignty advantage (closest to threshold first)
    def priority_key(mission: Dict) -> float:
        body_id = mission.get("target_body", "mars")
        body = bodies.get(body_id)
        if body:
            # Priority = how close to sovereignty (1.0 = exactly at threshold)
            return abs(1.0 - body.advantage)
        return float('inf')

    original_order = [m.get("id", i) for i, m in enumerate(queue)]
    optimized = sorted(queue, key=priority_key)
    new_order = [m.get("id", i) for i, m in enumerate(optimized)]

    # Only emit receipt if order changed
    if original_order != new_order:
        emit_receipt("xai_optimization", {
            "tenant_id": SYSTEM_TENANT_ID,
            "original_order": original_order,
            "optimized_order": new_order,
            "optimization_type": "sovereignty_priority",
        })

    return optimized


def create_body_state(
    body_id: str,
    crew: int,
    config: SystemConfig
) -> BodyState:
    """Create a body state with computed rates.

    Args:
        body_id: Identifier for the body (e.g., "mars", "moon").
        crew: Number of crew on this body.
        config: System configuration.

    Returns:
        Initialized BodyState.
    """
    # Light delay based on body
    if body_id == "earth":
        delay_s = 0
        bandwidth = MARS_RELAY_MBPS * 10  # Earth has more bandwidth
    elif body_id == "mars":
        delay_s = LIGHT_DELAY_MAX * 60  # Worst case: 22 minutes
        bandwidth = MARS_RELAY_MBPS
    elif body_id == "moon":
        delay_s = 1.3  # ~1.3 second light delay
        bandwidth = MARS_RELAY_MBPS * 5
    else:
        delay_s = LIGHT_DELAY_MAX * 60
        bandwidth = MARS_RELAY_MBPS

    # Neuralink parameters
    if config.neuralink_enabled:
        neuralink_fraction = config.neuralink_fraction
        neuralink_bw = config.neuralink_bandwidth_mbps
    else:
        neuralink_fraction = 0.0
        neuralink_bw = 0.0

    # Compute internal rate using internal_compression_rate
    internal_rate = internal_compression_rate(
        crew=crew,
        compute_flops=config.compute_flops,
        neuralink_fraction=neuralink_fraction,
        neuralink_bandwidth_mbps=neuralink_bw,
        xai_enabled=config.xai_enabled
    )

    # Compute external rate
    external_rate = earth_input_rate(bandwidth, delay_s)

    # Compute advantage and sovereignty
    if external_rate > 0:
        advantage = internal_rate / external_rate
    else:
        advantage = float('inf') if internal_rate > 0 else 0

    # Sovereignty check with physics floor enforcement
    # Below MINIMUM_VIABLE_CREW (4), NO amount of tech helps
    if crew < MINIMUM_VIABLE_CREW:
        sovereign = False  # Physics floor: 24/7 coverage requires 4 minimum
    else:
        # Also check against effective_threshold based on tech enablement
        from .entropy import effective_threshold
        required_crew = effective_threshold(config.neuralink_enabled, config.xai_enabled)
        sovereign = crew >= max(required_crew, MINIMUM_VIABLE_CREW)

    return BodyState(
        id=body_id,
        delay_s=delay_s,
        bandwidth_share=bandwidth / (MARS_RELAY_MBPS * 10),
        relay_path=['earth'] if body_id != 'earth' else [],
        internal_rate=internal_rate,
        external_rate=external_rate,
        advantage=advantage,
        sovereign=sovereign,
        entropy=100.0,  # Initial entropy
        status="nominal",
        crew=crew,
        neuralink_fraction=neuralink_fraction,
        neuralink_bandwidth_mbps=neuralink_bw,
        compute_flops=config.compute_flops,
    )


def evolve_bodies(
    bodies: Dict[str, BodyState],
    config: SystemConfig,
    sol: int
) -> Dict[str, BodyState]:
    """Evolve body states for one sol.

    Updates entropy based on internal/external rates.
    Passes xai_enabled to internal_compression_rate calculation.

    Args:
        bodies: Current body states.
        config: System configuration.
        sol: Current sol number.

    Returns:
        Updated body states.
    """
    updated = {}

    for body_id, body in bodies.items():
        if body_id == 'earth':
            # Earth doesn't need sovereignty
            updated[body_id] = body
            continue

        # Recalculate rates (may change with conditions)
        internal_rate = internal_compression_rate(
            crew=body.crew,
            compute_flops=body.compute_flops,
            neuralink_fraction=body.neuralink_fraction,
            neuralink_bandwidth_mbps=body.neuralink_bandwidth_mbps,
            xai_enabled=config.xai_enabled
        )

        external_rate = earth_input_rate(
            MARS_RELAY_MBPS * body.bandwidth_share * 10,
            body.delay_s
        )

        # Update entropy: decreases if internal > external (shedding)
        entropy_delta = (external_rate - internal_rate) * 0.01
        new_entropy = max(0, body.entropy + entropy_delta)

        # Update advantage and sovereignty
        if external_rate > 0:
            advantage = internal_rate / external_rate
        else:
            advantage = float('inf') if internal_rate > 0 else 0

        # Sovereignty check with physics floor enforcement
        # Below MINIMUM_VIABLE_CREW (4), NO amount of tech helps
        if body.crew < MINIMUM_VIABLE_CREW:
            sovereign = False  # Physics floor: 24/7 coverage requires 4 minimum
        else:
            # Use effective_threshold based on tech enablement
            required_crew = effective_threshold(config.neuralink_enabled, config.xai_enabled)
            sovereign = body.crew >= max(required_crew, MINIMUM_VIABLE_CREW)

        updated[body_id] = BodyState(
            id=body.id,
            delay_s=body.delay_s,
            bandwidth_share=body.bandwidth_share,
            relay_path=body.relay_path,
            internal_rate=internal_rate,
            external_rate=external_rate,
            advantage=advantage,
            sovereign=sovereign,
            entropy=new_entropy,
            status=body.status,
            crew=body.crew,
            neuralink_fraction=body.neuralink_fraction,
            neuralink_bandwidth_mbps=body.neuralink_bandwidth_mbps,
            compute_flops=body.compute_flops,
        )

    return updated


def tick(
    bodies: Dict[str, BodyState],
    config: SystemConfig,
    sol: int,
    queue: List[Dict]
) -> tuple:
    """Execute one simulation tick.

    Includes minimum viable crew check.

    Args:
        bodies: Current body states.
        config: System configuration.
        sol: Current sol number.
        queue: Mission queue.

    Returns:
        Tuple of (updated_bodies, updated_queue, receipts).
    """
    receipts = []

    # Minimum viable crew check
    for body_id, body in bodies.items():
        if body_id == 'earth':
            continue

        if not check_minimum_viable(body):
            # Emit minimum_crew_violation_receipt
            receipt = emit_receipt("minimum_crew_violation", {
                "tenant_id": SYSTEM_TENANT_ID,
                "body_id": body_id,
                "crew": body.crew,
                "minimum_required": MINIMUM_VIABLE_CREW,
                "reason": "24/7 coverage requires minimum 2 shifts × 2 people",
                "sol": sol,
            })
            receipts.append(receipt)

            # Set body status to critical
            body.status = "critical"
            body.sovereign = False  # Cannot achieve sovereignty regardless of tech

    # Evolve bodies
    bodies = evolve_bodies(bodies, config, sol)

    # Optimize queue if xAI enabled
    if config.xai_enabled and config.xai_logistics_mode in ("queue", "both"):
        queue = optimize_queue_xai(queue, bodies)

    return bodies, queue, receipts


def run_simulation(config: SystemConfig) -> SimulationResult:
    """Run full multi-body simulation.

    MAIN ENTRY POINT for system simulation.

    Args:
        config: System configuration.

    Returns:
        SimulationResult with body states and receipts.
    """
    # Initialize bodies
    bodies = {}

    if 'earth' in config.bodies_enabled:
        bodies['earth'] = create_body_state('earth', 0, config)

    if 'mars' in config.bodies_enabled:
        bodies['mars'] = create_body_state('mars', config.mars_crew, config)

    if 'moon' in config.bodies_enabled:
        bodies['moon'] = create_body_state('moon', config.moon_crew, config)

    # Initialize queue and receipts
    queue: List[Dict] = []
    all_receipts: List[Dict] = []

    # Run simulation
    for sol in range(config.duration_sols):
        bodies, queue, tick_receipts = tick(bodies, config, sol, queue)
        all_receipts.extend(tick_receipts)

    # Calculate final entropy
    final_entropy = sum(b.entropy for b in bodies.values() if b.id != 'earth')

    # Emit threshold_reduction receipt if applicable
    if config.neuralink_enabled:
        baseline = effective_threshold(False, False)
        enhanced = effective_threshold(config.neuralink_enabled, config.xai_enabled)
        reduction_pct = (1 - enhanced / baseline) * 100

        emit_receipt("threshold_reduction", {
            "tenant_id": SYSTEM_TENANT_ID,
            "baseline_threshold": baseline,
            "enhanced_threshold": enhanced,
            "reduction_percent": reduction_pct,
            "neuralink_enabled": config.neuralink_enabled,
            "xai_enabled": config.xai_enabled,
        })

    return SimulationResult(
        bodies=bodies,
        duration_sols=config.duration_sols,
        receipts=all_receipts,
        queue=queue,
        final_entropy=final_entropy,
    )
