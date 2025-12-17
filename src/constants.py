"""constants.py - Single Source of Truth for Physics Constants

ALL physics constants live here. No other module should define these.

CATEGORIES:
1. Shannon Physics - Entropy bounds and floor values
2. Alpha Metrics - Resilience calculation parameters
3. Cache/Overflow - GNN caching thresholds
4. Pruning - Merkle entropy pruning parameters
5. Blackout - Mars conjunction parameters
6. Validation - Stoprule thresholds

Source: Grok analysis + CLAUDEME S8 standards
"""

# =============================================================================
# SHANNON PHYSICS - Entropy Bounds
# =============================================================================

ENTROPY_ASYMPTOTE_E = 2.71828
"""physics: Shannon entropy bound ~e*ln(n). This is physics (~e), NOT tunable.
The value ~e appears because Merkle batch entropy bounds as ~e*ln(n).
GNN doesn't create this bound - it surfaces it by removing noise."""

SHANNON_FLOOR_ALPHA = 2.71828
"""physics: Baseline alpha without engineering = e (Shannon bound on resilience).
This is the FLOOR, not ceiling. Without engineering, alpha = e."""

ALPHA_CEILING_TARGET = 3.0
"""physics: e * max_factor where max_factor ~ 1.1 with full ML optimization.
Ceiling is ~3.0 with ML optimization."""

# =============================================================================
# ALPHA CALCULATION - Retention Factors
# =============================================================================

RETENTION_FACTOR_MAX = 1.10
"""physics: Ceiling / floor = 3.0 / 2.718 ~ 1.10. Theoretical max compounding."""

RETENTION_FACTOR_MIN = 0.95
"""physics: Minimum valid retention factor. Below this indicates bug."""

RETENTION_FACTOR_STOPRULE_MAX = 1.15
"""physics: StopRule if retention exceeds this. Unphysical value indicates bug."""

RETENTION_FACTOR_GNN_RANGE = (1.008, 1.015)
"""physics: Isolated GNN contribution per Grok ablation."""

RETENTION_FACTOR_PRUNE_RANGE = (1.008, 1.015)
"""physics: Isolated pruning contribution per Grok ablation."""

# =============================================================================
# ALPHA VALIDATION - Bounds and Thresholds
# =============================================================================

MIN_EFF_ALPHA_FLOOR = 2.63
"""physics: Minimum acceptable effective alpha before stoprule."""

MIN_EFF_ALPHA_VALIDATED = 2.7185
"""physics: Validated minimum effective alpha from 1000-run sweep at 90d."""

ALPHA_BELOW_FLOOR_THRESHOLD = 2.70
"""physics: StopRule triggers if alpha drops below this."""

ALPHA_ABOVE_CEILING_THRESHOLD = 3.1
"""physics: StopRule triggers if alpha exceeds this (ceiling + margin)."""

ABLATION_MODES = ["full", "no_cache", "no_prune", "baseline"]
"""physics: Four-mode isolation testing."""

ALPHA_FORMULA_VERSION = "v1.0"
"""Track formula evolution."""

# =============================================================================
# GNN CACHE - Depth and Overflow
# =============================================================================

CACHE_DEPTH_BASELINE = int(1e8)
"""physics: ~150d buffer at 50k entries/sol (10^8 entries)."""

CACHE_DEPTH_MIN = int(1e7)
"""physics: ~90d minimal coverage (10^7 entries)."""

CACHE_DEPTH_MAX = int(1e10)
"""physics: ~300d theoretical max (10^10 entries)."""

OVERFLOW_THRESHOLD_DAYS = 200
"""physics: Cache overflow stoprule trigger (without pruning)."""

OVERFLOW_THRESHOLD_DAYS_PRUNED = 300
"""physics: Cache overflow threshold with pruning enabled (~50% extension)."""

OVERFLOW_CAPACITY_PCT = 0.95
"""physics: Halt at 95% saturation."""

ENTRIES_PER_SOL = 50000
"""physics: Merkle batch scaling factor."""

# =============================================================================
# GNN NONLINEAR CURVE - Retention Modeling
# =============================================================================

ASYMPTOTE_ALPHA = 2.72
"""physics: e-like stability ceiling from GNN saturation."""

NONLINEAR_RETENTION_FLOOR = 1.25
"""physics: Asymptotic retention floor (better than linear at 90d)."""

RETENTION_BASE_FACTOR = 1.4
"""physics: Baseline retention factor at 43d blackout."""

CURVE_TYPE = "gnn_nonlinear"
"""physics: Model identifier - replaces linear."""

DECAY_LAMBDA = 0.003
"""physics: Exponential decay rate for GNN boost."""

SATURATION_KAPPA = 0.05
"""physics: Saturation coefficient for asymptotic behavior."""

# =============================================================================
# PRUNING - Merkle Entropy Parameters
# =============================================================================

PRUNING_TARGET_ALPHA = 2.80
"""physics: Target effective alpha with ln(n) compression via pruning."""

LN_N_TRIM_FACTOR_BASE = 0.3
"""physics: Conservative initial redundancy reduction (30% of ln(n))."""

LN_N_TRIM_FACTOR_MAX = 0.5
"""physics: Aggressive pruning ceiling (50% of ln(n))."""

OVER_PRUNE_STOPRULE_THRESHOLD = 0.6
"""physics: StopRule triggers if trim_factor exceeds this (too aggressive)."""

ENTROPY_PRUNE_THRESHOLD = 0.1
"""physics: Branches with entropy < 0.1*ln(n) are pruning candidates."""

DEDUP_PRIORITY = 1.0
"""physics: Deterministic dedup runs first (zero risk)."""

PREDICTIVE_PRIORITY = 0.7
"""physics: GNN-predicted pruning weighted lower (bounded uncertainty)."""

MIN_PROOF_PATHS_RETAINED = 3
"""physics: Safety - always keep at least 3 audit/proof paths."""

MIN_CONFIDENCE_THRESHOLD = 0.7
"""physics: StopRule on predictive phase if GNN confidence < 0.7."""

MIN_QUORUM_FRACTION = 2/3
"""physics: Quorum must be maintained at >= 2/3 nodes after pruning."""

DEDUP_RATIO_EXPECTED = 0.15
"""physics: Expected dedup ratio for typical Merkle batches (>=15%)."""

PREDICTIVE_ACCURACY_TARGET = 0.85
"""physics: Target GNN prediction accuracy for low-entropy branches."""

# =============================================================================
# BLACKOUT - Mars Conjunction Parameters
# =============================================================================

BLACKOUT_BASE_DAYS = 43
"""physics: Mars solar conjunction maximum duration in days."""

BLACKOUT_PRUNING_TARGET_DAYS = 250
"""physics: Extended survival target with entropy pruning enabled."""

BLACKOUT_MAX_UNREALISTIC = 120
"""physics: Unrealistic blackout duration stoprule threshold."""

QUORUM_FAIL_DAYS = 180
"""physics: Quorum degradation onset before cache overflow."""

CACHE_BREAK_DAYS = 200
"""physics: Cache overflow failure point."""

# =============================================================================
# REROUTING - Adaptive Reroute Parameters
# =============================================================================

REROUTING_ALPHA_BOOST_LOCKED = 0.07
"""physics: Validated rerouting alpha boost (immutable)."""

REROUTE_ALPHA_BOOST = REROUTING_ALPHA_BOOST_LOCKED
"""physics: Alias for rerouting boost."""

BLACKOUT_SWEEP_MAX_DAYS = 200
"""physics: Maximum blackout sweep range for reroute testing."""

# =============================================================================
# MARS COMMUNICATION - Physical Parameters
# =============================================================================

MARS_LIGHT_DELAY_MIN_S = 180
"""physics: Minimum one-way light delay to Mars in seconds (~3 min at opposition)."""

MARS_LIGHT_DELAY_MAX_S = 1320
"""physics: Maximum one-way light delay to Mars in seconds (~22 min at conjunction)."""

MARS_LIGHT_DELAY_AVG_S = 480
"""physics: Average one-way light delay to Mars in seconds (~8 min)."""

STARLINK_MARS_BANDWIDTH_MIN_MBPS = 2.0
"""physics: Minimum expected Starlink bandwidth for Mars."""

STARLINK_MARS_BANDWIDTH_MAX_MBPS = 10.0
"""physics: Maximum expected Starlink bandwidth for Mars."""

STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS = 5.0
"""physics: Expected Starlink bandwidth for Mars."""

TAU_REFERENCE_S = 300.0
"""physics: Reference tau baseline in seconds (5 minutes)."""

TAU_BASE_CURRENT_S = 300.0
"""physics: Current human-in-loop tau baseline."""

TAU_MIN_AUTONOMY_S = 30.0
"""physics: Minimum achievable tau with full autonomy."""

# =============================================================================
# VALIDATION - Stage Gate Parameters
# =============================================================================

STAGE_GATE_TRIGGER_ALPHA = 1.9
"""physics: Alpha threshold for stage gate trigger."""

ALPHA_CONFIDENCE_THRESHOLD = 0.70
"""physics: Confidence threshold for alpha-based decisions."""

# =============================================================================
# PARTITION - Node Quorum Parameters
# =============================================================================

NODE_BASELINE = 5
"""physics: Baseline node count for partition testing."""

QUORUM_THRESHOLD = 3
"""physics: Minimum nodes for quorum (3 of 5)."""

BASE_ALPHA = 2.68
"""physics: Base alpha for partition simulations."""

# =============================================================================
# SPEC FILE PATHS - Configuration Locations
# =============================================================================

ALPHA_FORMULA_SPEC_PATH = "data/alpha_formula_spec.json"
"""Path to alpha formula specification file."""

ENTROPY_PRUNING_SPEC_PATH = "data/entropy_pruning_spec.json"
"""Path to entropy pruning specification file."""

GNN_CACHE_SPEC_PATH = "data/gnn_cache_spec.json"
"""Path to GNN cache specification file."""

BLACKOUT_EXTENSION_SPEC_PATH = "data/blackout_extension_spec.json"
"""Path to blackout extension specification file."""

NODE_PARTITION_SPEC_PATH = "data/node_partition_spec.json"
"""Path to node partition specification file."""

REROUTE_BLACKOUT_SPEC_PATH = "data/reroute_blackout_spec.json"
"""Path to reroute blackout specification file."""
