"""topology.py - Persistent Homology for DM-as-Geometry

THE HYPOTHESIS:
  What if dark matter isn't particles?
  What if it's topology—curved spacetime without mass?

  Persistence diagrams encode galactic structure in 20-50 bits.
  NFW profiles need 150 bits.
  If topology compresses better, geometry wins.

H1 loops = halo cores (closed structures in the rotation curve residuals)
H2 voids = dark energy fingerprints (2D "holes" in higher-dimensional embedding)

If the residuals from a KAN fit show persistent topological features that
correlate with physics regime, topology is carrying information about dark matter.

Source: CLAUDEME.md (§0, §8)
"""

import time
from typing import Optional

import numpy as np
from ripser import ripser
from scipy.optimize import linear_sum_assignment

from .core import dual_hash, emit_receipt, StopRule


# === CONSTANTS (Module Top) ===

TENANT_ID = "axiom-witness"
"""CLAUDEME tenant isolation."""

PERSISTENCE_NOISE_FLOOR = 0.012
"""Features below this persistence are noise (Grok threshold)."""

DM_COMPLEXITY_THRESHOLD = 48
"""Bits threshold for dark matter flag."""

DM_H1_MINIMUM = 2
"""Minimum H1 features for DM classification."""

HALO_PERSISTENCE_LOW = 0.04
"""Lower bound for halo H1 persistence."""

HALO_PERSISTENCE_HIGH = 0.08
"""Upper bound for halo H1 persistence."""

DE_H2_THRESHOLD = 3
"""H2 count threshold for dark energy signature."""

MAX_DIMENSION = 2
"""Compute H0, H1, H2."""

MAX_COMPLEXITY_BITS = 65
"""SLO ceiling for topology bits (Scenario 4)."""

COMPUTATION_TIMEOUT = 5.0
"""Maximum seconds for persistence computation."""


# === UTILITY FUNCTIONS ===

def time_delay_embed(signal: np.ndarray, embed_dim: int = 3, delay: int = 1) -> np.ndarray:
    """Embed 1D signal into higher-dimensional space for Rips complex construction.

    Takens' theorem: Time-delay embedding reconstructs attractor topology from
    1D observations. Rotation curve residuals embed into a space where H1 loops
    correspond to cyclic patterns and H2 voids to more complex structure.

    Args:
        signal: 1D array, shape (n,)
        embed_dim: Target embedding dimension (default 3)
        delay: Time delay between coordinates (default 1)

    Returns:
        Point cloud array, shape (n - (embed_dim-1)*delay, embed_dim)

    Note: Pure math utility - does NOT emit receipt.
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    # Number of points in embedded space
    n_embedded = n - (embed_dim - 1) * delay

    if n_embedded <= 0:
        raise ValueError(f"Signal too short ({n}) for embedding dim={embed_dim}, delay={delay}")

    # Build embedded point cloud
    embedded = np.zeros((n_embedded, embed_dim))
    for d in range(embed_dim):
        start_idx = d * delay
        embedded[:, d] = signal[start_idx:start_idx + n_embedded]

    return embedded


def compute_persistence(residuals: np.ndarray, max_dim: int = MAX_DIMENSION) -> dict:
    """Compute persistence diagrams for H0, H1, H2 from rotation curve residuals.

    Args:
        residuals: 1D array of (v_observed - v_predicted), shape (n_points,)
        max_dim: Maximum homology dimension (default MAX_DIMENSION = 2)

    Returns:
        dict with keys:
            "h0": dict with "birth", "death", "persistence" arrays
            "h1": dict with "birth", "death", "persistence" arrays
            "h2": dict with "birth", "death", "persistence" arrays
            "embedding_dim": int (the embedding used)

    Note: Pure function, no side effects. Features with infinite death are excluded.
          Does NOT emit receipt (pure computation).
    """
    residuals = np.asarray(residuals).flatten()

    # Embed 1D residuals into 3D space for H2 computation
    embed_dim = 3
    delay = 1

    try:
        point_cloud = time_delay_embed(residuals, embed_dim=embed_dim, delay=delay)
    except ValueError:
        # Signal too short - return empty diagrams
        return {
            "h0": {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])},
            "h1": {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])},
            "h2": {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])},
            "embedding_dim": embed_dim
        }

    # Compute Rips filtration using ripser
    result = ripser(point_cloud, maxdim=max_dim)
    diagrams = result["dgms"]

    def process_diagram(dgm: np.ndarray) -> dict:
        """Extract finite features from diagram."""
        if len(dgm) == 0:
            return {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}

        # Filter out infinite death values
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]

        if len(dgm_finite) == 0:
            return {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}

        birth = dgm_finite[:, 0]
        death = dgm_finite[:, 1]
        persistence = death - birth

        return {"birth": birth, "death": death, "persistence": persistence}

    return {
        "h0": process_diagram(diagrams[0]) if len(diagrams) > 0 else {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])},
        "h1": process_diagram(diagrams[1]) if len(diagrams) > 1 else {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])},
        "h2": process_diagram(diagrams[2]) if len(diagrams) > 2 else {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])},
        "embedding_dim": embed_dim
    }


def wasserstein_distance(diag1: dict, diag2: dict, p: int = 1) -> float:
    """Compute Wasserstein-p distance between two persistence diagrams.

    Args:
        diag1: Persistence diagram dict with "birth", "death" arrays
        diag2: Persistence diagram dict with "birth", "death" arrays
        p: Order of Wasserstein distance (default 1 per Grok recommendation)

    Returns:
        float: Wasserstein-p distance (non-negative)

    Note: Does NOT emit receipt (pure math).
    """
    # Extract (birth, death) pairs
    birth1 = np.asarray(diag1.get("birth", []))
    death1 = np.asarray(diag1.get("death", []))
    birth2 = np.asarray(diag2.get("birth", []))
    death2 = np.asarray(diag2.get("death", []))

    # Handle empty diagrams
    n1 = len(birth1)
    n2 = len(birth2)

    if n1 == 0 and n2 == 0:
        return 0.0

    # Cost of matching to diagonal: (birth, death) -> (birth+death)/2, (birth+death)/2
    # Distance = (death - birth) / sqrt(2) for L2, or (death - birth) / 2 for L1 approximation
    def diagonal_cost(birth: np.ndarray, death: np.ndarray) -> np.ndarray:
        """Cost to match point to diagonal."""
        persistence = death - birth
        if p == 1:
            return persistence / 2.0  # L1 cost to diagonal
        else:
            return persistence / np.sqrt(2.0)  # L2 cost to diagonal

    if n1 == 0:
        # All points in diag2 match to diagonal
        costs = diagonal_cost(birth2, death2)
        return float(np.sum(costs ** p) ** (1.0 / p)) if p > 1 else float(np.sum(costs))

    if n2 == 0:
        # All points in diag1 match to diagonal
        costs = diagonal_cost(birth1, death1)
        return float(np.sum(costs ** p) ** (1.0 / p)) if p > 1 else float(np.sum(costs))

    # Build cost matrix for optimal matching
    # Size: (n1 + n2) x (n1 + n2) to handle unmatched points
    total = n1 + n2
    cost_matrix = np.full((total, total), np.inf)

    # Points 1 (birth1, death1) and points 2 (birth2, death2)
    pts1 = np.column_stack([birth1, death1])
    pts2 = np.column_stack([birth2, death2])

    # Cost between points from both diagrams
    for i in range(n1):
        for j in range(n2):
            diff = pts1[i] - pts2[j]
            if p == 1:
                cost_matrix[i, j] = np.sum(np.abs(diff))
            else:
                cost_matrix[i, j] = np.sqrt(np.sum(diff ** 2))

    # Cost for matching points in diag1 to diagonal (represented as positions n2, n2+1, ...)
    diag1_to_diag_costs = diagonal_cost(birth1, death1)
    for i in range(n1):
        for k in range(n2, total):
            if k - n2 == i:  # Each point has its own diagonal slot
                cost_matrix[i, k] = diag1_to_diag_costs[i]

    # Cost for matching points in diag2 to diagonal (represented as positions n1, n1+1, ...)
    diag2_to_diag_costs = diagonal_cost(birth2, death2)
    for j in range(n2):
        for k in range(n1, total):
            if k - n1 == j:  # Each point has its own diagonal slot
                cost_matrix[k, j] = diag2_to_diag_costs[j]

    # Diagonal-to-diagonal matching (zero cost for unmatched pairs)
    for i in range(n1, total):
        for j in range(n2, total):
            cost_matrix[i, j] = 0.0

    # Solve optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return float(total_cost)


def count_features(diagram: dict, threshold: float = PERSISTENCE_NOISE_FLOOR) -> int:
    """Count persistent features above noise floor.

    Args:
        diagram: Single dimension dict with "persistence" array
        threshold: Minimum persistence to count (default PERSISTENCE_NOISE_FLOOR)

    Returns:
        int: Count of features with persistence > threshold

    Note: Does NOT emit receipt (utility).
    """
    persistence = np.asarray(diagram.get("persistence", []))
    if len(persistence) == 0:
        return 0
    return int(np.sum(persistence > threshold))


def compute_complexity_bits(diagrams: dict) -> float:
    """Compute total information content of persistence diagrams in bits.

    Formula:
        bits = sum_d sum_f log2(1 + persistence_f / noise_floor)
        Sum over dimensions d = {H0, H1, H2}
        Sum over features f with persistence > PERSISTENCE_NOISE_FLOOR

    Intuition: Each significant feature contributes log2(1 + relative_persistence) bits.
    More persistent features = more information. If topology encodes the same
    information as NFW (~150 bits) in 20-50 bits, topology is a better compression.

    Args:
        diagrams: dict with keys "h0", "h1", "h2"

    Returns:
        float: Total bits (expect 20-65 for typical galaxies)

    Note: Does NOT emit receipt (pure computation).
    """
    total_bits = 0.0

    for dim_key in ["h0", "h1", "h2"]:
        diag = diagrams.get(dim_key, {})
        persistence = np.asarray(diag.get("persistence", []))

        for pers in persistence:
            if pers > PERSISTENCE_NOISE_FLOOR:
                relative_pers = pers / PERSISTENCE_NOISE_FLOOR
                total_bits += np.log2(1.0 + relative_pers)

    return float(total_bits)


def classify_topology(
    h1_count: int,
    h2_count: int,
    complexity_bits: float,
    h1_persistence_mean: float
) -> str:
    """Classify physics regime from topological signature.

    Classification Rules (from Grok thresholds):
        - H1 < 2 AND complexity < 30: "newtonian"
        - H1 >= 2 AND H1 <= 5 AND 0.04 <= mean_H1_persistence <= 0.08: "dm_halo"
        - H1 >= 2 AND complexity > 48 but persistence outside halo range: "mond"
        - H2 > 3: "novel" (possible DE signature)
        - complexity > 48 AND H1 > 2: "dm_halo" (DM flag from Grok)
        - None of above: "undetermined"

    Priority: novel > dm_halo > mond > newtonian > undetermined

    Args:
        h1_count: Number of H1 features above noise floor
        h2_count: Number of H2 features above noise floor
        complexity_bits: Total information content in bits
        h1_persistence_mean: Mean persistence of H1 features

    Returns:
        str: One of "newtonian", "mond", "dm_halo", "novel", "undetermined"

    Note: Does NOT emit receipt (classification utility).
    """
    # Priority 1: Novel (H2 > 3 indicates dark energy signature)
    if h2_count > DE_H2_THRESHOLD:
        return "novel"

    # Priority 2: dm_halo (DM flag: complexity > 48 AND H1 > 2)
    if complexity_bits > DM_COMPLEXITY_THRESHOLD and h1_count > DM_H1_MINIMUM:
        return "dm_halo"

    # Priority 3: dm_halo (H1 in range [2,5] with specific persistence)
    if (h1_count >= DM_H1_MINIMUM and h1_count <= 5 and
        HALO_PERSISTENCE_LOW <= h1_persistence_mean <= HALO_PERSISTENCE_HIGH):
        return "dm_halo"

    # Priority 4: MOND (H1 >= 2 AND complexity > 48 but persistence outside halo range)
    if (h1_count >= DM_H1_MINIMUM and complexity_bits > DM_COMPLEXITY_THRESHOLD and
        not (HALO_PERSISTENCE_LOW <= h1_persistence_mean <= HALO_PERSISTENCE_HIGH)):
        return "mond"

    # Priority 5: Newtonian (simple topology)
    if h1_count < DM_H1_MINIMUM and complexity_bits < 30:
        return "newtonian"

    # Default
    return "undetermined"


def topology_complexity(residuals: np.ndarray) -> dict:
    """Full topological analysis pipeline.

    Compute persistence, count features, compute bits, classify.

    Args:
        residuals: 1D array of rotation curve residuals

    Returns:
        dict with keys:
            "h1_count": int
            "h2_count": int
            "h1_persistence_mean": float
            "h2_persistence_mean": float
            "total_bits": float
            "classification": str
            "diagrams": dict (raw H0/H1/H2 diagrams)

    Note: Does NOT emit receipt (intermediate computation).
    """
    # Compute persistence diagrams
    diagrams = compute_persistence(residuals)

    # Count features above noise floor
    h1_count = count_features(diagrams["h1"])
    h2_count = count_features(diagrams["h2"])

    # Compute mean H1 persistence (for features above noise floor)
    h1_pers = np.asarray(diagrams["h1"]["persistence"])
    h1_significant = h1_pers[h1_pers > PERSISTENCE_NOISE_FLOOR]
    h1_persistence_mean = float(np.mean(h1_significant)) if len(h1_significant) > 0 else 0.0

    # Compute mean H2 persistence
    h2_pers = np.asarray(diagrams["h2"]["persistence"])
    h2_significant = h2_pers[h2_pers > PERSISTENCE_NOISE_FLOOR]
    h2_persistence_mean = float(np.mean(h2_significant)) if len(h2_significant) > 0 else 0.0

    # Compute total complexity bits
    total_bits = compute_complexity_bits(diagrams)

    # Classify physics regime
    classification = classify_topology(h1_count, h2_count, total_bits, h1_persistence_mean)

    return {
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h1_persistence_mean": h1_persistence_mean,
        "h2_persistence_mean": h2_persistence_mean,
        "total_bits": total_bits,
        "classification": classification,
        "diagrams": diagrams
    }


# === STOPRULES ===

def stoprule_invalid_residuals(reason: str) -> None:
    """Trigger stoprule for invalid residuals.

    Emits anomaly receipt and raises StopRule.

    Args:
        reason: Description of the validation failure
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "residuals",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "reason": reason
    })
    raise StopRule(f"Invalid residuals for topology analysis: {reason}")


def emit_timeout_anomaly(elapsed: float) -> None:
    """Emit anomaly receipt for computation timeout (graceful degradation).

    Per GÖDEL scenario: graceful degradation, not crash.

    Args:
        elapsed: Actual computation time in seconds
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "topology_time",
        "baseline": COMPUTATION_TIMEOUT,
        "delta": elapsed - COMPUTATION_TIMEOUT,
        "classification": "degradation",
        "action": "alert"
    })


# === MAIN ENTRY POINT ===

def analyze_galaxy(
    galaxy_id: str,
    residuals: np.ndarray,
    baseline_diagram: Optional[dict] = None
) -> dict:
    """MAIN ENTRY POINT. Full topological analysis with receipt emission.

    Args:
        galaxy_id: Identifier string for the galaxy
        residuals: 1D array of (v_obs - v_predicted) from witness.py
        baseline_diagram: Optional baseline for Wasserstein comparison (e.g., average SPARC)

    Returns:
        The topology receipt dict

    MUST emit receipt (topology_receipt).
    """
    residuals = np.asarray(residuals).flatten()

    # Validate residuals
    if len(residuals) == 0:
        stoprule_invalid_residuals("empty residuals array")

    if len(residuals) < 10:
        stoprule_invalid_residuals(f"too few points ({len(residuals)} < 10)")

    if np.any(np.isnan(residuals)):
        stoprule_invalid_residuals("residuals contain NaN values")

    # Time the computation
    start_time = time.time()

    # Run topological analysis
    topo_result = topology_complexity(residuals)

    elapsed = time.time() - start_time

    # Check for timeout (graceful degradation per GÖDEL scenario)
    timeout_flag = False
    if elapsed > COMPUTATION_TIMEOUT:
        emit_timeout_anomaly(elapsed)
        timeout_flag = True

    # Compute Wasserstein distance to baseline if provided
    wasserstein_to_baseline = None
    if baseline_diagram is not None:
        wasserstein_to_baseline = wasserstein_distance(
            topo_result["diagrams"]["h1"],
            baseline_diagram,
            p=1
        )

    # Build payload for hashing
    payload_data = {
        "galaxy_id": galaxy_id,
        "h1_count": topo_result["h1_count"],
        "h2_count": topo_result["h2_count"],
        "total_bits": topo_result["total_bits"],
        "classification": topo_result["classification"]
    }

    # Emit topology receipt
    receipt = emit_receipt("topology", {
        "tenant_id": TENANT_ID,
        "galaxy_id": galaxy_id,
        "h1_count": topo_result["h1_count"],
        "h2_count": topo_result["h2_count"],
        "h1_persistence_mean": topo_result["h1_persistence_mean"],
        "h2_persistence_mean": topo_result["h2_persistence_mean"],
        "total_bits": topo_result["total_bits"],
        "wasserstein_to_baseline": wasserstein_to_baseline,
        "classification": topo_result["classification"],
        "timeout": timeout_flag,
        "computation_time": elapsed,
        "payload_hash": dual_hash(str(payload_data))
    })

    return receipt


def create_baseline_diagram(residuals_list: list) -> dict:
    """Create average baseline persistence diagram from multiple galaxies.

    Uses Wasserstein barycenter approximation: average the H1 diagrams
    by taking the centroid of aligned (birth, death) points.

    Args:
        residuals_list: List of residual arrays from multiple galaxies

    Returns:
        Composite baseline H1 diagram dict with "birth", "death", "persistence" arrays

    Use case: Create SPARC baseline for Wasserstein comparison
              (v2 scope, synthetic baseline for v1)

    Note: Does NOT emit receipt (baseline construction).
    """
    if len(residuals_list) == 0:
        return {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}

    # Collect all H1 diagrams
    all_diagrams = []
    for residuals in residuals_list:
        try:
            diag = compute_persistence(np.asarray(residuals).flatten())
            h1 = diag["h1"]
            if len(h1["birth"]) > 0:
                all_diagrams.append(h1)
        except Exception:
            continue

    if len(all_diagrams) == 0:
        return {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}

    # Simple barycenter approximation: concatenate all points and find clusters
    # For v1, we use a simpler approach: take the average number of features
    # and use mean birth/death values

    all_births = []
    all_deaths = []

    for diag in all_diagrams:
        # Only include significant features
        mask = diag["persistence"] > PERSISTENCE_NOISE_FLOOR
        all_births.extend(diag["birth"][mask].tolist())
        all_deaths.extend(diag["death"][mask].tolist())

    if len(all_births) == 0:
        return {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}

    # Compute average number of features per galaxy
    avg_n_features = len(all_births) / len(all_diagrams)
    n_baseline = max(1, int(round(avg_n_features)))

    # Sort by persistence and take top n_baseline
    all_births = np.array(all_births)
    all_deaths = np.array(all_deaths)
    all_pers = all_deaths - all_births

    # Take representative features (sorted by persistence)
    sorted_idx = np.argsort(all_pers)[::-1]
    top_idx = sorted_idx[:n_baseline]

    birth_baseline = all_births[top_idx]
    death_baseline = all_deaths[top_idx]
    pers_baseline = all_pers[top_idx]

    return {
        "birth": birth_baseline,
        "death": death_baseline,
        "persistence": pers_baseline
    }


# === LEGACY COMPATIBILITY FUNCTIONS ===

def classify_features(
    diagram: dict,
    threshold: float = PERSISTENCE_NOISE_FLOOR
) -> list:
    """Filter noise and extract significant features from persistence diagram.

    Legacy compatibility function for existing code.

    Args:
        diagram: Either dict with h0/h1/h2 keys or single diagram dict
        threshold: Minimum persistence to keep

    Returns:
        List of (birth, death, persistence) tuples for significant features
    """
    features = []

    # Check if it's a multi-diagram dict (has h0, h1, h2 keys)
    if "h1" in diagram or "h2" in diagram:
        for key in ["h1", "h2"]:
            if key in diagram:
                sub_diag = diagram[key]
                birth = np.asarray(sub_diag.get("birth", []))
                death = np.asarray(sub_diag.get("death", []))
                persistence = np.asarray(sub_diag.get("persistence", []))

                for i in range(len(persistence)):
                    if persistence[i] >= threshold:
                        features.append((float(birth[i]), float(death[i]), float(persistence[i])))
    else:
        # Single diagram
        birth = np.asarray(diagram.get("birth", []))
        death = np.asarray(diagram.get("death", []))
        persistence = np.asarray(diagram.get("persistence", []))

        for i in range(len(persistence)):
            if persistence[i] >= threshold:
                features.append((float(birth[i]), float(death[i]), float(persistence[i])))

    return features


def h1_interpretation(features: list) -> dict:
    """Interpret H1 features as dark matter topology indicators.

    Legacy compatibility function.

    Args:
        features: List of (birth, death, persistence) tuples

    Returns:
        Dict with hole_count, max_persistence, total_persistence, interpretation
    """
    if not features:
        return {
            "hole_count": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "interpretation": "No topological holes detected - trivial H1"
        }

    persistences = [f[2] for f in features]
    max_pers = max(persistences)
    total_pers = sum(persistences)
    hole_count = len(features)

    if max_pers > 0.5:
        interpretation = "Strong H1 features - dark matter halo topology likely"
    elif max_pers > 0.1:
        interpretation = "Moderate H1 features - possible dark matter substructure"
    else:
        interpretation = "Weak H1 features - noise or minimal dark matter contribution"

    return {
        "hole_count": hole_count,
        "max_persistence": float(max_pers),
        "total_persistence": float(total_pers),
        "interpretation": interpretation
    }


def h2_interpretation(features: list) -> dict:
    """Interpret H2 features as gravitational cavity indicators.

    Legacy compatibility function.

    Args:
        features: List of (birth, death, persistence) tuples from H2 diagram

    Returns:
        Dict with void_count, max_persistence, total_persistence, interpretation
    """
    if not features:
        return {
            "void_count": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "interpretation": "No topological voids detected - trivial H2"
        }

    persistences = [f[2] for f in features]
    max_pers = max(persistences)
    total_pers = sum(persistences)
    void_count = len(features)

    if max_pers > 0.5:
        interpretation = "Strong H2 features - significant gravitational cavities"
    elif max_pers > 0.1:
        interpretation = "Moderate H2 features - possible void structure"
    else:
        interpretation = "Weak H2 features - minimal cavity structure"

    return {
        "void_count": void_count,
        "max_persistence": float(max_pers),
        "total_persistence": float(total_pers),
        "interpretation": interpretation
    }


def trivial_topology_check(diagram: dict) -> bool:
    """Check if topology is trivial (no significant H1/H2 features).

    Legacy compatibility function.

    Args:
        diagram: Dict with h1, h2 keys or H1, H2 keys

    Returns:
        True if no significant H1 or H2 features exist
    """
    # Handle both lowercase and uppercase keys
    h1_key = "h1" if "h1" in diagram else "H1"
    h2_key = "h2" if "h2" in diagram else "H2"

    h1_features = classify_features({h1_key: diagram.get(h1_key, diagram.get("H1", {}))})
    h2_features = classify_features({h2_key: diagram.get(h2_key, diagram.get("H2", {}))})

    return len(h1_features) == 0 and len(h2_features) == 0


def topology_loss_term(obs: dict, pred: dict) -> float:
    """Compute Wasserstein distance between observed and predicted H1 diagrams.

    Legacy compatibility function.

    Args:
        obs: Observed persistence diagram dict
        pred: Predicted persistence diagram dict

    Returns:
        Wasserstein distance (float), 0.0 if either is empty
    """
    # Handle both key formats
    obs_h1 = obs.get("h1", obs.get("H1", {}))
    pred_h1 = pred.get("h1", pred.get("H1", {}))

    # Convert to new format if needed
    if isinstance(obs_h1, np.ndarray):
        # Old format: (N, 2) array of [birth, death]
        if len(obs_h1) == 0:
            obs_h1 = {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}
        else:
            finite_mask = np.isfinite(obs_h1[:, 1])
            obs_finite = obs_h1[finite_mask]
            obs_h1 = {
                "birth": obs_finite[:, 0] if len(obs_finite) > 0 else np.array([]),
                "death": obs_finite[:, 1] if len(obs_finite) > 0 else np.array([]),
                "persistence": obs_finite[:, 1] - obs_finite[:, 0] if len(obs_finite) > 0 else np.array([])
            }

    if isinstance(pred_h1, np.ndarray):
        if len(pred_h1) == 0:
            pred_h1 = {"birth": np.array([]), "death": np.array([]), "persistence": np.array([])}
        else:
            finite_mask = np.isfinite(pred_h1[:, 1])
            pred_finite = pred_h1[finite_mask]
            pred_h1 = {
                "birth": pred_finite[:, 0] if len(pred_finite) > 0 else np.array([]),
                "death": pred_finite[:, 1] if len(pred_finite) > 0 else np.array([]),
                "persistence": pred_finite[:, 1] - pred_finite[:, 0] if len(pred_finite) > 0 else np.array([])
            }

    return wasserstein_distance(obs_h1, pred_h1, p=1)


def topology_receipt_emit(galaxy_id: str, diagram: dict, interpretation: dict) -> dict:
    """Emit topology_receipt with diagram and interpretation data.

    Legacy compatibility function.

    Args:
        galaxy_id: Galaxy identifier string
        diagram: Dict with H0, H1, H2 persistence diagrams
        interpretation: Dict with h1_interpretation and h2_interpretation results

    Returns:
        Emitted receipt dict
    """
    # Count features
    h1_features = classify_features({"h1": diagram.get("h1", diagram.get("H1", {}))})
    h2_features = classify_features({"h2": diagram.get("h2", diagram.get("H2", {}))})

    receipt_data = {
        "tenant_id": TENANT_ID,
        "galaxy_id": galaxy_id,
        "h1_hole_count": len(h1_features),
        "h2_void_count": len(h2_features),
        "trivial_topology": len(h1_features) == 0 and len(h2_features) == 0,
        "interpretation": interpretation
    }

    return emit_receipt("topology", receipt_data)
