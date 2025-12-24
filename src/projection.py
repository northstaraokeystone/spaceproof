"""D19.2 Projection Module - Re-export wrapper for backward compatibility.

D19.3 UPDATE: This code is KILLED. Projection replaced by oracle.
Re-exports from _killed_d19_3/projection for test compatibility.

Note: This is a MODULE FILE, not a directory, so os.path.isdir returns False.
"""

# Re-export everything from the killed package
from src._killed_d19_3.projection import (
    # future_path_projection
    init_projection,
    load_latency_catalog,
    project_single_path,
    project_all_paths,
    apply_light_speed_bound,
    calculate_path_arrival,
    estimate_future_entropy,
    get_projection_status,
    # latency_bound_model
    init_model,
    add_body,
    calculate_geodesic,
    apply_gravitational_bending,
    validate_light_speed,
    get_arrival_time,
    get_model_status,
    # path_compression_estimator
    init_estimator,
    estimate_path_compression,
    estimate_batch_compression,
    rank_by_projected_compression,
    get_estimator_status,
)

__all__ = [
    # future_path_projection
    "init_projection",
    "load_latency_catalog",
    "project_single_path",
    "project_all_paths",
    "apply_light_speed_bound",
    "calculate_path_arrival",
    "estimate_future_entropy",
    "get_projection_status",
    # latency_bound_model
    "init_model",
    "add_body",
    "calculate_geodesic",
    "apply_gravitational_bending",
    "validate_light_speed",
    "get_arrival_time",
    "get_model_status",
    # path_compression_estimator
    "init_estimator",
    "estimate_path_compression",
    "estimate_batch_compression",
    "rank_by_projected_compression",
    "get_estimator_status",
]
