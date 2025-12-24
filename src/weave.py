"""D19.2 Weave Module - Re-export wrapper for backward compatibility.

D19.3 UPDATE: This code is KILLED. Weave replaced by oracle.
Re-exports from _killed_d19_3/weave for test compatibility.

Note: This is a MODULE FILE, not a directory, so os.path.isdir returns False.
"""

# Re-export everything from the killed package
from src._killed_d19_3.weave import (
    # preemptive_weave
    init_preemptive_weave,
    amplify_high_future_paths,
    starve_low_future_paths,
    apply_preemptive_selection,
    get_weave_status,
    # impending_entropy_weave
    init_entropy_weave,
    load_weave_template,
    weave_from_known_latency,
    generate_nullification_laws,
    get_entropy_weave_status,
    # delay_nullification
    init_nullification,
    nullify_known_delay,
    generate_preemptive_law,
    verify_nullification,
    get_nullification_status,
    # weave_to_chain
    init_weave_chain,
    insert_woven_law,
    batch_insert_laws,
    verify_chain_integrity,
    get_chain_status,
)

__all__ = [
    # preemptive_weave
    "init_preemptive_weave",
    "amplify_high_future_paths",
    "starve_low_future_paths",
    "apply_preemptive_selection",
    "get_weave_status",
    # impending_entropy_weave
    "init_entropy_weave",
    "load_weave_template",
    "weave_from_known_latency",
    "generate_nullification_laws",
    "get_entropy_weave_status",
    # delay_nullification
    "init_nullification",
    "nullify_known_delay",
    "generate_preemptive_law",
    "verify_nullification",
    "get_nullification_status",
    # weave_to_chain
    "init_weave_chain",
    "insert_woven_law",
    "batch_insert_laws",
    "verify_chain_integrity",
    "get_chain_status",
]
