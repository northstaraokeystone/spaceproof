"""cli/fractal.py - Fractal CLI commands

Commands for fractal ceiling breach and alpha boost operations.
"""

from .base import print_header, print_result, print_slo_check, print_receipt_note

from src.fractal_layers import (
    load_fractal_spec,
    multi_scale_fractal,
    get_fractal_info,
    FRACTAL_SCALES,
    FRACTAL_UPLIFT_TARGET,
    ALPHA_CEILING_SINGLE_SCALE,
    ALPHA_NEAR_CEILING,
)
from src.quantum_rl_hybrid import (
    quantum_fractal_hybrid,
    QUANTUM_RETENTION_BOOST,
    FRACTAL_CONTRIBUTION,
    HYBRID_BOOST_TOTAL,
)


def cmd_fractal_push(tree_size: int, simulate: bool) -> None:
    """Run fractal ceiling breach.

    Args:
        tree_size: Merkle tree size
        simulate: Whether to output receipts
    """
    print_header(f"FRACTAL CEILING BREACH (n={tree_size:.2e})")

    # Load spec
    try:
        spec = load_fractal_spec()
        print("\nSpec loaded:")
        print_result("alpha_near_ceiling", spec['alpha_near_ceiling'])
        print_result("fractal_uplift_target", spec['fractal_uplift_target'])
        print_result("ceiling_break_target", spec['ceiling_break_target'])
        print_result("scales", spec['scales'])
    except FileNotFoundError:
        print("\nUsing defaults")

    print("\nConfiguration:")
    print_result("tree_size", tree_size)
    print_result("scales", FRACTAL_SCALES)
    print_result("single_scale_alpha", ALPHA_NEAR_CEILING)
    print_result("target_uplift", FRACTAL_UPLIFT_TARGET)

    print("\nComputing fractal multi-scale entropy...")

    result = multi_scale_fractal(
        tree_size=tree_size,
        entropy=ALPHA_NEAR_CEILING
    )

    print("\nRESULTS:")
    print_result("Fractal dimension", result['fractal_dimension'])
    print_result("Cross-scale correlation", result['fractal_correlation'])
    print_result("Fractal uplift", result['fractal_uplift'])
    print_result("Single-scale alpha", result['single_scale_alpha'])
    print_result("Fractal alpha", result['fractal_alpha'])
    print_result("Ceiling breached", result['ceiling_breached'])

    print("\nMulti-Scale Entropies:")
    for scale, entropy in result['multi_scale_entropies'].items():
        print(f"  Scale {scale}: {entropy}")

    print("\nSLO VALIDATION:")
    uplift_ok = result['fractal_uplift'] >= FRACTAL_UPLIFT_TARGET * 0.8
    ceiling_ok = result['ceiling_breached']
    dimension_ok = 1.5 <= result['fractal_dimension'] <= 2.0

    print_slo_check(f"Uplift >= {FRACTAL_UPLIFT_TARGET * 0.8}", uplift_ok, result['fractal_uplift'])
    print_slo_check(f"Alpha > {ALPHA_CEILING_SINGLE_SCALE}", ceiling_ok, result['fractal_alpha'])
    print_slo_check("Dimension in [1.5, 2.0]", dimension_ok, result['fractal_dimension'])

    if simulate:
        print_receipt_note("fractal_layer")

    print("=" * 60)


def cmd_alpha_boost(mode: str, tree_size: int, simulate: bool) -> None:
    """Run with specified alpha boost mode.

    Args:
        mode: Boost mode - off, quantum, fractal, or hybrid
        tree_size: Merkle tree size
        simulate: Whether to output receipts
    """
    print_header(f"ALPHA BOOST MODE: {mode.upper()}")

    print("\nConfiguration:")
    print_result("mode", mode)
    print_result("tree_size", tree_size)
    print_result("base_alpha", ALPHA_NEAR_CEILING)

    if mode == "off":
        print("\nRESULTS (no boost):")
        print_result("Final alpha", ALPHA_NEAR_CEILING)
        print_result("Ceiling breached", False)
        print("=" * 60)
        return

    elif mode == "quantum":
        print("\nApplying quantum boost only...")
        final_alpha = ALPHA_NEAR_CEILING + QUANTUM_RETENTION_BOOST
        print("\nRESULTS:")
        print_result("Quantum contribution", f"+{QUANTUM_RETENTION_BOOST}")
        print_result("Final alpha", final_alpha)
        print_result("Ceiling breached", final_alpha > ALPHA_CEILING_SINGLE_SCALE)

    elif mode == "fractal":
        print("\nApplying fractal boost only...")
        fractal_result = multi_scale_fractal(
            tree_size=tree_size,
            entropy=ALPHA_NEAR_CEILING
        )
        print("\nRESULTS:")
        print_result("Fractal contribution", f"+{fractal_result['fractal_uplift']}")
        print_result("Final alpha", fractal_result['fractal_alpha'])
        print_result("Ceiling breached", fractal_result['ceiling_breached'])

    elif mode == "hybrid":
        print("\nApplying quantum-fractal hybrid boost...")

        # Get fractal result
        fractal_result = multi_scale_fractal(
            tree_size=tree_size,
            entropy=ALPHA_NEAR_CEILING
        )

        # Apply hybrid
        hybrid_result = quantum_fractal_hybrid(
            state={"alpha": ALPHA_NEAR_CEILING},
            fractal_result=fractal_result
        )

        print("\nRESULTS:")
        print_result("Quantum contribution", f"+{hybrid_result['quantum_contribution']}")
        print_result("Fractal contribution", f"+{hybrid_result['fractal_contribution']}")
        print_result("Total hybrid boost", f"+{hybrid_result['total_hybrid_boost']}")
        print_result("Base alpha", hybrid_result['base_alpha'])
        print_result("Final alpha", hybrid_result['final_alpha'])
        print_result("Instability", hybrid_result['instability'])
        print_result("Ceiling breached", hybrid_result['ceiling_breached'])

        print("\nSLO VALIDATION:")
        ceiling_ok = hybrid_result['ceiling_breached']
        instability_ok = hybrid_result['instability'] == 0.0
        print_slo_check(f"Alpha > {ALPHA_CEILING_SINGLE_SCALE}", ceiling_ok, hybrid_result['final_alpha'])
        print_slo_check("Instability == 0.0", instability_ok, hybrid_result['instability'])

        if simulate:
            print_receipt_note("quantum_fractal_hybrid")

    else:
        print(f"\nUnknown mode: {mode}")
        print("Valid modes: off, quantum, fractal, hybrid")

    print("=" * 60)


def cmd_fractal_info() -> None:
    """Show fractal configuration."""
    print_header("FRACTAL LAYERS CONFIGURATION")

    info = get_fractal_info()

    print("\nConfiguration:")
    print_result("fractal_scales", info['fractal_scales'])
    print_result("fractal_dimension_min", info['fractal_dimension_min'])
    print_result("fractal_dimension_max", info['fractal_dimension_max'])
    print_result("fractal_uplift_target", info['fractal_uplift_target'])
    print_result("alpha_ceiling_single_scale", info['alpha_ceiling_single_scale'])
    print_result("alpha_near_ceiling", info['alpha_near_ceiling'])
    print_result("correlation_coeff_range", info['correlation_coeff_range'])
    print_result("n_scales", info['n_scales'])

    print("\nFormulas:")
    formulas = info['formulas']
    print(f"  Entropy: {formulas['entropy']}")
    print(f"  Dimension: {formulas['dimension']}")
    print(f"  Uplift: {formulas['uplift']}")

    print("\nExpected Results:")
    expected = info['expected_results']
    print_result("single_scale", expected['single_scale'])
    print_result("fractal_uplift", expected['fractal_uplift'])
    print_result("fractal_alpha", expected['fractal_alpha'])
    print_result("ceiling_status", expected['ceiling_status'])

    print(f"\nDescription: {info['description']}")

    print_receipt_note("fractal_info")
    print("=" * 60)


def cmd_ceiling_status() -> None:
    """Show current ceiling breach status."""
    print_header("SHANNON CEILING STATUS")

    print("\nSingle-Scale Analysis:")
    print_result("Current alpha", ALPHA_NEAR_CEILING)
    print_result("Shannon ceiling", ALPHA_CEILING_SINGLE_SCALE)
    print_result("Gap to ceiling", ALPHA_CEILING_SINGLE_SCALE - ALPHA_NEAR_CEILING)
    print_result("Status", "NEAR CEILING")

    print("\nAvailable Boosts:")
    print_result("Quantum boost", f"+{QUANTUM_RETENTION_BOOST}")
    print_result("Fractal boost", f"+{FRACTAL_CONTRIBUTION}")
    print_result("Hybrid boost", f"+{HYBRID_BOOST_TOTAL}")

    print("\nProjected Alpha (with hybrid):")
    hybrid_alpha = ALPHA_NEAR_CEILING + HYBRID_BOOST_TOTAL
    print_result("Base + hybrid", hybrid_alpha)
    print_result("Ceiling breached", hybrid_alpha > ALPHA_CEILING_SINGLE_SCALE)

    print("\nRecommendation: Use --alpha_boost hybrid")

    print("=" * 60)
