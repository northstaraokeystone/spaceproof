"""D12 recursion CLI commands."""

import json


def cmd_d12_info():
    """Show D12 configuration."""
    from src.fractal_layers import get_d12_info

    info = get_d12_info()
    print(json.dumps(info, indent=2))


def cmd_d12_push(tree_size: int, base_alpha: float, simulate: bool = False):
    """Run D12 recursion push for alpha >= 3.65."""
    from src.fractal_layers import d12_push

    result = d12_push(tree_size, base_alpha, simulate=simulate)
    print(json.dumps(result, indent=2))


def cmd_d12_mercury_hybrid(tree_size: int, base_alpha: float, simulate: bool = False):
    """Run integrated D12 + Mercury thermal hybrid."""
    from src.mercury_thermal_hybrid import d12_mercury_hybrid

    result = d12_mercury_hybrid(tree_size, base_alpha, simulate=simulate)
    print(json.dumps(result, indent=2))
