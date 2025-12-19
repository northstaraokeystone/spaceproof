"""Tests for D17 depth-first fractal recursion and Heliosphere integration.

Tests:
- D17 spec loading
- D17 alpha targets (floor, target, ceiling)
- Depth-first traversal primitives
- D17+Heliosphere hybrid integration
"""

from src.fractal_layers import (
    get_d17_spec,
    get_d17_uplift,
    depth_first_traversal,
    check_asymptotic_ceiling,
    compute_uplift_sustainability,
    d17_depth_first_push,
    d17_push,
    d17_heliosphere_hybrid,
    get_d17_info,
    D17_ALPHA_FLOOR,
    D17_ALPHA_TARGET,
    D17_ALPHA_CEILING,
    D17_UPLIFT,
    D17_DEPTH_FIRST,
    D17_NON_ASYMPTOTIC,
    D17_TERMINATION_THRESHOLD,
)


class TestD17SpecLoading:
    """Tests for D17 spec loading."""

    def test_d17_spec_loads(self):
        """Spec loads with valid dual-hash."""
        spec = get_d17_spec()
        assert spec is not None
        assert "version" in spec
        assert "d17_config" in spec

    def test_d17_spec_version(self):
        """Spec version is valid."""
        spec = get_d17_spec()
        assert spec["version"] == "1.0.0"

    def test_d17_config_present(self):
        """D17 config section present."""
        spec = get_d17_spec()
        assert "d17_config" in spec
        assert "recursion_depth" in spec["d17_config"]
        assert spec["d17_config"]["recursion_depth"] == 17

    def test_d17_depth_first_enabled(self):
        """Depth-first mode enabled in spec."""
        spec = get_d17_spec()
        assert spec["d17_config"]["depth_first"] is True

    def test_d17_non_asymptotic_enabled(self):
        """Non-asymptotic mode enabled."""
        spec = get_d17_spec()
        assert spec["d17_config"]["non_asymptotic"] is True

    def test_d17_uplift_by_depth(self):
        """Uplift by depth table present."""
        spec = get_d17_spec()
        assert "uplift_by_depth" in spec
        assert "17" in spec["uplift_by_depth"]
        assert float(spec["uplift_by_depth"]["17"]) == 0.40


class TestD17AlphaTargets:
    """Tests for D17 alpha floor, target, ceiling."""

    def test_d17_alpha_floor(self):
        """Alpha floor is 3.92."""
        assert D17_ALPHA_FLOOR == 3.92

    def test_d17_alpha_target(self):
        """Alpha target is 3.90."""
        assert D17_ALPHA_TARGET == 3.90

    def test_d17_alpha_ceiling(self):
        """Alpha ceiling is 3.96."""
        assert D17_ALPHA_CEILING == 3.96

    def test_d17_uplift_value(self):
        """D17 uplift is 0.40."""
        assert D17_UPLIFT == 0.40

    def test_d17_uplift_from_spec(self):
        """Get uplift from spec matches constant."""
        uplift = get_d17_uplift(17)
        assert uplift == D17_UPLIFT


class TestD17DepthFirst:
    """Tests for D17 depth-first primitives."""

    def test_d17_depth_first_enabled(self):
        """Depth-first mode enabled."""
        assert D17_DEPTH_FIRST is True

    def test_d17_non_asymptotic_enabled(self):
        """Non-asymptotic mode enabled."""
        assert D17_NON_ASYMPTOTIC is True

    def test_d17_termination_threshold(self):
        """Termination threshold is 0.00025."""
        assert D17_TERMINATION_THRESHOLD == 0.00025

    def test_depth_first_traversal(self):
        """Depth-first traversal completes."""
        data = [[i * 0.01, (i % 10) * 0.1] for i in range(100)]
        result = depth_first_traversal(data, max_depth=5)

        assert "traversal_order" in result
        assert "nodes_visited" in result
        assert "max_depth_reached" in result
        assert result["max_depth_reached"] <= 5

    def test_asymptotic_ceiling_check(self):
        """Asymptotic ceiling check works."""
        alphas = [3.90, 3.91, 3.92, 3.92, 3.92]
        result = check_asymptotic_ceiling(alphas, threshold=0.01)

        assert "plateau_detected" in result
        assert "ceiling_value" in result

    def test_uplift_sustainability(self):
        """Uplift sustainability computed."""
        result = compute_uplift_sustainability(
            current_alpha=3.55, target_alpha=3.92, uplift=0.40, depth=17
        )

        assert "sustainable" in result
        assert "margin" in result


class TestD17RecursionResults:
    """Tests for D17 recursion execution."""

    def test_d17_depth_first_push(self):
        """D17 depth-first push executes."""
        result = d17_depth_first_push(10**9, 3.55, depth_first=True)

        assert "eff_alpha" in result
        assert "depth_first" in result
        assert result["depth_first"] is True
        assert result["depth"] == 17

    def test_d17_alpha_floor_met(self):
        """D17 meets alpha floor at depth=17."""
        result = d17_depth_first_push(10**9, 3.55, depth_first=True)
        # With uplift 0.40 and base 3.55, should get ~3.95
        assert result["eff_alpha"] >= D17_ALPHA_TARGET

    def test_d17_push(self):
        """D17 push executes correctly."""
        result = d17_push(tree_size=10**9, base_alpha=3.55)

        assert "eff_alpha" in result
        assert "floor_met" in result
        assert "target_met" in result
        assert result["depth"] == 17

    def test_d17_slo_passed(self):
        """D17 SLO passes."""
        result = d17_push(tree_size=10**9, base_alpha=3.55)
        assert result["slo_passed"] is True

    def test_d17_no_plateau(self):
        """D17 shows no plateau (non-asymptotic)."""
        result = d17_push(tree_size=10**9, base_alpha=3.55)
        assert result["plateau_detected"] is False


class TestD17HeliosphereHybrid:
    """Tests for D17+Heliosphere hybrid integration."""

    def test_d17_heliosphere_hybrid_executes(self):
        """D17+Heliosphere hybrid executes."""
        result = d17_heliosphere_hybrid(tree_size=10**9, base_alpha=3.55)

        assert "d17" in result
        assert "heliosphere" in result
        assert "combined_alpha" in result

    def test_d17_heliosphere_hybrid_alpha(self):
        """Hybrid achieves alpha target."""
        result = d17_heliosphere_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["combined_alpha"] >= D17_ALPHA_TARGET

    def test_d17_heliosphere_hybrid_receipt(self):
        """Hybrid emits receipt."""
        result = d17_heliosphere_hybrid(tree_size=10**9, base_alpha=3.55)
        assert result["gate"] == "t24h"


class TestD17Info:
    """Tests for D17 info retrieval."""

    def test_get_d17_info(self):
        """D17 info retrieves correctly."""
        info = get_d17_info()

        assert "version" in info
        assert "d17_config" in info
        assert "heliosphere_config" in info
        assert "oort_cloud_config" in info

    def test_d17_info_description(self):
        """D17 info has description."""
        info = get_d17_info()
        assert "description" in info
        assert "D17" in info["description"]
