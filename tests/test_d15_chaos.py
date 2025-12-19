"""Tests for D15 quantum-entangled fractal recursion + chaos integration."""


class TestD15FractalRecursion:
    """Tests for D15 depth-15 quantum-entangled fractal recursion."""

    def test_d15_spec_load(self) -> None:
        """Test loading D15 spec from JSON."""
        from src.fractal_layers import get_d15_spec

        spec = get_d15_spec()
        assert spec is not None
        assert "d15_config" in spec
        assert spec["d15_config"]["recursion_depth"] == 15
        assert spec["d15_config"]["alpha_floor"] == 3.81
        assert spec["d15_config"]["alpha_target"] == 3.80
        assert spec["d15_config"]["alpha_ceiling"] == 3.84
        assert spec["d15_config"]["quantum_entanglement"] is True
        assert spec["d15_config"]["entanglement_correlation"] == 0.99

    def test_d15_uplift(self) -> None:
        """Test D15 uplift value."""
        from src.fractal_layers import get_d15_uplift, D15_UPLIFT

        uplift = get_d15_uplift()
        assert uplift == D15_UPLIFT
        assert uplift == 0.36

    def test_d15_info(self) -> None:
        """Test D15 info retrieval."""
        from src.fractal_layers import get_d15_info

        info = get_d15_info()
        assert info is not None
        assert "version" in info
        assert "d15_config" in info
        assert "chaotic_nbody_config" in info
        assert "halo2_config" in info

    def test_d15_entanglement_correlation(self) -> None:
        """Test entanglement correlation computation."""
        from src.fractal_layers import compute_entanglement_correlation

        result = compute_entanglement_correlation(depth=15)
        assert result is not None
        assert "correlation" in result
        assert "target" in result
        assert "target_met" in result
        assert result["target"] == 0.99

    def test_d15_entangled_termination(self) -> None:
        """Test entangled termination check."""
        from src.fractal_layers import entangled_termination_check

        # Should not terminate with high delta
        result = entangled_termination_check(0.1, threshold=0.001)
        assert result["should_terminate"] is False

        # Should terminate with low delta
        result = entangled_termination_check(0.0001, threshold=0.001)
        assert result["should_terminate"] is True

    def test_d15_quantum_push(self) -> None:
        """Test D15 quantum-entangled push."""
        from src.fractal_layers import d15_quantum_push

        result = d15_quantum_push(
            base_alpha=3.45,
            tree_size=10**6,
            depth=15,
        )

        assert result is not None
        assert "eff_alpha" in result
        assert "entanglement_correlation" in result
        assert "quantum_entanglement" in result
        assert result["quantum_entanglement"] is True

    def test_d15_recursive_fractal(self) -> None:
        """Test D15 recursive fractal computation."""
        from src.fractal_layers import d15_recursive_fractal

        result = d15_recursive_fractal(
            base_alpha=3.45,
            tree_size=10**6,
            max_depth=15,
            adaptive=True,
        )

        assert result is not None
        assert "eff_alpha" in result
        assert "depth_reached" in result
        assert "adaptive" in result
        assert result["depth_reached"] <= 15

    def test_d15_push_simulate(self) -> None:
        """Test D15 push in simulation mode."""
        from src.fractal_layers import d15_push

        result = d15_push(
            tree_size=10**6,
            base_alpha=3.45,
            simulate=True,
            adaptive=True,
        )

        assert result is not None
        assert result["mode"] == "simulate"
        assert "eff_alpha" in result
        assert "floor_met" in result
        assert "target_met" in result
        assert "ceiling_met" in result
        assert "quantum_entanglement" in result

    def test_d15_push_execute(self) -> None:
        """Test D15 push in execute mode."""
        from src.fractal_layers import d15_push

        result = d15_push(
            tree_size=10**6,
            base_alpha=3.45,
            simulate=False,
            adaptive=True,
        )

        assert result is not None
        assert result["mode"] == "execute"
        assert result["eff_alpha"] >= 3.81  # Floor should be met

    def test_d15_constants(self) -> None:
        """Test D15 constants are correctly defined."""
        from src.fractal_layers import (
            D15_ALPHA_FLOOR,
            D15_ALPHA_TARGET,
            D15_ALPHA_CEILING,
            D15_UPLIFT,
            D15_TREE_MIN,
            D15_QUANTUM_ENTANGLEMENT,
            D15_ENTANGLEMENT_CORRELATION,
        )

        assert D15_ALPHA_FLOOR == 3.81
        assert D15_ALPHA_TARGET == 3.80
        assert D15_ALPHA_CEILING == 3.84
        assert D15_UPLIFT == 0.36
        assert D15_TREE_MIN == 10**12
        assert D15_QUANTUM_ENTANGLEMENT is True
        assert D15_ENTANGLEMENT_CORRELATION == 0.99


class TestD15ChaosHybrid:
    """Tests for D15 + chaos + backbone integration."""

    def test_d15_chaos_hybrid_simulate(self) -> None:
        """Test D15+chaos hybrid in simulation mode."""
        from src.interstellar_backbone import d15_chaos_hybrid

        result = d15_chaos_hybrid(
            tree_size=10**6,
            base_alpha=3.45,
            simulate=True,
        )

        assert result is not None
        assert result["mode"] == "simulate"
        assert "d15_result" in result
        assert "chaos_result" in result
        assert "backbone_result" in result
        assert "combined_alpha" in result
        assert "chaos_tolerance" in result

    def test_d15_chaos_hybrid_execute(self) -> None:
        """Test D15+chaos hybrid in execute mode."""
        from src.interstellar_backbone import d15_chaos_hybrid

        result = d15_chaos_hybrid(
            tree_size=10**6,
            base_alpha=3.45,
            simulate=False,
        )

        assert result is not None
        assert result["mode"] == "execute"
        assert result["combined_alpha"] >= 3.81  # Floor should be met

    def test_chaos_validation_integration(self) -> None:
        """Test chaos validation integration with backbone."""
        from src.interstellar_backbone import integrate_chaos_validation

        result = integrate_chaos_validation()
        assert result is not None
        assert "chaos_integrated" in result
        assert "lyapunov_exponent" in result
        assert "stability" in result

    def test_backbone_chaos_status(self) -> None:
        """Test backbone chaos status retrieval."""
        from src.interstellar_backbone import get_backbone_chaos_status

        result = get_backbone_chaos_status()
        assert result is not None
        assert "chaos_enabled" in result
        assert "body_count" in result
        assert result["body_count"] == 7
