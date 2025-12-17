"""Tests for fractal layers and quantum-fractal hybrid functionality."""

import os


class TestFractalSpec:
    """Tests for fractal spec loading."""

    def test_spec_file_exists(self):
        """Fractal hybrid spec file should exist."""
        spec_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "fractal_hybrid_spec.json"
        )
        assert os.path.exists(spec_path)

    def test_spec_loads(self, capsys):
        """Spec should load without error."""
        from src.fractal_layers import load_fractal_spec

        spec = load_fractal_spec()
        assert spec is not None
        assert "alpha_near_ceiling" in spec
        assert "fractal_uplift_target" in spec

    def test_spec_values(self, capsys):
        """Spec values should match expected."""
        from src.fractal_layers import load_fractal_spec

        spec = load_fractal_spec()
        assert spec["alpha_near_ceiling"] == 2.99
        assert spec["fractal_uplift_target"] == 0.05
        assert spec["ceiling_break_target"] == 3.05


class TestFractalDimension:
    """Tests for fractal dimension computation."""

    def test_dimension_bounds(self):
        """Dimension should be in physical bounds [1.5, 2.0]."""
        from src.fractal_layers import compute_fractal_dimension

        # Test with various scale-entropy relationships
        scale_entropies = {1: 2.99, 2: 3.01, 4: 3.03, 8: 3.05, 16: 3.07}
        dimension = compute_fractal_dimension(scale_entropies)

        assert 1.5 <= dimension <= 2.0

    def test_dimension_requires_multiple_scales(self):
        """Dimension computation needs at least 2 scales."""
        from src.fractal_layers import compute_fractal_dimension, FRACTAL_DIMENSION_MIN

        # Single scale returns minimum
        result = compute_fractal_dimension({1: 2.99})
        assert result == FRACTAL_DIMENSION_MIN


class TestMultiScaleEntropy:
    """Tests for multi-scale entropy computation."""

    def test_entropy_scales_with_scale(self):
        """Entropy should increase with scale."""
        from src.fractal_layers import multi_scale_entropy

        result = multi_scale_entropy(1000000, 2.99)

        scales = sorted(result.keys())
        for i in range(len(scales) - 1):
            assert result[scales[i]] <= result[scales[i + 1]]

    def test_all_scales_computed(self):
        """All requested scales should be computed."""
        from src.fractal_layers import multi_scale_entropy, FRACTAL_SCALES

        result = multi_scale_entropy(1000000, 2.99)

        for scale in FRACTAL_SCALES:
            assert scale in result


class TestFractalCorrelation:
    """Tests for cross-scale correlation."""

    def test_correlation_bounds(self):
        """Correlation should be in [0.01, 0.03]."""
        from src.fractal_layers import (
            fractal_correlation,
            multi_scale_entropy,
            CORRELATION_COEFF_MIN,
            CORRELATION_COEFF_MAX,
        )

        scale_entropies = multi_scale_entropy(1000000, 2.99)
        correlation = fractal_correlation(scale_entropies)

        assert CORRELATION_COEFF_MIN <= correlation <= CORRELATION_COEFF_MAX


class TestFractalUplift:
    """Tests for fractal uplift computation."""

    def test_uplift_positive(self):
        """Uplift should be positive."""
        from src.fractal_layers import fractal_uplift

        result = fractal_uplift(2.99, 0.02)
        assert result > 0

    def test_uplift_with_full_analysis(self, capsys):
        """Uplift from full analysis should reach target."""
        from src.fractal_layers import (
            multi_scale_entropy,
            fractal_correlation,
            fractal_uplift,
            compute_fractal_dimension,
            FRACTAL_UPLIFT_TARGET,
        )

        scale_entropies = multi_scale_entropy(1000000, 2.99)
        correlation = fractal_correlation(scale_entropies)
        dimension = compute_fractal_dimension(scale_entropies)
        uplift = fractal_uplift(2.99, correlation, dimension, len(scale_entropies))

        # Should reach at least 80% of target
        assert uplift >= FRACTAL_UPLIFT_TARGET * 0.8


class TestMultiScaleFractal:
    """Tests for full multi-scale fractal analysis."""

    def test_ceiling_breach(self, capsys):
        """Fractal alpha should breach ceiling."""
        from src.fractal_layers import multi_scale_fractal, ALPHA_CEILING_SINGLE_SCALE

        result = multi_scale_fractal(1000000, 2.99)

        assert result["fractal_alpha"] > ALPHA_CEILING_SINGLE_SCALE
        assert result["ceiling_breached"] is True

    def test_result_structure(self, capsys):
        """Result should have expected structure."""
        from src.fractal_layers import multi_scale_fractal

        result = multi_scale_fractal(1000000, 2.99)

        assert "multi_scale_entropies" in result
        assert "fractal_dimension" in result
        assert "fractal_correlation" in result
        assert "fractal_uplift" in result
        assert "single_scale_alpha" in result
        assert "fractal_alpha" in result
        assert "ceiling_breached" in result


class TestQuantumFractalHybrid:
    """Tests for quantum-fractal hybrid policy."""

    def test_hybrid_boost(self, capsys):
        """Hybrid should provide combined boost."""
        from src.fractal_layers import multi_scale_fractal
        from src.quantum_rl_hybrid import (
            quantum_fractal_hybrid,
            QUANTUM_RETENTION_BOOST,
        )

        fractal_result = multi_scale_fractal(1000000, 2.99)
        hybrid_result = quantum_fractal_hybrid(
            state={"alpha": 2.99},
            fractal_result=fractal_result
        )

        assert hybrid_result["quantum_contribution"] == QUANTUM_RETENTION_BOOST
        assert hybrid_result["fractal_contribution"] > 0
        assert hybrid_result["total_hybrid_boost"] > 0.07  # ~0.08

    def test_hybrid_ceiling_breach(self, capsys):
        """Hybrid should breach ceiling."""
        from src.fractal_layers import multi_scale_fractal
        from src.quantum_rl_hybrid import quantum_fractal_hybrid

        fractal_result = multi_scale_fractal(1000000, 2.99)
        hybrid_result = quantum_fractal_hybrid(
            state={"alpha": 2.99},
            fractal_result=fractal_result
        )

        assert hybrid_result["ceiling_breached"] is True
        assert hybrid_result["final_alpha"] > 3.0

    def test_hybrid_zero_instability(self, capsys):
        """Hybrid should maintain zero instability."""
        from src.fractal_layers import multi_scale_fractal
        from src.quantum_rl_hybrid import quantum_fractal_hybrid

        fractal_result = multi_scale_fractal(1000000, 2.99)
        hybrid_result = quantum_fractal_hybrid(
            state={"alpha": 2.99},
            fractal_result=fractal_result
        )

        assert hybrid_result["instability"] == 0.0


class TestHybridBoostInfo:
    """Tests for hybrid boost info function."""

    def test_info_structure(self):
        """Info should have expected structure."""
        from src.quantum_rl_hybrid import get_hybrid_boost_info

        info = get_hybrid_boost_info()

        assert "quantum_fractal_hybrid" in info
        assert "quantum_contribution" in info
        assert "fractal_contribution" in info
        assert "hybrid_boost_total" in info
        assert "expected_results" in info
        assert "physics" in info

    def test_expected_values(self):
        """Info should have correct values."""
        from src.quantum_rl_hybrid import (
            get_hybrid_boost_info,
            QUANTUM_RETENTION_BOOST,
            FRACTAL_CONTRIBUTION,
            HYBRID_BOOST_TOTAL,
        )

        info = get_hybrid_boost_info()

        assert info["quantum_contribution"] == QUANTUM_RETENTION_BOOST
        assert info["fractal_contribution"] == FRACTAL_CONTRIBUTION
        assert info["hybrid_boost_total"] == HYBRID_BOOST_TOTAL


class TestCLIFractalCommands:
    """Tests for CLI fractal commands."""

    def test_fractal_info_runs(self, capsys):
        """Fractal info command should run."""
        from cli.fractal import cmd_fractal_info

        cmd_fractal_info()
        captured = capsys.readouterr()
        assert "FRACTAL LAYERS CONFIGURATION" in captured.out

    def test_ceiling_status_runs(self, capsys):
        """Ceiling status command should run."""
        from cli.fractal import cmd_ceiling_status

        cmd_ceiling_status()
        captured = capsys.readouterr()
        assert "SHANNON CEILING STATUS" in captured.out

    def test_fractal_push_runs(self, capsys):
        """Fractal push command should run."""
        from cli.fractal import cmd_fractal_push

        cmd_fractal_push(1000000, False)
        captured = capsys.readouterr()
        assert "FRACTAL CEILING BREACH" in captured.out
        assert "Ceiling breached" in captured.out

    def test_alpha_boost_hybrid_runs(self, capsys):
        """Alpha boost hybrid command should run."""
        from cli.fractal import cmd_alpha_boost

        cmd_alpha_boost("hybrid", 1000000, False)
        captured = capsys.readouterr()
        assert "ALPHA BOOST MODE: HYBRID" in captured.out
        assert "Quantum contribution" in captured.out
        assert "Fractal contribution" in captured.out

    def test_alpha_boost_off(self, capsys):
        """Alpha boost off should show no boost."""
        from cli.fractal import cmd_alpha_boost

        cmd_alpha_boost("off", 1000000, False)
        captured = capsys.readouterr()
        assert "no boost" in captured.out

    def test_alpha_boost_quantum(self, capsys):
        """Alpha boost quantum should show quantum only."""
        from cli.fractal import cmd_alpha_boost

        cmd_alpha_boost("quantum", 1000000, False)
        captured = capsys.readouterr()
        assert "ALPHA BOOST MODE: QUANTUM" in captured.out
        assert "Quantum contribution" in captured.out


class TestConstants:
    """Tests for module constants."""

    def test_fractal_constants(self):
        """Fractal constants should have expected values."""
        from src.fractal_layers import (
            FRACTAL_SCALES,
            FRACTAL_UPLIFT_TARGET,
            ALPHA_CEILING_SINGLE_SCALE,
            ALPHA_NEAR_CEILING,
        )

        assert FRACTAL_SCALES == [1, 2, 4, 8, 16]
        assert FRACTAL_UPLIFT_TARGET == 0.05
        assert ALPHA_CEILING_SINGLE_SCALE == 3.0
        assert ALPHA_NEAR_CEILING == 2.99

    def test_hybrid_constants(self):
        """Hybrid constants should have expected values."""
        from src.quantum_rl_hybrid import (
            QUANTUM_FRACTAL_HYBRID,
            QUANTUM_RETENTION_BOOST,
            FRACTAL_CONTRIBUTION,
            HYBRID_BOOST_TOTAL,
            ALPHA_CEILING,
        )

        assert QUANTUM_FRACTAL_HYBRID is True
        assert QUANTUM_RETENTION_BOOST == 0.03
        assert FRACTAL_CONTRIBUTION == 0.05
        assert HYBRID_BOOST_TOTAL == 0.08
        assert ALPHA_CEILING == 3.0
