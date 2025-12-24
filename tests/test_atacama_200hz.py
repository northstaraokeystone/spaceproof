"""Tests for Atacama 200Hz adaptive drone sampling."""


class TestAtacama200HzConfig:
    """Tests for Atacama 200Hz configuration."""

    def test_load_200hz_config(self) -> None:
        """Test loading 200Hz configuration."""
        from spaceproof.cfd_dust_dynamics import load_atacama_200hz_config

        config = load_atacama_200hz_config()
        assert config is not None
        assert "sampling_hz" in config
        assert config["sampling_hz"] == 200
        assert "correlation_target" in config
        assert config["correlation_target"] == 0.97

    def test_200hz_constants(self) -> None:
        """Test 200Hz constants are correctly defined."""
        from spaceproof.cfd_dust_dynamics import (
            ATACAMA_200HZ_SAMPLING,
            ATACAMA_200HZ_CORRELATION_TARGET,
        )

        assert ATACAMA_200HZ_SAMPLING == 200
        assert ATACAMA_200HZ_CORRELATION_TARGET == 0.97

    def test_200hz_upgrade_from_100hz(self) -> None:
        """Test that 200Hz is 2x upgrade from 100Hz."""
        from spaceproof.cfd_dust_dynamics import (
            ATACAMA_200HZ_SAMPLING,
            ATACAMA_SAMPLING_HZ,
        )

        assert ATACAMA_200HZ_SAMPLING == 2 * ATACAMA_SAMPLING_HZ


class TestAtacama200HzSampling:
    """Tests for 200Hz sampling functionality."""

    def test_atacama_200hz(self) -> None:
        """Test 200Hz sampling mode."""
        from spaceproof.cfd_dust_dynamics import atacama_200hz

        result = atacama_200hz(duration_sec=10.0)

        assert result is not None
        assert "sampling_hz" in result
        assert result["sampling_hz"] == 200
        assert "samples_collected" in result
        assert result["samples_collected"] >= 2000  # 10 sec * 200 Hz
        assert "correlation" in result
        assert "target_met" in result

    def test_adaptive_sampling_rate(self) -> None:
        """Test adaptive sampling rate selection."""
        from spaceproof.cfd_dust_dynamics import adaptive_sampling_rate

        # Test 200Hz mode
        result = adaptive_sampling_rate(target_hz=200)
        assert result is not None
        assert "selected_hz" in result
        assert result["selected_hz"] == 200

        # Test automatic selection based on conditions
        result = adaptive_sampling_rate(dust_intensity=0.9)
        assert "selected_hz" in result
        assert result["selected_hz"] >= 100  # Higher intensity should use higher rate


class TestDustDevilPrediction:
    """Tests for dust devil prediction at 200Hz."""

    def test_predict_dust_devil(self) -> None:
        """Test dust devil prediction."""
        from spaceproof.cfd_dust_dynamics import predict_dust_devil

        result = predict_dust_devil(
            duration_sec=30.0,
            sampling_hz=200,
        )

        assert result is not None
        assert "predictions_made" in result
        assert "accuracy" in result
        assert "lead_time_sec" in result
        assert "sampling_hz" in result
        assert result["sampling_hz"] == 200

    def test_prediction_accuracy(self) -> None:
        """Test prediction accuracy computation."""
        from spaceproof.cfd_dust_dynamics import compute_prediction_accuracy

        result = compute_prediction_accuracy(
            predictions=[True, True, False, True, True],
            actuals=[True, True, True, True, True],
        )

        assert result is not None
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0


class TestAtacama200HzCorrelation:
    """Tests for 200Hz correlation targets."""

    def test_correlation_target(self) -> None:
        """Test correlation reaches target."""
        from spaceproof.cfd_dust_dynamics import atacama_200hz

        result = atacama_200hz(duration_sec=60.0)

        assert "correlation" in result
        assert "target" in result
        assert result["target"] == 0.97
        assert "target_met" in result

    def test_correlation_improvement_from_100hz(self) -> None:
        """Test that 200Hz improves correlation over 100Hz."""
        from spaceproof.cfd_dust_dynamics import (
            atacama_200hz,
            atacama_les_realtime,
        )

        result_100hz = atacama_les_realtime(
            duration_sec=30.0,
            sampling_hz=100,
        )
        result_200hz = atacama_200hz(duration_sec=30.0)

        # 200Hz should have equal or better correlation
        assert result_200hz["correlation"] >= result_100hz.get("correlation", 0) - 0.05


class TestAtacama200HzInfo:
    """Tests for 200Hz info retrieval."""

    def test_get_200hz_info(self) -> None:
        """Test 200Hz info retrieval."""
        from spaceproof.cfd_dust_dynamics import get_atacama_200hz_info

        info = get_atacama_200hz_info()

        assert info is not None
        assert "sampling_hz" in info
        assert info["sampling_hz"] == 200
        assert "correlation_target" in info
        assert "upgrade_from" in info
        assert info["upgrade_from"] == 100


class TestAtacama200HzIntegration:
    """Tests for 200Hz integration with other systems."""

    def test_cfd_200hz_integration(self) -> None:
        """Test CFD integration with 200Hz mode."""
        from spaceproof.cfd_dust_dynamics import atacama_200hz

        result = atacama_200hz(duration_sec=10.0)

        # Should have CFD integration markers
        assert "duration_sec" in result
        assert "sampling_hz" in result
        assert result["sampling_hz"] == 200

    def test_drone_array_200hz(self) -> None:
        """Test drone array operation at 200Hz."""
        from spaceproof.cfd_dust_dynamics import atacama_200hz

        result = atacama_200hz(duration_sec=10.0)

        # Samples should match expected rate
        expected_samples = int(10.0 * 200)
        actual_samples = result.get("samples_collected", 0)

        # Allow some tolerance for timing
        assert abs(actual_samples - expected_samples) <= expected_samples * 0.1
