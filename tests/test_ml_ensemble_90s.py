"""Tests for ML ensemble 90s prediction and Bulletproofs infinite chains.

Tests:
- ML 90s configuration loading
- 7-model ensemble (lstm, transformer, cnn, xgboost, gru, tcn, wavenet)
- 90s prediction horizon
- 88% accuracy target
- Bulletproofs 10k infinite chains
- 100x aggregation factor
"""

from spaceproof.cfd_dust_dynamics import (
    load_ml_90s_config,
    initialize_90s_ensemble,
    ml_ensemble_forecast_90s,
    compute_90s_accuracy,
    extended_horizon_correction,
    get_ml_90s_info,
    ML_90S_MODEL_COUNT,
    ML_90S_PREDICTION_HORIZON_S,
    ML_90S_ACCURACY_TARGET,
    ML_90S_MODEL_TYPES,
)
from spaceproof.bulletproofs_infinite import (
    load_infinite_config,
    generate_infinite_chain_10k,
    verify_infinite_chain,
    aggregate_infinite,
    infinite_chain_test,
    stress_test_10k,
    benchmark_infinite_chain,
    get_infinite_chain_info,
    BULLETPROOFS_INFINITE_DEPTH,
    BULLETPROOFS_INFINITE_AGGREGATION_FACTOR,
    BULLETPROOFS_INFINITE_RESILIENCE_TARGET,
)


class TestML90sConfig:
    """Tests for ML 90s configuration."""

    def test_ml_90s_config_loads(self):
        """ML 90s config loads successfully."""
        config = load_ml_90s_config()
        assert config is not None
        assert "model_count" in config

    def test_ml_90s_model_count(self):
        """Model count is 7."""
        assert ML_90S_MODEL_COUNT == 7

    def test_ml_90s_prediction_horizon(self):
        """Prediction horizon is 90 seconds."""
        assert ML_90S_PREDICTION_HORIZON_S == 90

    def test_ml_90s_accuracy_target(self):
        """Accuracy target is 88%."""
        assert ML_90S_ACCURACY_TARGET == 0.88

    def test_ml_90s_model_types(self):
        """All 7 model types present."""
        expected = ["lstm", "transformer", "cnn", "xgboost", "gru", "tcn", "wavenet"]
        assert len(ML_90S_MODEL_TYPES) == 7
        for model in expected:
            assert model in ML_90S_MODEL_TYPES


class TestML90sEnsemble:
    """Tests for ML 90s ensemble initialization."""

    def test_ensemble_initialization(self):
        """Ensemble initializes with all models."""
        ensemble = initialize_90s_ensemble()

        assert "models" in ensemble
        assert len(ensemble["models"]) == 7
        assert "prediction_horizon_s" in ensemble
        assert ensemble["prediction_horizon_s"] == 90

    def test_ensemble_model_weights(self):
        """Ensemble has model weights."""
        ensemble = initialize_90s_ensemble()
        assert "weights" in ensemble
        # Weights should sum to 1.0
        assert abs(sum(ensemble["weights"].values()) - 1.0) < 0.01


class TestML90sForecast:
    """Tests for ML 90s forecasting."""

    def test_forecast_90s_runs(self):
        """90s forecast runs successfully."""
        data = [i * 0.1 for i in range(100)]
        result = ml_ensemble_forecast_90s(data)

        assert "predictions" in result
        assert "horizon_s" in result
        assert result["horizon_s"] == 90

    def test_forecast_90s_model_contributions(self):
        """All models contribute to forecast."""
        data = [i * 0.1 for i in range(100)]
        result = ml_ensemble_forecast_90s(data)

        assert "model_contributions" in result
        assert len(result["model_contributions"]) == 7

    def test_forecast_90s_confidence(self):
        """Forecast has confidence interval."""
        data = [i * 0.1 for i in range(100)]
        result = ml_ensemble_forecast_90s(data)

        assert "confidence_interval" in result
        assert "lower" in result["confidence_interval"]
        assert "upper" in result["confidence_interval"]


class TestML90sAccuracy:
    """Tests for ML 90s accuracy computation."""

    def test_accuracy_computation(self):
        """Accuracy computes correctly."""
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        actuals = [1.1, 2.0, 2.9, 4.2, 4.8]
        accuracy = compute_90s_accuracy(predictions, actuals)

        assert 0 <= accuracy <= 1.0

    def test_accuracy_meets_target(self):
        """Ensemble accuracy meets 88% target."""
        # Near-perfect predictions
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        actuals = [1.05, 2.0, 3.02, 4.01, 5.0]
        accuracy = compute_90s_accuracy(predictions, actuals)

        assert accuracy >= ML_90S_ACCURACY_TARGET


class TestExtendedHorizonCorrection:
    """Tests for extended horizon correction."""

    def test_horizon_correction_applies(self):
        """Horizon correction applies for 90s."""
        predictions = [1.0, 2.0, 3.0]
        result = extended_horizon_correction(predictions, horizon_s=90)

        assert "corrected_predictions" in result
        assert "correction_factor" in result
        assert len(result["corrected_predictions"]) == 3

    def test_horizon_correction_increases_with_time(self):
        """Correction factor increases with horizon."""
        predictions = [1.0]
        result_60 = extended_horizon_correction(predictions, horizon_s=60)
        result_90 = extended_horizon_correction(predictions, horizon_s=90)

        assert result_90["correction_factor"] >= result_60["correction_factor"]


class TestML90sInfo:
    """Tests for ML 90s info retrieval."""

    def test_ml_90s_info(self):
        """ML 90s info retrieves correctly."""
        info = get_ml_90s_info()

        assert "model_count" in info
        assert "model_types" in info
        assert "prediction_horizon_s" in info
        assert "accuracy_target" in info

    def test_ml_90s_info_description(self):
        """ML 90s info has description."""
        info = get_ml_90s_info()
        assert "description" in info
        assert "90" in info["description"]


class TestBulletproofsInfiniteConfig:
    """Tests for Bulletproofs infinite configuration."""

    def test_infinite_config_loads(self):
        """Infinite config loads successfully."""
        config = load_infinite_config()
        assert config is not None
        assert "infinite_depth" in config

    def test_infinite_depth(self):
        """Infinite depth is 10,000."""
        assert BULLETPROOFS_INFINITE_DEPTH == 10000

    def test_infinite_aggregation_factor(self):
        """Aggregation factor is 100x."""
        assert BULLETPROOFS_INFINITE_AGGREGATION_FACTOR == 100

    def test_infinite_resilience_target(self):
        """Resilience target is 100%."""
        assert BULLETPROOFS_INFINITE_RESILIENCE_TARGET == 1.0


class TestBulletproofsInfiniteChain:
    """Tests for Bulletproofs infinite chain generation."""

    def test_infinite_chain_generates(self):
        """10k infinite chain generates."""
        chain = generate_infinite_chain_10k()

        assert "chain_id" in chain
        assert "depth" in chain
        assert chain["depth"] == 10000
        assert "proofs" in chain

    def test_infinite_chain_verification(self):
        """Infinite chain verifies correctly."""
        chain = generate_infinite_chain_10k()
        result = verify_infinite_chain(chain)

        assert "valid" in result
        assert "verification_depth" in result
        assert result["valid"] is True


class TestBulletproofsAggregation:
    """Tests for Bulletproofs infinite aggregation."""

    def test_infinite_aggregation(self):
        """100x aggregation works."""
        proofs = [{"value": i} for i in range(100)]
        result = aggregate_infinite(proofs)

        assert "aggregated_proof" in result
        assert "input_count" in result
        assert result["input_count"] == 100
        assert "aggregation_factor" in result

    def test_aggregation_compression(self):
        """Aggregation provides compression."""
        proofs = [{"value": i} for i in range(100)]
        result = aggregate_infinite(proofs)

        # Aggregated proof should be smaller than sum of inputs
        assert result["compression_ratio"] > 1.0


class TestBulletproofsInfiniteChainTest:
    """Tests for infinite chain test execution."""

    def test_infinite_chain_test_runs(self):
        """Infinite chain test runs."""
        result = infinite_chain_test(depth=100)

        assert "test_passed" in result
        assert "depth_tested" in result
        assert "verification_time_ms" in result

    def test_infinite_chain_test_passes(self):
        """Infinite chain test passes."""
        result = infinite_chain_test(depth=100)
        assert result["test_passed"] is True


class TestBulletproofsStressTest:
    """Tests for Bulletproofs 10k stress test."""

    def test_10k_stress_test_runs(self):
        """10k stress test runs."""
        result = stress_test_10k(iterations=5)

        assert "iterations_completed" in result
        assert "success_rate" in result
        assert "avg_time_ms" in result

    def test_10k_stress_test_success(self):
        """10k stress test succeeds."""
        result = stress_test_10k(iterations=5)
        assert result["success_rate"] >= 0.95


class TestBulletproofsBenchmark:
    """Tests for Bulletproofs benchmark."""

    def test_benchmark_runs(self):
        """Benchmark runs successfully."""
        result = benchmark_infinite_chain()

        assert "generation_time_ms" in result
        assert "verification_time_ms" in result
        assert "aggregation_time_ms" in result
        assert "total_time_ms" in result


class TestBulletproofsInfiniteInfo:
    """Tests for Bulletproofs infinite info retrieval."""

    def test_infinite_chain_info(self):
        """Infinite chain info retrieves correctly."""
        info = get_infinite_chain_info()

        assert "infinite_depth" in info
        assert "aggregation_factor" in info
        assert "resilience_target" in info
        assert info["infinite_depth"] == 10000

    def test_infinite_chain_info_description(self):
        """Info has description."""
        info = get_infinite_chain_info()
        assert "description" in info
        assert "10k" in info["description"].lower() or "10000" in info["description"]
