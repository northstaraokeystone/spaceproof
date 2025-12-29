"""Tests for v6.0 scenarios (governance, training, privacy, offline)."""

import pytest
from spaceproof.sim.scenarios import (
    GovernanceConfig,
    GovernanceResult,
    GovernanceScenario,
    TrainingProductionConfig,
    TrainingProductionResult,
    TrainingProductionScenario,
    PrivacyEnforcementConfig,
    PrivacyEnforcementResult,
    PrivacyEnforcementScenario,
    OfflineResilienceConfig,
    OfflineResilienceResult,
    OfflineResilienceScenario,
)


class TestGovernanceScenario:
    """Tests for GovernanceScenario."""

    def test_config_defaults(self):
        """GovernanceConfig has sensible defaults."""
        config = GovernanceConfig()
        assert config.cycles == 500
        assert config.seed == 42
        assert config.decisions_to_inject == 50
        assert config.interventions_to_inject == 10
        assert config.policy_changes_to_inject == 5

    def test_scenario_runs(self):
        """GovernanceScenario runs without error."""
        config = GovernanceConfig(
            cycles=10,
            decisions_to_inject=5,
            interventions_to_inject=3,
        )
        scenario = GovernanceScenario(config)
        result = scenario.run()
        assert isinstance(result, GovernanceResult)

    def test_scenario_provenance(self):
        """GovernanceScenario tracks provenance."""
        config = GovernanceConfig(decisions_to_inject=10)
        scenario = GovernanceScenario(config)
        result = scenario.run()
        assert result.decisions_with_provenance == 10

    def test_scenario_raci(self):
        """GovernanceScenario assigns RACI."""
        config = GovernanceConfig(decisions_to_inject=10)
        scenario = GovernanceScenario(config)
        result = scenario.run()
        assert result.decisions_with_raci == 10

    def test_scenario_training_examples(self):
        """GovernanceScenario produces training examples."""
        config = GovernanceConfig(interventions_to_inject=10)
        scenario = GovernanceScenario(config)
        result = scenario.run()
        assert result.training_examples_produced >= 8

    def test_scenario_audit_slo(self):
        """GovernanceScenario meets audit generation SLO."""
        config = GovernanceConfig()
        scenario = GovernanceScenario(config)
        result = scenario.run()
        assert result.audit_generation_time_ms < 5000  # 5 second SLO


class TestTrainingProductionScenario:
    """Tests for TrainingProductionScenario."""

    def test_config_defaults(self):
        """TrainingProductionConfig has sensible defaults."""
        config = TrainingProductionConfig()
        assert config.cycles == 500
        assert config.decisions_to_inject == 100
        assert config.interventions_to_inject == 20
        assert config.critical_interventions == 5

    def test_scenario_runs(self):
        """TrainingProductionScenario runs without error."""
        config = TrainingProductionConfig(
            cycles=10,
            decisions_to_inject=10,
            interventions_to_inject=5,
        )
        scenario = TrainingProductionScenario(config)
        result = scenario.run()
        assert isinstance(result, TrainingProductionResult)

    def test_intervention_conversion(self):
        """Interventions convert to training examples."""
        config = TrainingProductionConfig(interventions_to_inject=10)
        scenario = TrainingProductionScenario(config)
        result = scenario.run()
        assert result.examples_created == result.interventions_processed

    def test_quality_threshold(self):
        """Quality scores meet threshold."""
        config = TrainingProductionConfig(interventions_to_inject=20)
        scenario = TrainingProductionScenario(config)
        result = scenario.run()
        # At least 80% above 0.8 quality
        total = sum(result.quality_score_distribution.values())
        high_quality = result.quality_score_distribution.get("high", 0)
        assert high_quality / total >= 0.8

    def test_critical_prioritization(self):
        """CRITICAL interventions are prioritized."""
        config = TrainingProductionConfig(critical_interventions=3)
        scenario = TrainingProductionScenario(config)
        result = scenario.run()
        assert result.critical_first is True

    def test_export_success(self):
        """Export to JSONL succeeds."""
        config = TrainingProductionConfig(interventions_to_inject=5)
        scenario = TrainingProductionScenario(config)
        result = scenario.run()
        assert result.export_successful is True


class TestPrivacyEnforcementScenario:
    """Tests for PrivacyEnforcementScenario."""

    def test_config_defaults(self):
        """PrivacyEnforcementConfig has sensible defaults."""
        config = PrivacyEnforcementConfig()
        assert config.cycles == 500
        assert config.receipts_with_pii == 100
        assert config.dp_requests == 50
        assert config.default_epsilon == 1.0
        assert config.initial_budget == 10.0

    def test_scenario_runs(self):
        """PrivacyEnforcementScenario runs without error."""
        config = PrivacyEnforcementConfig(
            cycles=10,
            receipts_with_pii=10,
            dp_requests=5,
        )
        scenario = PrivacyEnforcementScenario(config)
        result = scenario.run()
        assert isinstance(result, PrivacyEnforcementResult)

    def test_pii_redaction(self):
        """All PII is redacted."""
        config = PrivacyEnforcementConfig(receipts_with_pii=20)
        scenario = PrivacyEnforcementScenario(config)
        result = scenario.run()
        assert result.pii_redacted == result.pii_detected

    def test_redaction_receipts(self):
        """Redaction receipts are emitted."""
        config = PrivacyEnforcementConfig(receipts_with_pii=10)
        scenario = PrivacyEnforcementScenario(config)
        result = scenario.run()
        assert result.redaction_receipts_emitted > 0

    def test_dp_bounds(self):
        """DP noise is within bounds."""
        config = PrivacyEnforcementConfig(dp_requests=20)
        scenario = PrivacyEnforcementScenario(config)
        result = scenario.run()
        assert result.noise_within_bounds == result.dp_queries_processed

    def test_budget_enforcement(self):
        """Privacy budget is enforced."""
        config = PrivacyEnforcementConfig(
            initial_budget=5.0,
            budget_exhaustion_attempts=10,
        )
        scenario = PrivacyEnforcementScenario(config)
        result = scenario.run()
        assert result.budget_enforced is True

    def test_no_leakage(self):
        """No PII leakage detected."""
        config = PrivacyEnforcementConfig(receipts_with_pii=20)
        scenario = PrivacyEnforcementScenario(config)
        result = scenario.run()
        assert result.leakage_detected == 0


class TestOfflineResilienceScenario:
    """Tests for OfflineResilienceScenario."""

    def test_config_defaults(self):
        """OfflineResilienceConfig has sensible defaults."""
        config = OfflineResilienceConfig()
        assert config.cycles == 500
        assert config.network_partitions == 3
        assert config.receipts_per_partition == 50
        assert config.conflicting_receipts == 50

    def test_scenario_runs(self):
        """OfflineResilienceScenario runs without error."""
        config = OfflineResilienceConfig(
            cycles=10,
            network_partitions=2,
            receipts_per_partition=10,
            conflicting_receipts=5,
        )
        scenario = OfflineResilienceScenario(config)
        result = scenario.run()
        assert isinstance(result, OfflineResilienceResult)

    def test_receipt_preservation(self):
        """All offline receipts are preserved."""
        config = OfflineResilienceConfig(
            network_partitions=2,
            receipts_per_partition=20,
            conflicting_receipts=5,
        )
        scenario = OfflineResilienceScenario(config)
        result = scenario.run()
        assert result.data_loss == 0

    def test_conflict_resolution(self):
        """All conflicts are resolved."""
        config = OfflineResilienceConfig(conflicting_receipts=10)
        scenario = OfflineResilienceScenario(config)
        result = scenario.run()
        assert result.conflicts_resolved == result.conflicts_detected

    def test_merkle_integrity(self):
        """Merkle chain integrity is maintained."""
        config = OfflineResilienceConfig()
        scenario = OfflineResilienceScenario(config)
        result = scenario.run()
        assert result.merkle_integrity_maintained is True

    def test_sync_latency(self):
        """Sync latency is within bounds."""
        config = OfflineResilienceConfig()
        scenario = OfflineResilienceScenario(config)
        result = scenario.run()
        assert result.sync_latency_ok is True

    def test_partition_simulation(self):
        """Network partitions are simulated."""
        config = OfflineResilienceConfig(network_partitions=3)
        scenario = OfflineResilienceScenario(config)
        result = scenario.run()
        assert result.partitions_simulated == 3
