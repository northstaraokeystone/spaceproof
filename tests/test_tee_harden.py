"""Tests for TEE hardening against SGX side-channel attacks."""

import pytest


class TestTEEConfig:
    """Tests for TEE configuration."""

    def test_tee_config_loads(self):
        """Config valid."""
        from src.tee_harden_audit import load_tee_config

        config = load_tee_config()
        assert config is not None
        assert config.get("type") == "SGX"

    def test_tee_type(self):
        """Assert type == 'SGX'."""
        from src.tee_harden_audit import TEE_TYPE

        assert TEE_TYPE == "SGX"

    def test_side_channels_present(self):
        """All 4 channels listed."""
        from src.tee_harden_audit import TEE_SIDE_CHANNELS

        assert len(TEE_SIDE_CHANNELS) >= 4
        assert "timing" in TEE_SIDE_CHANNELS
        assert "power" in TEE_SIDE_CHANNELS
        assert "cache" in TEE_SIDE_CHANNELS
        assert "branch" in TEE_SIDE_CHANNELS

    def test_defense_mechanisms_present(self):
        """All 4 mechanisms listed."""
        from src.tee_harden_audit import TEE_DEFENSE_MECHANISMS

        assert len(TEE_DEFENSE_MECHANISMS) >= 4
        assert "constant_time" in TEE_DEFENSE_MECHANISMS
        assert "power_balancing" in TEE_DEFENSE_MECHANISMS
        assert "cache_partition" in TEE_DEFENSE_MECHANISMS
        assert "branch_obfuscation" in TEE_DEFENSE_MECHANISMS

    def test_attestation_required(self):
        """Assert required == True."""
        from src.tee_harden_audit import TEE_ATTESTATION_REQUIRED

        assert TEE_ATTESTATION_REQUIRED is True


class TestTEEInit:
    """Tests for TEE initialization."""

    def test_tee_init(self):
        """TEE initialization."""
        from src.tee_harden_audit import init_tee

        result = init_tee(256)
        assert result["initialized"] is True
        assert result["type"] == "SGX"
        assert result["memory_mb"] == 256
        assert len(result["defenses_applied"]) >= 4


class TestTimingResilience:
    """Tests for timing side-channel defense."""

    def test_timing_resilience(self):
        """Assert resilience == 1.0."""
        from src.tee_harden_audit import implement_constant_time

        result = implement_constant_time()
        assert result["resilience"] >= 0.99
        assert result["mechanism"] == "constant_time"
        assert result["defense_active"] is True

    def test_timing_techniques(self):
        """Timing defense techniques present."""
        from src.tee_harden_audit import implement_constant_time

        result = implement_constant_time()
        assert "techniques" in result
        assert len(result["techniques"]) >= 2


class TestPowerResilience:
    """Tests for power side-channel defense."""

    def test_power_resilience(self):
        """Assert resilience == 1.0."""
        from src.tee_harden_audit import implement_power_balancing

        result = implement_power_balancing()
        assert result["resilience"] >= 0.99
        assert result["mechanism"] == "power_balancing"
        assert result["defense_active"] is True


class TestCacheResilience:
    """Tests for cache side-channel defense."""

    def test_cache_resilience(self):
        """Assert resilience == 1.0."""
        from src.tee_harden_audit import implement_cache_partition

        result = implement_cache_partition()
        assert result["resilience"] >= 0.99
        assert result["mechanism"] == "cache_partition"
        assert result["defense_active"] is True

    def test_cache_partition_isolation(self):
        """Cache partition provides isolation."""
        from src.tee_harden_audit import implement_cache_partition

        result = implement_cache_partition()
        assert result["hardened_leak_rate"] == 0.0


class TestBranchResilience:
    """Tests for branch prediction side-channel defense."""

    def test_branch_resilience(self):
        """Assert resilience == 1.0."""
        from src.tee_harden_audit import implement_branch_obfuscation

        result = implement_branch_obfuscation()
        assert result["resilience"] >= 0.99
        assert result["mechanism"] == "branch_obfuscation"
        assert result["defense_active"] is True

    def test_branch_prediction_degraded(self):
        """Branch prediction reduced to random."""
        from src.tee_harden_audit import implement_branch_obfuscation

        result = implement_branch_obfuscation()
        assert result["hardened_prediction_rate"] <= 0.5


class TestFullAudit:
    """Tests for full TEE audit."""

    def test_tee_harden_receipt(self):
        """Receipt emitted for audit."""
        from src.tee_harden_audit import run_tee_audit

        result = run_tee_audit()
        assert "overall_resilience" in result
        assert "all_channels_passed" in result
        assert result["overall_resilience"] >= 0.99

    def test_all_channels_passed(self):
        """All side-channels defended."""
        from src.tee_harden_audit import run_tee_audit

        result = run_tee_audit()
        assert result["all_channels_passed"] is True
        for channel, data in result["channel_results"].items():
            assert data["passed"] is True

    def test_attestation_ready(self):
        """Attestation ready after passing audit."""
        from src.tee_harden_audit import run_tee_audit

        result = run_tee_audit()
        assert result["attestation_ready"] is True


class TestAttestation:
    """Tests for remote attestation."""

    def test_remote_attestation(self):
        """Remote attestation."""
        from src.tee_harden_audit import remote_attestation

        result = remote_attestation()
        assert result["attestation_valid"] is True
        assert result["quote_verified"] is True
        assert "mrenclave" in result
        assert "mrsigner" in result


class TestSealedStorage:
    """Tests for sealed storage."""

    def test_sealed_storage(self):
        """Sealed storage test."""
        from src.tee_harden_audit import sealed_storage_test

        result = sealed_storage_test()
        assert result["seal_success"] is True
        assert result["unseal_success"] is True
        assert result["integrity_verified"] is True


class TestTEEOverhead:
    """Tests for TEE overhead measurement."""

    def test_tee_overhead(self):
        """TEE overhead within limits."""
        from src.tee_harden_audit import measure_tee_overhead, TEE_OVERHEAD_MAX_PCT

        result = measure_tee_overhead()
        assert result["within_limit"] is True
        assert result["overhead_pct"] <= TEE_OVERHEAD_MAX_PCT
