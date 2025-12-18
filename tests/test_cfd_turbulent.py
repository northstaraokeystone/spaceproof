"""Tests for turbulent CFD Navier-Stokes implementation."""

import pytest


class TestCFDTurbulentConfig:
    """Tests for turbulent CFD configuration."""

    def test_cfd_turbulent_config(self):
        """Turbulent config valid."""
        from src.cfd_dust_dynamics import load_turbulent_cfd_config

        config = load_turbulent_cfd_config()
        assert config is not None
        assert config.get("model") == "navier_stokes_turbulent"

    def test_reynolds_threshold(self):
        """Assert threshold == 2300."""
        from src.cfd_dust_dynamics import CFD_REYNOLDS_TURBULENT_THRESHOLD

        assert CFD_REYNOLDS_TURBULENT_THRESHOLD == 2300

    def test_reynolds_mars_storm(self):
        """Assert Re == 5000 for Mars storm."""
        from src.cfd_dust_dynamics import CFD_REYNOLDS_MARS_TURBULENT

        assert CFD_REYNOLDS_MARS_TURBULENT == 5000

    def test_turbulence_model(self):
        """Assert model == 'k_epsilon'."""
        from src.cfd_dust_dynamics import CFD_TURBULENCE_MODEL_KEPS

        assert CFD_TURBULENCE_MODEL_KEPS == "k_epsilon"


class TestFlowTransition:
    """Tests for flow regime transition detection."""

    def test_transition_laminar(self):
        """Re < 2300 -> 'laminar'."""
        from src.cfd_dust_dynamics import transition_check

        regime = transition_check(50)
        assert regime == "laminar"

        regime = transition_check(2000)
        assert regime == "laminar"

    def test_transition_transitional(self):
        """2300 <= Re < 4000 -> 'transitional'."""
        from src.cfd_dust_dynamics import transition_check

        regime = transition_check(2500)
        assert regime == "transitional"

        regime = transition_check(3500)
        assert regime == "transitional"

    def test_transition_turbulent(self):
        """Re >= 4000 -> 'turbulent'."""
        from src.cfd_dust_dynamics import transition_check

        regime = transition_check(5000)
        assert regime == "turbulent"

        regime = transition_check(10000)
        assert regime == "turbulent"


class TestKEpsilonModel:
    """Tests for k-epsilon turbulence model."""

    def test_eddy_viscosity_calculation(self):
        """Physics correct for eddy viscosity."""
        from src.cfd_dust_dynamics import compute_turbulent_viscosity

        # k=0.1, epsilon=0.1 should give nu_t = 0.09 * 0.01 / 0.1 = 0.009
        nu_t = compute_turbulent_viscosity(0.1, 0.1)
        assert nu_t > 0
        assert nu_t == pytest.approx(0.009, rel=0.01)

    def test_k_epsilon_closure(self):
        """Model produces valid output."""
        from src.cfd_dust_dynamics import k_epsilon_closure

        result = k_epsilon_closure(10.0, 1.0)
        assert "k_m2_s2" in result
        assert "epsilon_m2_s3" in result
        assert "nu_t_m2_s" in result
        assert result["k_m2_s2"] > 0
        assert result["epsilon_m2_s3"] > 0

    def test_k_epsilon_constants(self):
        """Model constants present."""
        from src.cfd_dust_dynamics import k_epsilon_closure

        result = k_epsilon_closure(10.0, 1.0)
        assert "constants" in result
        assert result["constants"]["c_mu"] == 0.09


class TestTurbulentSimulation:
    """Tests for turbulent flow simulation."""

    def test_simulate_turbulent_laminar(self):
        """Laminar regime simulation."""
        from src.cfd_dust_dynamics import simulate_turbulent

        result = simulate_turbulent(reynolds=50, duration_s=100)
        assert result["regime"] == "laminar"
        assert result["is_turbulent"] is False
        assert result["turbulence_model"] == "laminar"

    def test_simulate_turbulent_turbulent(self):
        """Turbulent regime simulation."""
        from src.cfd_dust_dynamics import simulate_turbulent

        result = simulate_turbulent(reynolds=5000, duration_s=100)
        assert result["regime"] == "turbulent"
        assert result["is_turbulent"] is True
        assert result["turbulence_model"] == "k_epsilon"

    def test_simulate_turbulent_settling(self):
        """Settling velocity computation."""
        from src.cfd_dust_dynamics import simulate_turbulent

        result = simulate_turbulent(reynolds=5000, duration_s=100, particle_size_um=10)
        assert "base_settling_m_s" in result
        assert "effective_settling_m_s" in result
        assert result["base_settling_m_s"] > 0


class TestTurbulentDustStorm:
    """Tests for turbulent dust storm simulation."""

    def test_dust_storm_turbulent(self):
        """Turbulent dust storm simulation."""
        from src.cfd_dust_dynamics import dust_storm_turbulent

        result = dust_storm_turbulent(intensity=0.8, duration_hrs=24)
        assert result["regime"] in ["transitional", "turbulent"]
        assert result["turbulence_model"] in ["laminar", "k_epsilon"]
        assert "dispersion_m2_s" in result

    def test_dust_storm_high_intensity(self):
        """High intensity storm is turbulent."""
        from src.cfd_dust_dynamics import dust_storm_turbulent

        result = dust_storm_turbulent(intensity=1.0, duration_hrs=24)
        assert result["reynolds"] > 2300  # Should be turbulent

    def test_cfd_turbulent_validation(self):
        """Full turbulent validation."""
        from src.cfd_dust_dynamics import run_turbulent_validation

        result = run_turbulent_validation()
        assert result["all_regimes_validated"] is True
        assert result["laminar_test"]["correct"] is True
        assert result["transitional_test"]["correct"] is True
        assert result["turbulent_test"]["correct"] is True


class TestCFDTurbulentReceipt:
    """Tests for CFD turbulent receipts."""

    def test_cfd_turbulent_receipt(self):
        """Receipt emitted for turbulent simulation."""
        from src.cfd_dust_dynamics import simulate_turbulent

        result = simulate_turbulent(reynolds=5000, duration_s=100)
        assert result["validated"] is True
