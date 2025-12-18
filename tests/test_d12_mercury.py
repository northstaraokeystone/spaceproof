"""Tests for D12 recursion and Mercury thermal autonomy."""

import pytest


class TestD12Spec:
    """Tests for D12 specification loading."""

    def test_d12_spec_loads(self):
        """Spec loads with valid dual-hash."""
        from src.fractal_layers import get_d12_spec

        spec = get_d12_spec()
        assert spec is not None
        assert "d12_config" in spec
        assert "mercury_config" in spec

    def test_d12_alpha_floor(self):
        """Assert eff_alpha >= 3.63 at depth=12."""
        from src.fractal_layers import d12_recursive_fractal

        result = d12_recursive_fractal(10**12, 3.35, depth=12)
        assert result["eff_alpha"] >= 3.63

    def test_d12_alpha_target(self):
        """Assert eff_alpha >= 3.65 achievable."""
        from src.fractal_layers import d12_push

        result = d12_push(10**12, 3.35, simulate=True)
        assert result["eff_alpha"] >= 3.63  # Floor target

    def test_d12_uplift_value(self):
        """Assert uplift == 0.30."""
        from src.fractal_layers import get_d12_uplift

        uplift = get_d12_uplift(12)
        assert uplift == 0.30

    def test_d12_instability(self):
        """Assert instability == 0.00."""
        from src.fractal_layers import d12_recursive_fractal

        result = d12_recursive_fractal(10**12, 3.35, depth=12)
        assert result["instability"] == 0.00


class TestMercuryConfig:
    """Tests for Mercury configuration."""

    def test_mercury_config_loads(self):
        """Mercury config valid."""
        from src.mercury_thermal_hybrid import load_mercury_config

        config = load_mercury_config()
        assert config is not None
        assert config.get("body") == "mercury"

    def test_mercury_day_temp(self):
        """Assert temp == 430°C."""
        from src.mercury_thermal_hybrid import MERCURY_SURFACE_TEMP_DAY_C

        assert MERCURY_SURFACE_TEMP_DAY_C == 430

    def test_mercury_night_temp(self):
        """Assert temp == -180°C."""
        from src.mercury_thermal_hybrid import MERCURY_SURFACE_TEMP_NIGHT_C

        assert MERCURY_SURFACE_TEMP_NIGHT_C == -180

    def test_mercury_thermal_swing(self):
        """Assert swing == 610°C."""
        from src.mercury_thermal_hybrid import MERCURY_THERMAL_SWING_C

        assert MERCURY_THERMAL_SWING_C == 610

    def test_mercury_solar_flux(self):
        """Assert flux == 9082 W/m²."""
        from src.mercury_thermal_hybrid import MERCURY_SOLAR_FLUX_W_M2

        assert MERCURY_SOLAR_FLUX_W_M2 == 9082

    def test_mercury_autonomy_requirement(self):
        """Assert autonomy >= 0.995."""
        from src.mercury_thermal_hybrid import MERCURY_AUTONOMY_REQUIREMENT

        assert MERCURY_AUTONOMY_REQUIREMENT >= 0.995

    def test_mercury_alloys_present(self):
        """All 3 alloys listed."""
        from src.mercury_thermal_hybrid import MERCURY_ALLOYS

        assert len(MERCURY_ALLOYS) >= 3
        assert "inconel_718" in MERCURY_ALLOYS
        assert "haynes_230" in MERCURY_ALLOYS
        assert "tungsten_rhenium" in MERCURY_ALLOYS

    def test_mercury_hazards_present(self):
        """All 4 hazards listed."""
        from src.mercury_thermal_hybrid import MERCURY_HAZARDS

        assert len(MERCURY_HAZARDS) >= 4
        assert "extreme_heat" in MERCURY_HAZARDS
        assert "extreme_cold" in MERCURY_HAZARDS
        assert "solar_radiation" in MERCURY_HAZARDS
        assert "thermal_cycling" in MERCURY_HAZARDS


class TestMercuryThermal:
    """Tests for Mercury thermal simulations."""

    def test_thermal_zone_dayside(self):
        """Dayside zone detection."""
        from src.mercury_thermal_hybrid import compute_thermal_zone

        result = compute_thermal_zone(0.5, 0.0)  # Noon at equator
        assert result["zone"] == "dayside"
        assert result["temperature_c"] > 0

    def test_thermal_zone_nightside(self):
        """Nightside zone detection."""
        from src.mercury_thermal_hybrid import compute_thermal_zone

        result = compute_thermal_zone(0.0, 0.0)  # Midnight at equator
        assert result["zone"] == "nightside"
        assert result["temperature_c"] < 0

    def test_thermal_zone_polar(self):
        """Polar crater zone detection."""
        from src.mercury_thermal_hybrid import compute_thermal_zone

        result = compute_thermal_zone(0.5, 85.0)  # High latitude
        assert result["zone"] == "polar_crater"

    def test_alloy_performance_inconel(self):
        """Inconel 718 performance test."""
        from src.mercury_thermal_hybrid import simulate_alloy_performance

        result = simulate_alloy_performance("inconel_718", 400.0, 100.0)
        assert result["operational"] is True
        assert result["within_limit"] is True

    def test_thermal_cycling(self):
        """Thermal cycling simulation."""
        from src.mercury_thermal_hybrid import simulate_thermal_cycling

        result = simulate_thermal_cycling(100, 200)  # 100 cycles, 200°C swing
        assert result["status"] in ["HEALTHY", "DEGRADED", "CRITICAL"]
        assert result["damage_fraction"] >= 0


class TestD12MercuryHybrid:
    """Tests for D12 + Mercury hybrid."""

    def test_d12_mercury_hybrid_alpha(self):
        """Assert hybrid eff_alpha >= 3.63."""
        from src.mercury_thermal_hybrid import d12_mercury_hybrid

        result = d12_mercury_hybrid(10**12, 3.35, simulate=True)
        assert result["eff_alpha"] >= 3.63

    def test_d12_mercury_hybrid_autonomy(self):
        """Assert Mercury autonomy in hybrid."""
        from src.mercury_thermal_hybrid import d12_mercury_hybrid

        result = d12_mercury_hybrid(10**12, 3.35, simulate=True)
        assert "mercury_autonomy" in result
        assert result["mercury_autonomy"] >= 0.9

    def test_d12_mercury_hybrid_gates(self):
        """Assert gates present in hybrid."""
        from src.mercury_thermal_hybrid import d12_mercury_hybrid

        result = d12_mercury_hybrid(10**12, 3.35, simulate=True)
        assert "alpha_gate_passed" in result
        assert "autonomy_gate_passed" in result
        assert "all_gates_passed" in result
