"""Tests for chaotic n-body gravitational simulations."""


class TestChaoticNBodyConfig:
    """Tests for chaotic n-body configuration."""

    def test_load_chaos_config(self) -> None:
        """Test loading chaos configuration."""
        from src.chaotic_nbody_sim import load_chaos_config

        config = load_chaos_config()
        assert config is not None
        assert "body_count" in config
        assert config["body_count"] == 7
        assert "integration_method" in config
        assert config["integration_method"] == "symplectic"
        assert "lyapunov_threshold" in config
        assert config["lyapunov_threshold"] == 0.1

    def test_chaos_constants(self) -> None:
        """Test chaos constants are correctly defined."""
        from src.chaotic_nbody_sim import (
            NBODY_COUNT,
            LYAPUNOV_EXPONENT_THRESHOLD,
            CHAOTIC_STABILITY_TARGET,
            SYMPLECTIC_INTEGRATION,
        )

        assert NBODY_COUNT == 7
        assert LYAPUNOV_EXPONENT_THRESHOLD == 0.1
        assert CHAOTIC_STABILITY_TARGET == 0.95
        assert SYMPLECTIC_INTEGRATION is True


class TestBodyInitialization:
    """Tests for n-body initialization."""

    def test_initialize_bodies(self) -> None:
        """Test body initialization."""
        from src.chaotic_nbody_sim import initialize_bodies

        bodies = initialize_bodies()
        assert bodies is not None
        assert len(bodies) == 7

        for body in bodies:
            assert "name" in body
            assert "mass" in body
            assert "position" in body
            assert "velocity" in body

    def test_body_mass_distribution(self) -> None:
        """Test that bodies have realistic mass distribution."""
        from src.chaotic_nbody_sim import initialize_bodies

        bodies = initialize_bodies()
        masses = [b["mass"] for b in bodies]

        # Sun should be most massive
        assert max(masses) > sum(masses[1:])

    def test_body_positions(self) -> None:
        """Test that bodies have 3D positions."""
        from src.chaotic_nbody_sim import initialize_bodies

        bodies = initialize_bodies()

        for body in bodies:
            pos = body["position"]
            assert len(pos) == 3
            assert all(isinstance(p, (int, float)) for p in pos)


class TestGravitationalForces:
    """Tests for gravitational force computation."""

    def test_compute_forces(self) -> None:
        """Test gravitational force computation."""
        from src.chaotic_nbody_sim import (
            initialize_bodies,
            compute_gravitational_forces,
        )

        bodies = initialize_bodies()
        forces = compute_gravitational_forces(bodies)

        assert forces is not None
        assert len(forces) == len(bodies)

        for force in forces:
            assert len(force) == 3  # 3D force vector

    def test_force_symmetry(self) -> None:
        """Test that forces are symmetric (Newton's third law)."""
        from src.chaotic_nbody_sim import (
            initialize_bodies,
            compute_gravitational_forces,
        )

        bodies = initialize_bodies()
        forces = compute_gravitational_forces(bodies)

        # Sum of all forces should be approximately zero
        total_force = [0.0, 0.0, 0.0]
        for force in forces:
            for i in range(3):
                total_force[i] += force[i]

        for component in total_force:
            assert abs(component) < 1e-10


class TestSymplecticIntegration:
    """Tests for symplectic integration."""

    def test_symplectic_integrate(self) -> None:
        """Test symplectic integration step."""
        from src.chaotic_nbody_sim import (
            initialize_bodies,
            symplectic_integrate,
        )

        bodies = initialize_bodies()
        dt = 0.001

        new_bodies = symplectic_integrate(bodies, dt)

        assert new_bodies is not None
        assert len(new_bodies) == len(bodies)

    def test_energy_conservation(self) -> None:
        """Test that symplectic integration conserves energy."""
        from src.chaotic_nbody_sim import (
            initialize_bodies,
            symplectic_integrate,
            compute_total_energy,
        )

        bodies = initialize_bodies()
        initial_energy = compute_total_energy(bodies)

        # Integrate for a few steps
        dt = 0.001
        for _ in range(100):
            bodies = symplectic_integrate(bodies, dt)

        final_energy = compute_total_energy(bodies)

        # Energy drift should be small
        drift = abs(final_energy - initial_energy) / abs(initial_energy)
        assert drift < 0.01  # Less than 1% drift


class TestLyapunovExponent:
    """Tests for Lyapunov exponent computation."""

    def test_compute_lyapunov(self) -> None:
        """Test Lyapunov exponent computation."""
        from src.chaotic_nbody_sim import compute_lyapunov_exponent

        result = compute_lyapunov_exponent(iterations=100)

        assert result is not None
        assert "lyapunov_exponent" in result
        assert "threshold" in result
        assert "is_stable" in result
        assert result["threshold"] == 0.1

    def test_lyapunov_stability_check(self) -> None:
        """Test stability determination from Lyapunov exponent."""
        from src.chaotic_nbody_sim import check_stability

        result = check_stability()

        assert result is not None
        assert "is_stable" in result
        assert "lyapunov_exponent" in result
        assert "stability_margin" in result


class TestSimulation:
    """Tests for full chaos simulation."""

    def test_simulate_chaos(self) -> None:
        """Test full chaos simulation."""
        from src.chaotic_nbody_sim import simulate_chaos

        result = simulate_chaos(iterations=100, dt=0.001, simulate=True)

        assert result is not None
        assert "mode" in result
        assert result["mode"] == "simulate"
        assert "iterations" in result
        assert "body_count" in result
        assert "lyapunov_exponent" in result
        assert "is_stable" in result
        assert "energy_conserved" in result

    def test_simulate_chaos_execute(self) -> None:
        """Test chaos simulation in execute mode."""
        from src.chaotic_nbody_sim import simulate_chaos

        result = simulate_chaos(iterations=100, dt=0.001, simulate=False)

        assert result is not None
        assert result["mode"] == "execute"
        assert result["body_count"] == 7

    def test_monte_carlo_stability(self) -> None:
        """Test Monte Carlo stability analysis."""
        from src.chaotic_nbody_sim import run_monte_carlo_stability

        result = run_monte_carlo_stability(runs=10, simulate=True)

        assert result is not None
        assert "runs" in result
        assert "stable_runs" in result
        assert "unstable_runs" in result
        assert "stability_rate" in result
        assert "stability_target" in result
        assert "target_met" in result


class TestBackboneIntegration:
    """Tests for backbone chaos tolerance integration."""

    def test_compute_backbone_tolerance(self) -> None:
        """Test backbone chaos tolerance computation."""
        from src.chaotic_nbody_sim import compute_backbone_chaos_tolerance

        result = compute_backbone_chaos_tolerance()

        assert result is not None
        assert "tolerance" in result
        assert "lyapunov_exponent" in result
        assert "stability" in result
        assert "backbone_compatible" in result

    def test_tolerance_range(self) -> None:
        """Test that tolerance is in valid range."""
        from src.chaotic_nbody_sim import compute_backbone_chaos_tolerance

        result = compute_backbone_chaos_tolerance()

        tolerance = result["tolerance"]
        assert 0.0 <= tolerance <= 1.0
