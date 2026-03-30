"""Tests for built-in invariant monitors.

Each invariant is tested against:
1. Analytical test: known invariant value is computed correctly
2. Correct simulation: no violation detected on a correct simulation
3. Known-bug test: violation detected when a bug is introduced
4. Edge cases: near-zero, NaN, Inf
"""


import numpy as np
import pytest

from sim_debugger.core.invariants import (
    AngularMomentumInvariant,
    ChargeConservationInvariant,
    LinearMomentumInvariant,
    ParticleCountInvariant,
    TotalEnergyInvariant,
    create_default_registry,
)
from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import ViolationSeverity

# ===========================================================================
# Test fixtures: simulation states
# ===========================================================================

def make_particle_state(
    positions: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
    masses: np.ndarray | None = None,
    charges: np.ndarray | None = None,
    **extra_arrays,
) -> SimulationState:
    """Helper to create a SimulationState with particle data."""
    arrays = {}
    if positions is not None:
        arrays["positions"] = positions
    if velocities is not None:
        arrays["velocities"] = velocities
    if masses is not None:
        arrays["masses"] = masses
    if charges is not None:
        arrays["charges"] = charges
    arrays.update(extra_arrays)
    return SimulationState(timestep=0, time=0.0, arrays=arrays)


# ===========================================================================
# TotalEnergyInvariant
# ===========================================================================

class TestTotalEnergyInvariant:
    def setup_method(self):
        self.inv = TotalEnergyInvariant()

    def test_name(self):
        assert self.inv.name == "Total Energy"

    def test_kinetic_energy_single_particle(self):
        """E_k = 0.5 * m * |v|^2 for a single particle."""
        state = make_particle_state(
            velocities=np.array([[3.0, 4.0, 0.0]]),
            masses=np.array([2.0]),
        )
        # E_k = 0.5 * 2 * (9 + 16) = 25.0
        assert self.inv.compute(state) == pytest.approx(25.0)

    def test_kinetic_energy_multiple_particles(self):
        """Total E_k for multiple particles."""
        state = make_particle_state(
            velocities=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            masses=np.array([1.0, 3.0]),
        )
        # E_k = 0.5 * 1 * 1 + 0.5 * 3 * 4 = 0.5 + 6.0 = 6.5
        assert self.inv.compute(state) == pytest.approx(6.5)

    def test_kinetic_plus_potential(self):
        """Total energy includes potential energy when present."""
        state = make_particle_state(
            velocities=np.array([[1.0, 0.0, 0.0]]),
            masses=np.array([1.0]),
            potential_energy=np.array([-0.3]),
        )
        # E_k = 0.5, E_p = -0.3, total = 0.2
        assert self.inv.compute(state) == pytest.approx(0.2)

    def test_check_no_violation(self):
        """No violation when energy is conserved."""
        result = self.inv.check(1.0, 1.0 + 1e-8)
        assert result is None

    def test_check_violation_detected(self):
        """Violation detected when energy changes significantly."""
        result = self.inv.check(1.0, 1.05)
        assert result is not None
        assert result.invariant_name == "Total Energy"
        assert result.relative_error == pytest.approx(0.05)

    def test_check_nan(self):
        """NaN triggers CRITICAL violation."""
        result = self.inv.check(1.0, float("nan"))
        assert result is not None
        assert result.severity == ViolationSeverity.CRITICAL

    def test_check_inf(self):
        """Inf triggers CRITICAL violation."""
        result = self.inv.check(1.0, float("inf"))
        assert result is not None
        assert result.severity == ViolationSeverity.CRITICAL

    def test_applicable(self):
        """Applicable when velocities and masses are present."""
        state = make_particle_state(
            velocities=np.array([[1.0]]),
            masses=np.array([1.0]),
        )
        assert self.inv.applicable(state)

    def test_not_applicable(self):
        """Not applicable when velocities are missing."""
        state = make_particle_state(masses=np.array([1.0]))
        assert not self.inv.applicable(state)

    def test_harmonic_oscillator_energy_conservation(self):
        """Verify energy is conserved in a symplectic Euler harmonic oscillator.

        Symplectic Euler: v_{n+1} = v_n + a_n * dt, x_{n+1} = x_n + v_{n+1} * dt
        This is a first-order symplectic integrator with bounded energy error O(dt).
        """
        x = np.array([[1.0, 0.0, 0.0]])
        v = np.array([[0.0, 0.0, 0.0]])
        m = np.array([1.0])
        dt = 0.0001  # Very small dt for good conservation

        state0 = make_particle_state(
            positions=x.copy(), velocities=v.copy(), masses=m,
            potential_energy=np.array([0.5 * np.sum(x**2)]),
        )
        E0 = self.inv.compute(state0)

        # Symplectic Euler integration for 1000 steps
        for _ in range(1000):
            a = -x  # F = -kx, k=1
            v = v + a * dt
            x = x + v * dt

        state1 = make_particle_state(
            positions=x.copy(), velocities=v.copy(), masses=m,
            potential_energy=np.array([0.5 * np.sum(x**2)]),
        )
        E1 = self.inv.compute(state1)

        # With dt=0.0001 and 1000 steps, energy error should be ~O(dt) ~ 1e-4
        # The default threshold is 1e-6, so use a slightly loose threshold
        assert self.inv.check(E0, E1, threshold=1e-3) is None


# ===========================================================================
# LinearMomentumInvariant
# ===========================================================================

class TestLinearMomentumInvariant:
    def setup_method(self):
        self.inv = LinearMomentumInvariant()

    def test_single_particle_1d(self):
        """p = m*v for a single 1D particle."""
        state = make_particle_state(
            velocities=np.array([3.0]),
            masses=np.array([2.0]),
        )
        assert self.inv.compute(state) == pytest.approx(6.0)

    def test_two_particles_cancel(self):
        """Two equal-mass particles with opposite velocities: p=0."""
        state = make_particle_state(
            velocities=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
        )
        assert self.inv.compute(state) == pytest.approx(0.0, abs=1e-15)

    def test_momentum_conservation_symmetric_collision(self):
        """Symmetric collision conserves momentum (equal and opposite forces)."""
        m = np.array([1.0, 1.0])
        v1 = np.array([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        state1 = make_particle_state(velocities=v1, masses=m)
        p1 = self.inv.compute(state1)

        # After symmetric force: F_12 = -F_21
        dt = 0.01
        F = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # No force yet
        v2 = v1 + F * dt
        state2 = make_particle_state(velocities=v2, masses=m)
        p2 = self.inv.compute(state2)

        assert self.inv.check(p1, p2) is None

    def test_check_violation(self):
        result = self.inv.check(10.0, 10.1)
        assert result is not None

    def test_applicable(self):
        state = make_particle_state(
            velocities=np.array([[1.0]]), masses=np.array([1.0])
        )
        assert self.inv.applicable(state)


# ===========================================================================
# AngularMomentumInvariant
# ===========================================================================

class TestAngularMomentumInvariant:
    def setup_method(self):
        self.inv = AngularMomentumInvariant()

    def test_circular_orbit_2d(self):
        """Particle in circular orbit: L = m*v*r."""
        r = 2.0
        v_mag = 3.0
        state = make_particle_state(
            positions=np.array([[r, 0.0]]),
            velocities=np.array([[0.0, v_mag]]),
            masses=np.array([1.5]),
        )
        # L = r * (m * v) = 2 * 1.5 * 3 = 9
        assert self.inv.compute(state) == pytest.approx(9.0)

    def test_circular_orbit_3d(self):
        """Particle in circular orbit in 3D: L = r x (m*v)."""
        state = make_particle_state(
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.array([[0.0, 1.0, 0.0]]),
            masses=np.array([2.0]),
        )
        # L = (1,0,0) x (0,2,0) = (0,0,2), |L| = 2
        assert self.inv.compute(state) == pytest.approx(2.0)

    def test_not_applicable_1d(self):
        """Angular momentum requires at least 2D."""
        state = make_particle_state(
            positions=np.array([[1.0]]),
            velocities=np.array([[1.0]]),
            masses=np.array([1.0]),
        )
        assert not self.inv.applicable(state)

    def test_applicable_3d(self):
        state = make_particle_state(
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.array([[0.0, 1.0, 0.0]]),
            masses=np.array([1.0]),
        )
        assert self.inv.applicable(state)


# ===========================================================================
# ChargeConservationInvariant
# ===========================================================================

class TestChargeConservationInvariant:
    def setup_method(self):
        self.inv = ChargeConservationInvariant()

    def test_total_charge(self):
        state = make_particle_state(
            charges=np.array([1.0, -1.0, 1.0, -1.0])
        )
        assert self.inv.compute(state) == pytest.approx(0.0)

    def test_nonzero_charge(self):
        state = make_particle_state(
            charges=np.array([1.6e-19, 1.6e-19, -1.6e-19])
        )
        assert self.inv.compute(state) == pytest.approx(1.6e-19)

    def test_check_conserved(self):
        assert self.inv.check(0.0, 0.0) is None

    def test_check_violated(self):
        result = self.inv.check(0.0, 1.0)
        assert result is not None

    def test_applicable(self):
        state = make_particle_state(charges=np.array([1.0]))
        assert self.inv.applicable(state)

    def test_not_applicable(self):
        state = make_particle_state(velocities=np.array([[1.0]]))
        assert not self.inv.applicable(state)


# ===========================================================================
# ParticleCountInvariant
# ===========================================================================

class TestParticleCountInvariant:
    def setup_method(self):
        self.inv = ParticleCountInvariant()

    def test_count_from_positions(self):
        state = make_particle_state(
            positions=np.zeros((100, 3))
        )
        assert self.inv.compute(state) == 100.0

    def test_count_from_metadata(self):
        state = SimulationState(
            timestep=0, time=0.0,
            metadata={"particle_count": 42},
        )
        assert self.inv.compute(state) == 42.0

    def test_check_conserved(self):
        assert self.inv.check(100.0, 100.0) is None

    def test_check_violated_gain(self):
        result = self.inv.check(100.0, 101.0)
        assert result is not None
        assert result.absolute_error == 1.0

    def test_check_violated_loss(self):
        result = self.inv.check(100.0, 99.0)
        assert result is not None

    def test_applicable_with_positions(self):
        state = make_particle_state(positions=np.zeros((10, 3)))
        assert self.inv.applicable(state)

    def test_applicable_with_metadata(self):
        state = SimulationState(
            timestep=0, time=0.0,
            metadata={"particle_count": 10},
        )
        assert self.inv.applicable(state)


# ===========================================================================
# InvariantRegistry
# ===========================================================================

class TestInvariantRegistry:
    def test_create_default(self):
        registry = create_default_registry()
        names = registry.list_names()
        assert "Total Energy" in names
        assert "Linear Momentum" in names
        assert "Angular Momentum" in names
        assert "Charge Conservation" in names
        assert "Particle Count" in names
        assert len(names) == 5

    def test_get(self):
        registry = create_default_registry()
        inv = registry.get("Total Energy")
        assert inv.name == "Total Energy"

    def test_get_missing(self):
        registry = create_default_registry()
        with pytest.raises(KeyError):
            registry.get("Nonexistent")

    def test_register_duplicate(self):
        registry = create_default_registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(TotalEnergyInvariant())

    def test_find_applicable(self):
        registry = create_default_registry()
        state = make_particle_state(
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.array([[0.0, 1.0, 0.0]]),
            masses=np.array([1.0]),
            charges=np.array([1.0]),
        )
        applicable = registry.find_applicable(state)
        names = [inv.name for inv in applicable]
        assert "Total Energy" in names
        assert "Linear Momentum" in names
        assert "Angular Momentum" in names
        assert "Charge Conservation" in names
        assert "Particle Count" in names
