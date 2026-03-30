"""Tests for PARSEC-specific invariant monitors.

Tests Boris energy conservation, Gauss's law, and Lorentz force
correctness against analytically known simulation scenarios.
"""

import numpy as np
import pytest

from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import ViolationSeverity
from sim_debugger.parsec.invariants import (
    BorisEnergyInvariant,
    GaussLawInvariant,
    LorentzForceInvariant,
)


def make_state(**kwargs) -> SimulationState:
    """Helper to create a SimulationState."""
    arrays = {}
    metadata = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            metadata[k] = v
    return SimulationState(timestep=0, time=0.0, arrays=arrays, metadata=metadata)


# ===========================================================================
# BorisEnergyInvariant
# ===========================================================================

class TestBorisEnergyInvariant:
    def setup_method(self):
        self.inv = BorisEnergyInvariant()

    def test_zero_residual_first_call(self):
        """First call returns 0.0 (no previous state to compare)."""
        state = make_state(
            velocities=np.array([[3.0, 4.0, 0.0]]),
            masses=np.array([2.0]),
            E_at_particles=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            positions=np.array([[0.0, 0.0, 0.0]]),
            dt=0.01,
        )
        assert self.inv.compute(state) == pytest.approx(0.0)

    def test_boris_b_rotation_preserves_energy(self):
        """In B-only case (E=0), Boris rotation preserves |v| exactly.

        The work-energy residual should be ~0 since delta_KE=0 and W_E=0.
        """
        inv = BorisEnergyInvariant()
        q_val = 1.0
        m = 1.0
        B = np.array([0.0, 0.0, 1.0])  # noqa: N806
        dt = 0.1

        v = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([m])
        charges = np.array([q_val])
        r0 = np.array([[0.0, 0.0, 0.0]])
        E_zero = np.array([[0.0, 0.0, 0.0]])  # noqa: N806

        # First call to establish baseline
        state_before = make_state(
            velocities=v.copy(), masses=masses, dt=dt,
            E_at_particles=E_zero, charges=charges, positions=r0,
        )
        val0 = inv.compute(state_before)
        assert val0 == pytest.approx(0.0)

        # Boris rotation (B-only, no E-field)
        t_vec = (q_val * B / m) * dt / 2.0
        t_mag_sq = np.dot(t_vec, t_vec)
        s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)

        v_minus = v[0].copy()
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)

        # Position update (simple Euler for test): r1 = r0 + v_plus * dt
        r1 = r0 + np.array([v_plus]) * dt

        state_after = make_state(
            velocities=np.array([v_plus]),
            masses=masses, dt=dt,
            E_at_particles=E_zero, charges=charges, positions=r1,
        )
        residual = inv.compute(state_after)

        # Residual should be ~0: delta_KE=0, W_E=0
        assert abs(residual) < 1e-14
        assert inv.check(val0, residual) is None

    def test_residual_detects_wrong_work(self):
        """When the work-energy balance is violated, the residual is nonzero."""
        inv = BorisEnergyInvariant()

        v0 = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])
        charges = np.array([1.0])
        r0 = np.array([[0.0, 0.0, 0.0]])
        E = np.array([[0.0, 0.0, 0.0]])  # noqa: N806

        # First call
        s0 = make_state(velocities=v0, masses=masses, dt=0.1,
                         E_at_particles=E, charges=charges, positions=r0)
        inv.compute(s0)

        # Second call: KE increased but E=0 so W_E=0 -> residual != 0
        v1 = np.array([[2.0, 0.0, 0.0]])  # KE went from 0.5 to 2.0
        r1 = np.array([[0.1, 0.0, 0.0]])
        s1 = make_state(velocities=v1, masses=masses, dt=0.1,
                         E_at_particles=E, charges=charges, positions=r1)
        residual = inv.compute(s1)

        # delta_KE = 2.0 - 0.5 = 1.5, W_E = 0 -> residual = 1.5
        assert abs(residual) > 1.0

    def test_check_detects_large_residual(self):
        """Check method detects a large residual."""
        result = self.inv.check(0.0, 0.5)
        assert result is not None
        assert result.severity in (
            ViolationSeverity.WARNING, ViolationSeverity.ERROR,
            ViolationSeverity.CRITICAL,
        )

    def test_check_passes_small_residual(self):
        """Check method passes when residual is small."""
        assert self.inv.check(0.0, 1e-8) is None

    def test_applicable(self):
        state = make_state(
            velocities=np.array([[1.0, 0.0, 0.0]]),
            masses=np.array([1.0]),
            E_at_particles=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            positions=np.array([[0.0, 0.0, 0.0]]),
            dt=0.01,
        )
        assert self.inv.applicable(state)

    def test_not_applicable_no_dt(self):
        state = make_state(
            velocities=np.array([[1.0, 0.0, 0.0]]),
            masses=np.array([1.0]),
        )
        assert not self.inv.applicable(state)

    def test_not_applicable_missing_E(self):
        """Not applicable without E_at_particles."""
        state = make_state(
            velocities=np.array([[1.0, 0.0, 0.0]]),
            masses=np.array([1.0]),
            charges=np.array([1.0]),
            positions=np.array([[0.0, 0.0, 0.0]]),
            dt=0.01,
        )
        assert not self.inv.applicable(state)


# ===========================================================================
# GaussLawInvariant
# ===========================================================================

class TestGaussLawInvariant:
    def setup_method(self):
        self.inv = GaussLawInvariant()

    def test_uniform_field_zero_charge_staggered(self):
        """Uniform E-field with zero charge on staggered grid: div(E) = 0."""
        N = 16
        E = np.zeros((N, N, 2))
        E[:, :, 0] = 1.0  # constant E_x, so forward diff = 0
        rho = np.zeros((N, N))
        eps_0 = 1.0

        state = make_state(
            E_field=E,
            charge_density=rho,
            dx=1.0,
            eps_0=eps_0,
            staggered_grid=True,
        )
        residual = self.inv.compute(state)
        assert residual < 1e-10

    def test_uniform_field_zero_charge_collocated(self):
        """Uniform E-field with zero charge on collocated grid: div(E) = 0."""
        N = 16
        E = np.zeros((N, N, 2))
        E[:, :, 0] = 1.0
        rho = np.zeros((N, N))
        eps_0 = 1.0

        state = make_state(
            E_field=E,
            charge_density=rho,
            dx=1.0,
            eps_0=eps_0,
            staggered_grid=False,
        )
        residual = self.inv.compute(state)
        assert residual < 1e-10

    def test_nonzero_charge_zero_field(self):
        """Non-zero charge density with zero E-field: Gauss violated."""
        N = 16
        E = np.zeros((N, N, 2))  # zero field
        rho = np.ones((N, N))  # non-zero charge
        eps_0 = 1.0

        state = make_state(
            E_field=E,
            charge_density=rho,
            dx=1.0,
            eps_0=eps_0,
            staggered_grid=True,
        )
        residual = self.inv.compute(state)
        # div(E) = 0 but rho/eps_0 = 1, so residual should be ~1
        assert residual > 0.5

    def test_check_zero_residual(self):
        """No violation when Gauss's law is satisfied."""
        assert self.inv.check(0.0, 1e-14) is None

    def test_check_large_residual(self):
        """Violation when residual is large."""
        result = self.inv.check(0.0, 0.1)
        assert result is not None

    def test_applicable(self):
        state = make_state(
            E_field=np.zeros((10, 3)),
            charge_density=np.zeros(10),
        )
        assert self.inv.applicable(state)

    def test_not_applicable(self):
        state = make_state(E_field=np.zeros((10, 3)))
        assert not self.inv.applicable(state)


# ===========================================================================
# LorentzForceInvariant
# ===========================================================================

class TestLorentzForceInvariant:
    def setup_method(self):
        self.inv = LorentzForceInvariant()

    def test_correct_lorentz_force(self):
        """F = q(E + v x B) computed correctly -> zero residual."""
        N = 5
        v = np.random.randn(N, 3)
        E = np.random.randn(N, 3)
        B = np.random.randn(N, 3)
        q = np.ones(N)

        # Correct Lorentz force
        F = q[:, np.newaxis] * (E + np.cross(v, B))

        state = make_state(
            applied_force=F,
            velocities=v,
            charges=q,
            E_at_particles=E,
            B_at_particles=B,
        )
        residual = self.inv.compute(state)
        assert residual < 1e-14

    def test_wrong_sign_cross_product(self):
        """Sign error in v x B -> detectable residual."""
        N = 5
        v = np.random.randn(N, 3) + 1.0  # Ensure non-zero
        E = np.random.randn(N, 3)
        B = np.random.randn(N, 3) + 1.0
        q = np.ones(N)

        # Wrong sign on cross product (common bug)
        F_wrong = q[:, np.newaxis] * (E - np.cross(v, B))

        state = make_state(
            applied_force=F_wrong,
            velocities=v,
            charges=q,
            E_at_particles=E,
            B_at_particles=B,
        )
        residual = self.inv.compute(state)
        assert residual > 0.1  # Should be clearly detectable

    def test_missing_charge_factor(self):
        """Missing q factor: F = E + v x B instead of q(E + v x B)."""
        N = 3
        v = np.array([[1.0, 2.0, 0.0]] * N)
        E = np.array([[1.0, 1.0, 1.0]] * N)
        B = np.array([[0.0, 0.0, 1.0]] * N)
        q = np.array([3.0, 3.0, 3.0])  # q != 1

        # Bug: missing charge factor
        F_wrong = E + np.cross(v, B)  # should be q * (E + v x B)
        # Verify E + v x B is non-zero for this test to be meaningful
        F_expected = q[:, np.newaxis] * (E + np.cross(v, B))
        assert np.linalg.norm(F_expected) > 1.0, "Test vectors must produce non-zero force"

        state = make_state(
            applied_force=F_wrong,
            velocities=v,
            charges=q,
            E_at_particles=E,
            B_at_particles=B,
        )
        residual = self.inv.compute(state)
        assert residual > 0.1

    def test_check_zero_residual(self):
        assert self.inv.check(0.0, 1e-14) is None

    def test_check_large_residual(self):
        result = self.inv.check(0.0, 0.5)
        assert result is not None

    def test_applicable(self):
        state = make_state(
            applied_force=np.zeros((3, 3)),
            velocities=np.zeros((3, 3)),
            charges=np.ones(3),
            E_at_particles=np.zeros((3, 3)),
            B_at_particles=np.zeros((3, 3)),
        )
        assert self.inv.applicable(state)

    def test_not_applicable(self):
        state = make_state(velocities=np.zeros((3, 3)))
        assert not self.inv.applicable(state)
