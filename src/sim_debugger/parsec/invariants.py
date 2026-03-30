"""PARSEC-specific invariant monitors.

Three invariant monitors tailored to particle-in-cell simulations and
the Boris particle pusher algorithm:

1. BorisEnergyInvariant -- energy conservation specific to the Boris pusher
2. GaussLawInvariant -- div(E) = rho/eps_0 (Gauss's law on the grid)
3. LorentzForceInvariant -- verifies F = q(E + v x B) consistency
"""

from __future__ import annotations

import math

import numpy as np

from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import (
    Violation,
    ViolationSeverity,
    classify_severity,
)


def _standard_check(
    invariant_name: str,
    prev_value: float,
    curr_value: float,
    threshold: float,
) -> Violation | None:
    """Standard violation check (shared logic, same as core._standard_check)."""
    if not math.isfinite(curr_value):
        return Violation(
            invariant_name=invariant_name,
            timestep=0,
            time=0.0,
            expected_value=prev_value,
            actual_value=curr_value,
            relative_error=float("inf"),
            absolute_error=float("inf"),
            severity=ViolationSeverity.CRITICAL,
        )

    absolute_error = abs(curr_value - prev_value)
    if abs(prev_value) > 1e-300:
        relative_error = absolute_error / abs(prev_value)
    else:
        relative_error = absolute_error

    if relative_error > threshold:
        severity = classify_severity(relative_error, threshold, curr_value)
        return Violation(
            invariant_name=invariant_name,
            timestep=0,
            time=0.0,
            expected_value=prev_value,
            actual_value=curr_value,
            relative_error=relative_error,
            absolute_error=absolute_error,
            severity=severity,
        )
    return None


# ---------------------------------------------------------------------------
# PARSEC Invariant #1: Boris Pusher Energy Conservation
# ---------------------------------------------------------------------------

class BorisEnergyInvariant:
    """Monitors energy conservation specific to the Boris particle pusher.

    The Boris pusher is a second-order, volume-preserving integrator with
    a three-step structure:

    1. Half E-push:  v^- = v^n + (q*dt)/(2*m) * E
    2. B-rotation:   v^+ = rotate(v^-, B, dt)   -- preserves |v|
    3. Half E-push:  v^{n+1} = v^+ + (q*dt)/(2*m) * E

    In the absence of electric fields (E=0), the B-rotation step exactly
    preserves |v|^2, so kinetic energy is exactly conserved. With electric
    fields, energy is conserved to O(dt^2).

    This invariant monitors the work-energy residual:

        residual = (KE_current - KE_prev) - W_E

    where W_E = sum_i(q_i * E_i . delta_r_i) is the work done by the
    electric field over one timestep. A correctly running Boris pusher
    should have residual ~ 0 at each step.

    Required arrays:
        - velocities: (N, D) particle velocities
        - masses: (N,) particle masses
        - E_at_particles: (N, D) electric field interpolated to particles
        - charges: (N,) particle charges
        - positions: (N, D) particle positions
    Required metadata:
        - dt: timestep size
    """

    def __init__(self) -> None:
        self._prev_ke: float | None = None
        self._prev_positions: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Boris Energy"

    @property
    def description(self) -> str:
        return (
            "Boris pusher energy conservation: kinetic energy change must "
            "equal work done by electric field"
        )

    @property
    def default_threshold(self) -> float:
        return 1e-6

    def _kinetic_energy(self, v: np.ndarray, m: np.ndarray | float) -> float:
        """Compute total kinetic energy 0.5 * sum(m_i * |v_i|^2)."""
        v_sq = np.sum(v * v, axis=-1) if v.ndim > 1 else v * v
        if np.isscalar(m) or (isinstance(m, np.ndarray) and m.ndim == 0):
            return 0.5 * float(np.asarray(m).item()) * float(np.sum(v_sq))
        return 0.5 * float(np.sum(m * v_sq))

    def compute(self, state: SimulationState) -> float:
        """Compute the work-energy residual for the Boris pusher.

        Returns the residual: (delta_KE - W_E) where
        W_E = sum_i(q_i * E_i . delta_r_i).

        On the first call (no previous state), returns 0.0 and stores
        the initial KE and positions for the next step.
        """
        v = state.get_array("velocities")
        m = state.get_array("masses")
        E = state.get_array("E_at_particles")  # noqa: N806
        q = state.get_array("charges")
        r = state.get_array("positions")

        curr_ke = self._kinetic_energy(v, m)

        if self._prev_ke is None or self._prev_positions is None:
            # First call: store initial state, residual is zero
            self._prev_ke = curr_ke
            self._prev_positions = np.array(r, copy=True)
            return 0.0

        # Displacement since last step
        dr = r - self._prev_positions

        # Work done by electric field: W_E = sum_i(q_i * E_i . dr_i)
        # q has shape (N,), E and dr have shape (N, D)
        if q.ndim == 0 or np.isscalar(q):
            W_E = float(q) * float(np.sum(E * dr))  # noqa: N806
        else:
            E_dot_dr = np.sum(E * dr, axis=-1)  # (N,)
            W_E = float(np.sum(q * E_dot_dr))  # noqa: N806

        # Work-energy residual: should be ~0 for a correct Boris pusher
        delta_ke = curr_ke - self._prev_ke
        residual = delta_ke - W_E

        # Update stored state for next step
        self._prev_ke = curr_ke
        self._prev_positions = np.array(r, copy=True)

        return residual

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        """Check if the work-energy residual exceeds threshold.

        For this invariant, the ideal value is zero, so we check the
        absolute value of the residual against the threshold.
        """
        thr = threshold if threshold is not None else self.default_threshold

        if not math.isfinite(curr_value):
            return Violation(
                invariant_name=self.name,
                timestep=0,
                time=0.0,
                expected_value=0.0,
                actual_value=curr_value,
                relative_error=float("inf"),
                absolute_error=float("inf"),
                severity=ViolationSeverity.CRITICAL,
            )

        if abs(curr_value) > thr:
            absolute_error = abs(curr_value)
            relative_error = absolute_error / thr
            severity = classify_severity(relative_error, 1.0, curr_value)
            return Violation(
                invariant_name=self.name,
                timestep=0,
                time=0.0,
                expected_value=0.0,
                actual_value=curr_value,
                relative_error=relative_error,
                absolute_error=absolute_error,
                severity=severity,
            )
        return None

    def applicable(self, state: SimulationState) -> bool:
        return (
            state.has_array("velocities")
            and state.has_array("masses")
            and state.has_array("E_at_particles")
            and state.has_array("charges")
            and state.has_array("positions")
            and "dt" in state.metadata
        )


# ---------------------------------------------------------------------------
# PARSEC Invariant #2: Gauss's Law (div E = rho / eps_0)
# ---------------------------------------------------------------------------

class GaussLawInvariant:
    """Monitors Gauss's law: div(E) = rho / eps_0 at every grid point.

    In an electromagnetic PIC code, Gauss's law should hold at every
    timestep if it holds at t=0 and the current deposition scheme is
    charge-conserving. A violation indicates a bug in current deposition,
    the field solver, or charge-boundary coupling.

    The invariant value is the L2 norm of the residual:
        residual = div(E) - rho / eps_0

    computed over all grid points. A correctly running simulation should
    have residual at machine precision.

    For Yee staggered grids (the default), div(E) uses forward differences:
        (E[i+1] - E[i]) / dx

    For collocated grids, central differences are used:
        (E[i+1] - E[i-1]) / (2*dx)

    Required arrays:
        - E_field: (Nx, Ny, Nz, 3) or (Nx, 3) electric field on grid
        - charge_density: (Nx, Ny, Nz) or (Nx,) charge density on grid
    Required metadata:
        - dx: grid spacing (scalar or tuple)
        - eps_0: permittivity of free space (default: 8.854e-12)
        - staggered_grid: bool (default: True) -- use forward differences
    """

    @property
    def name(self) -> str:
        return "Gauss's Law"

    @property
    def description(self) -> str:
        return "Gauss's law: div(E) = rho/eps_0 (charge conservation on grid)"

    @property
    def default_threshold(self) -> float:
        return 1e-10

    def compute(self, state: SimulationState) -> float:
        """Compute L2 norm of Gauss's law residual.

        Uses forward differences for Yee staggered grids (default) or
        central differences for collocated grids.
        """
        E = state.get_array("E_field")  # noqa: N806
        rho = state.get_array("charge_density")
        eps_0 = state.metadata.get("eps_0", 8.854187817e-12)
        dx = state.metadata.get("dx", 1.0)
        staggered_grid = state.metadata.get("staggered_grid", True)

        # Determine dimensionality from E_field shape
        # E_field is (..., D) where D is the number of spatial dimensions
        ndim = E.shape[-1]

        if isinstance(dx, (int, float)):
            dx_arr = [float(dx)] * ndim
        else:
            dx_arr = [float(d) for d in dx]

        if staggered_grid:
            # Forward difference for Yee staggered grid:
            # div(E)_i = sum_d (E_d[i+1] - E_d[i]) / dx_d
            # This produces an array one element shorter along each axis.
            # We compute on the interior to keep shape consistent with rho.
            E_d = E[..., 0]
            # Build forward difference along axis 0
            diff = np.diff(E_d, axis=0) / dx_arr[0]
            # For additional dimensions, slice to matching shape
            for d in range(1, ndim):
                E_d = E[..., d]
                d_diff = np.diff(E_d, axis=d) / dx_arr[d]
                # Trim previous result and new diff to common shape
                common_shape = [
                    min(diff.shape[ax], d_diff.shape[ax])
                    for ax in range(diff.ndim)
                ]
                slices = tuple(slice(0, s) for s in common_shape)
                diff = diff[slices] + d_diff[slices]
            div_E = diff  # noqa: N806

            # Trim rho to match the divergence array shape
            rho_trimmed = rho[tuple(slice(0, s) for s in div_E.shape)]
            residual = div_E - rho_trimmed / eps_0
        else:
            # Central differences for collocated grids
            div_E = np.zeros_like(rho, dtype=float)  # noqa: N806
            for d in range(ndim):
                E_d = E[..., d]
                div_E += np.gradient(E_d, dx_arr[d], axis=d)
            residual = div_E - rho / eps_0

        # Return L2 norm of residual (normalised by grid size)
        return float(np.sqrt(np.mean(residual ** 2)))

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        """Check if Gauss's law residual has grown.

        For Gauss's law, we check the absolute value against threshold
        (not the relative change), since the ideal value is zero.
        """
        thr = threshold if threshold is not None else self.default_threshold

        if not math.isfinite(curr_value):
            return Violation(
                invariant_name=self.name,
                timestep=0,
                time=0.0,
                expected_value=0.0,
                actual_value=curr_value,
                relative_error=float("inf"),
                absolute_error=float("inf"),
                severity=ViolationSeverity.CRITICAL,
            )

        # For Gauss's law, the ideal value is 0. Check if the absolute
        # residual exceeds the threshold.
        if curr_value > thr:
            absolute_error = curr_value
            relative_error = curr_value / thr  # relative to threshold
            severity = classify_severity(relative_error, 1.0, curr_value)
            return Violation(
                invariant_name=self.name,
                timestep=0,
                time=0.0,
                expected_value=0.0,
                actual_value=curr_value,
                relative_error=relative_error,
                absolute_error=absolute_error,
                severity=severity,
            )
        return None

    def applicable(self, state: SimulationState) -> bool:
        return state.has_array("E_field") and state.has_array("charge_density")


# ---------------------------------------------------------------------------
# PARSEC Invariant #3: Lorentz Force Correctness
# ---------------------------------------------------------------------------

class LorentzForceInvariant:
    """Monitors Lorentz force correctness: F_applied == q(E + v x B).

    Verifies that the force applied to each particle in the simulation
    matches the analytically expected Lorentz force. Detects sign errors,
    missing factors, wrong field interpolation, and cross-product bugs.

    The invariant value is the L2 norm of the force residual, normalised
    by the expected force magnitude.

    Required arrays:
        - applied_force: (N, 3) force actually applied to particles
        - velocities: (N, 3) particle velocities
        - charges: (N,) particle charges
        - E_at_particles: (N, 3) electric field at particle positions
        - B_at_particles: (N, 3) magnetic field at particle positions
    """

    @property
    def name(self) -> str:
        return "Lorentz Force"

    @property
    def description(self) -> str:
        return "Lorentz force correctness: F = q(E + v x B)"

    @property
    def default_threshold(self) -> float:
        return 1e-10

    def compute(self, state: SimulationState) -> float:
        """Compute normalised L2 residual of Lorentz force.

        Returns ||F_applied - q(E + v x B)|| / ||q(E + v x B)||
        averaged over all particles.
        """
        F_applied = state.get_array("applied_force")
        v = state.get_array("velocities")
        q = state.get_array("charges")
        E = state.get_array("E_at_particles")
        B = state.get_array("B_at_particles")

        # Validate 3D input (cross product requires 3 components)
        if v.ndim < 2 or v.shape[-1] != 3:
            raise ValueError(
                f"Lorentz force requires 3D velocity vectors (shape (N, 3)), "
                f"got shape {v.shape}. For 2D simulations, pad vectors to 3D "
                f"with zero z-component before passing to LorentzForceInvariant."
            )

        # Expected Lorentz force: F = q * (E + v x B)
        v_cross_B = np.cross(v, B)
        F_expected = q[:, np.newaxis] * (E + v_cross_B)

        # Residual
        residual = F_applied - F_expected

        # Per-particle residual norm
        residual_norm = np.linalg.norm(residual, axis=-1)
        expected_norm = np.linalg.norm(F_expected, axis=-1)

        # Normalised residual (with near-zero protection)
        mask = expected_norm > 1e-300
        if not np.any(mask):
            # All forces are zero -- check if applied is also zero
            return float(np.mean(residual_norm))

        normalised = np.zeros_like(residual_norm)
        normalised[mask] = residual_norm[mask] / expected_norm[mask]
        normalised[~mask] = residual_norm[~mask]

        return float(np.mean(normalised))

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        """Check if the Lorentz force residual exceeds threshold.

        Like Gauss's law, the ideal value is zero, so we check absolute value.
        """
        thr = threshold if threshold is not None else self.default_threshold

        if not math.isfinite(curr_value):
            return Violation(
                invariant_name=self.name,
                timestep=0,
                time=0.0,
                expected_value=0.0,
                actual_value=curr_value,
                relative_error=float("inf"),
                absolute_error=float("inf"),
                severity=ViolationSeverity.CRITICAL,
            )

        if curr_value > thr:
            absolute_error = curr_value
            relative_error = curr_value / thr
            severity = classify_severity(relative_error, 1.0, curr_value)
            return Violation(
                invariant_name=self.name,
                timestep=0,
                time=0.0,
                expected_value=0.0,
                actual_value=curr_value,
                relative_error=relative_error,
                absolute_error=absolute_error,
                severity=severity,
            )
        return None

    def applicable(self, state: SimulationState) -> bool:
        return (
            state.has_array("applied_force")
            and state.has_array("velocities")
            and state.has_array("charges")
            and state.has_array("E_at_particles")
            and state.has_array("B_at_particles")
        )
