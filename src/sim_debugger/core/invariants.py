"""Invariant definitions, base protocol, and built-in invariant monitors.

Defines the Invariant protocol that all monitors must implement, the invariant
registry for discovering and managing active monitors, and the five built-in
invariant monitors:

1. TotalEnergyInvariant -- kinetic + potential + field energy conservation
2. LinearMomentumInvariant -- total linear momentum conservation
3. AngularMomentumInvariant -- total angular momentum conservation
4. ChargeConservationInvariant -- total charge conservation
5. ParticleCountInvariant -- particle number conservation
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import (
    Violation,
    ViolationSeverity,
    classify_severity,
)

# ---------------------------------------------------------------------------
# Invariant Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Invariant(Protocol):
    """Protocol that all invariant monitors must implement.

    An invariant is a scalar quantity that should be conserved (constant)
    over time in a correctly implemented simulation. The monitor computes
    the quantity from the simulation state and checks whether consecutive
    values stay within tolerance.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the invariant (e.g. 'Total Energy')."""
        ...

    @property
    def description(self) -> str:
        """One-line description of what this invariant monitors."""
        ...

    @property
    def default_threshold(self) -> float:
        """Default relative tolerance for violation detection."""
        ...

    def compute(self, state: SimulationState) -> float:
        """Compute the invariant value from the current simulation state.

        Args:
            state: The simulation state snapshot.

        Returns:
            The scalar invariant value.
        """
        ...

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        """Compare consecutive invariant values and detect violations.

        Args:
            prev_value: Invariant value at the previous timestep.
            curr_value: Invariant value at the current timestep.
            threshold: Relative tolerance (uses default_threshold if None).

        Returns:
            A Violation if the change exceeds threshold, else None.
        """
        ...

    def applicable(self, state: SimulationState) -> bool:
        """Heuristic check: is this invariant meaningful for the given state?

        Args:
            state: A representative simulation state.

        Returns:
            True if this invariant can be computed from the state's arrays.
        """
        ...


# ---------------------------------------------------------------------------
# Invariant Registry
# ---------------------------------------------------------------------------

class InvariantRegistry:
    """Registry of available invariant monitors.

    Maintains a mapping from invariant names to their implementations.
    Provides lookup, registration, and discovery of applicable invariants.
    """

    def __init__(self) -> None:
        self._invariants: dict[str, Invariant] = {}

    def register(self, invariant: Invariant) -> None:
        """Register an invariant monitor.

        Args:
            invariant: The invariant to register.

        Raises:
            ValueError: If an invariant with the same name is already registered.
        """
        if invariant.name in self._invariants:
            raise ValueError(
                f"Invariant '{invariant.name}' is already registered."
            )
        self._invariants[invariant.name] = invariant

    def get(self, name: str) -> Invariant:
        """Look up an invariant by name.

        Args:
            name: The invariant name (case-sensitive).

        Returns:
            The invariant monitor.

        Raises:
            KeyError: If no invariant with the given name is registered.
        """
        if name not in self._invariants:
            available = ", ".join(sorted(self._invariants.keys()))
            raise KeyError(
                f"Unknown invariant '{name}'. Available: {available}"
            )
        return self._invariants[name]

    def list_all(self) -> list[Invariant]:
        """Return all registered invariants."""
        return list(self._invariants.values())

    def list_names(self) -> list[str]:
        """Return names of all registered invariants."""
        return sorted(self._invariants.keys())

    def find_applicable(self, state: SimulationState) -> list[Invariant]:
        """Find all invariants that are applicable to the given state.

        Args:
            state: A representative simulation state.

        Returns:
            List of invariant monitors whose applicable() returns True.
        """
        return [inv for inv in self._invariants.values() if inv.applicable(state)]


# ---------------------------------------------------------------------------
# Helper: standard check logic
# ---------------------------------------------------------------------------

def _standard_check(
    invariant_name: str,
    prev_value: float,
    curr_value: float,
    threshold: float,
    timestep: int = 0,
    time: float = 0.0,
    absolute_threshold: float | None = None,
) -> Violation | None:
    """Standard violation check shared by all built-in invariants.

    Uses relative error when |prev_value| is sufficiently large.
    When prev_value is near zero, uses absolute_threshold (if provided)
    to avoid the dimensionally inconsistent comparison of an absolute
    error against a dimensionless relative threshold.

    Args:
        invariant_name: Name of the invariant being checked.
        prev_value: Previous timestep's invariant value.
        curr_value: Current timestep's invariant value.
        threshold: Relative tolerance (dimensionless).
        timestep: Current timestep index (for violation reporting).
        time: Current simulation time (for violation reporting).
        absolute_threshold: Absolute tolerance for near-zero values.
            When prev_value is near zero and this is provided, the
            absolute error is compared against this threshold instead
            of the relative one. If None, falls back to comparing
            absolute error against `threshold` (legacy behaviour).

    Returns:
        A Violation if the change exceeds threshold, else None.
    """
    # Handle NaN / Inf
    if not math.isfinite(curr_value):
        return Violation(
            invariant_name=invariant_name,
            timestep=timestep,
            time=time,
            expected_value=prev_value,
            actual_value=curr_value,
            relative_error=float("inf"),
            absolute_error=float("inf"),
            severity=ViolationSeverity.CRITICAL,
        )

    absolute_error = abs(curr_value - prev_value)

    # Relative error with near-zero protection
    if abs(prev_value) > 1e-300:
        relative_error = absolute_error / abs(prev_value)
        effective_threshold = threshold
    else:
        # When prev_value is near zero, use absolute threshold if provided.
        # This avoids comparing a dimensioned absolute error against a
        # dimensionless relative threshold.
        relative_error = absolute_error
        if absolute_threshold is not None:
            effective_threshold = absolute_threshold
        else:
            # Legacy fallback: compare absolute error against relative
            # threshold (dimensionally inconsistent but preserves backward
            # compatibility for callers that don't set absolute_threshold).
            effective_threshold = threshold

    if relative_error > effective_threshold:
        severity = classify_severity(relative_error, effective_threshold, curr_value)
        return Violation(
            invariant_name=invariant_name,
            timestep=timestep,
            time=time,
            expected_value=prev_value,
            actual_value=curr_value,
            relative_error=relative_error,
            absolute_error=absolute_error,
            severity=severity,
        )

    return None


# ---------------------------------------------------------------------------
# Built-in Invariant #1: Total Energy
# ---------------------------------------------------------------------------

class TotalEnergyInvariant:
    """Monitors total energy conservation: E_kinetic + E_potential + E_field.

    Applicable to any Hamiltonian system. Requires the state to contain
    at least velocity and mass arrays for kinetic energy. Potential and
    field energy arrays are included if present.

    Required arrays:
        - velocities: (N, D) array of particle velocities
        - masses: (N,) array of particle masses
    Optional arrays:
        - potential_energy: scalar or (N,) per-particle potential energies
        - E_field, B_field: field arrays for electromagnetic field energy
    """

    @property
    def name(self) -> str:
        return "Total Energy"

    @property
    def description(self) -> str:
        return "Total energy conservation (kinetic + potential + field)"

    @property
    def default_threshold(self) -> float:
        return 1e-6

    def compute(self, state: SimulationState) -> float:
        """Compute total energy from the simulation state."""
        total = 0.0

        # Kinetic energy: 0.5 * sum(m * |v|^2)
        if state.has_array("velocities") and state.has_array("masses"):
            v = state.get_array("velocities")
            m = state.get_array("masses")
            # v can be (N, D) or (N,); m is (N,) or scalar
            if v.ndim == 1:
                v_sq = v * v
            else:
                v_sq = np.sum(v * v, axis=-1)

            if np.isscalar(m) or m.ndim == 0:
                kinetic = 0.5 * float(m) * np.sum(v_sq)
            else:
                kinetic = 0.5 * np.sum(m * v_sq)
            total += kinetic

        # Potential energy (pre-computed by the simulation)
        if state.has_array("potential_energy"):
            pe = state.get_array("potential_energy")
            total += float(np.sum(pe))

        # Electromagnetic field energy: (eps_0/2) * |E|^2 + (1/(2*mu_0)) * |B|^2
        # Only computed for grid fields (ndim > 2). Per-particle fields
        # (shape (N, 3) i.e. ndim == 2 with last dim <= 3) are skipped
        # because their field energy is already captured in potential_energy.
        if state.has_array("E_field"):
            E = state.get_array("E_field")  # noqa: N806
            if E.ndim > 2:
                # Grid field: shape is (Nx, [Ny, [Nz,]] D)
                eps_0 = state.metadata.get("eps_0", 8.854187817e-12)
                dx = state.metadata.get("dx", 1.0)
                ndim_spatial = E.ndim - 1  # Last dimension is vector components
                if isinstance(dx, (int, float)):
                    cell_volume = float(dx) ** ndim_spatial
                else:
                    cell_volume = float(np.prod(dx))
                total += 0.5 * eps_0 * float(np.sum(E * E)) * cell_volume
            # else: per-particle E (ndim == 2, shape (N, D)) -- skip

        if state.has_array("B_field"):
            B = state.get_array("B_field")  # noqa: N806
            if B.ndim > 2:
                # Grid field: shape is (Nx, [Ny, [Nz,]] D)
                mu_0 = state.metadata.get("mu_0", 1.2566370614e-6)
                dx = state.metadata.get("dx", 1.0)
                ndim_spatial = B.ndim - 1
                if isinstance(dx, (int, float)):
                    cell_volume = float(dx) ** ndim_spatial
                else:
                    cell_volume = float(np.prod(dx))
                total += 0.5 / mu_0 * float(np.sum(B * B)) * cell_volume
            # else: per-particle B (ndim == 2) -- skip

        return total

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        thr = threshold if threshold is not None else self.default_threshold
        return _standard_check(self.name, prev_value, curr_value, thr)

    def applicable(self, state: SimulationState) -> bool:
        return state.has_array("velocities") and state.has_array("masses")


# ---------------------------------------------------------------------------
# Built-in Invariant #2: Linear Momentum
# ---------------------------------------------------------------------------

class LinearMomentumInvariant:
    """Monitors total linear momentum conservation: sum(m_i * v_i).

    Conserved when there are no external forces. Tracks each Cartesian
    component of total momentum independently to detect both magnitude
    and direction changes (including pure rotations of the momentum vector).

    Required arrays:
        - velocities: (N, D) array of particle velocities
        - masses: (N,) array of particle masses
    """

    def __init__(self) -> None:
        self._prev_components: np.ndarray | None = None
        self._curr_components: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Linear Momentum"

    @property
    def description(self) -> str:
        return "Total linear momentum conservation (per-component)"

    @property
    def default_threshold(self) -> float:
        return 1e-6

    def compute(self, state: SimulationState) -> float:
        """Compute total linear momentum vector; return magnitude.

        Internally stores the full vector for per-component check().
        """
        v = state.get_array("velocities")
        m = state.get_array("masses")

        if v.ndim == 1:
            if np.isscalar(m) or m.ndim == 0:
                p_total = np.array([float(m) * np.sum(v)])
            else:
                p_total = np.array([np.sum(m * v)])
        else:
            if np.isscalar(m) or m.ndim == 0:
                p_total = float(m) * np.sum(v, axis=0)
            else:
                p_total = np.sum(m[:, np.newaxis] * v, axis=0)

        self._prev_components = self._curr_components
        self._curr_components = np.asarray(p_total, dtype=float)

        return float(np.linalg.norm(p_total))

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
        timestep: int = 0,
        time: float = 0.0,
    ) -> Violation | None:
        thr = threshold if threshold is not None else self.default_threshold

        # Per-component check: detects direction changes that magnitude misses
        if self._prev_components is not None and self._curr_components is not None:
            prev_c = self._prev_components
            curr_c = self._curr_components
            for i in range(len(prev_c)):
                if abs(prev_c[i]) > 1e-300:
                    rel = abs(curr_c[i] - prev_c[i]) / abs(prev_c[i])
                    if rel > thr:
                        return Violation(
                            invariant_name=self.name,
                            timestep=timestep, time=time,
                            expected_value=prev_c[i], actual_value=curr_c[i],
                            relative_error=rel,
                            absolute_error=abs(curr_c[i] - prev_c[i]),
                            severity=classify_severity(rel, thr),
                        )
                elif abs(curr_c[i]) > thr:
                    return Violation(
                        invariant_name=self.name,
                        timestep=timestep, time=time,
                        expected_value=prev_c[i], actual_value=curr_c[i],
                        relative_error=float("inf"),
                        absolute_error=abs(curr_c[i] - prev_c[i]),
                        severity=classify_severity(1.0, thr),
                    )

        # Magnitude check as fallback
        return _standard_check(self.name, prev_value, curr_value, thr,
                               timestep=timestep, time=time)

    def applicable(self, state: SimulationState) -> bool:
        return state.has_array("velocities") and state.has_array("masses")


# ---------------------------------------------------------------------------
# Built-in Invariant #3: Angular Momentum
# ---------------------------------------------------------------------------

class AngularMomentumInvariant:
    """Monitors total angular momentum conservation: sum(r_i x (m_i * v_i)).

    Conserved for rotationally symmetric systems with no external torques.
    Tracks the signed value (2D) or per-component vector (3D) to detect
    direction changes, not just magnitude changes.

    Required arrays:
        - positions: (N, D) array of particle positions (D >= 2)
        - velocities: (N, D) array of particle velocities
        - masses: (N,) array of particle masses
    """

    def __init__(self) -> None:
        self._prev_components: np.ndarray | None = None
        self._curr_components: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Angular Momentum"

    @property
    def description(self) -> str:
        return "Total angular momentum conservation (per-component)"

    @property
    def default_threshold(self) -> float:
        return 1e-6

    def compute(self, state: SimulationState) -> float:
        """Compute total angular momentum; return magnitude.

        Internally stores signed/vector value for per-component check().
        In 2D, stores the signed scalar Lz. In 3D, stores (Lx, Ly, Lz).
        """
        r = state.get_array("positions")
        v = state.get_array("velocities")
        m = state.get_array("masses")

        if r.ndim < 2 or r.shape[1] < 2:
            raise ValueError(
                "Angular momentum requires at least 2D positions. "
                f"Got shape {r.shape}."
            )

        if np.isscalar(m) or m.ndim == 0:
            p = float(m) * v
        else:
            p = m[:, np.newaxis] * v

        ndim = r.shape[1]
        if ndim == 2:
            L_z = float(np.sum(r[:, 0] * p[:, 1] - r[:, 1] * p[:, 0]))
            L_vec = np.array([L_z])
        elif ndim == 3:
            L = np.cross(r, p)
            L_vec = np.sum(L, axis=0).astype(float)
        else:
            raise ValueError(
                f"Angular momentum not defined for {ndim}D. Use 2D or 3D."
            )

        self._prev_components = self._curr_components
        self._curr_components = L_vec

        return float(np.linalg.norm(L_vec))

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
        timestep: int = 0,
        time: float = 0.0,
    ) -> Violation | None:
        thr = threshold if threshold is not None else self.default_threshold

        # Per-component check: detects sign flips and direction changes
        if self._prev_components is not None and self._curr_components is not None:
            prev_c = self._prev_components
            curr_c = self._curr_components
            for i in range(len(prev_c)):
                if abs(prev_c[i]) > 1e-300:
                    rel = abs(curr_c[i] - prev_c[i]) / abs(prev_c[i])
                    if rel > thr:
                        return Violation(
                            invariant_name=self.name,
                            timestep=timestep, time=time,
                            expected_value=prev_c[i], actual_value=curr_c[i],
                            relative_error=rel,
                            absolute_error=abs(curr_c[i] - prev_c[i]),
                            severity=classify_severity(rel, thr),
                        )
                elif abs(curr_c[i]) > thr:
                    return Violation(
                        invariant_name=self.name,
                        timestep=timestep, time=time,
                        expected_value=prev_c[i], actual_value=curr_c[i],
                        relative_error=float("inf"),
                        absolute_error=abs(curr_c[i] - prev_c[i]),
                        severity=classify_severity(1.0, thr),
                    )

        return _standard_check(self.name, prev_value, curr_value, thr,
                               timestep=timestep, time=time)

    def applicable(self, state: SimulationState) -> bool:
        return (
            state.has_array("positions")
            and state.has_array("velocities")
            and state.has_array("masses")
            and state.get_array("positions").ndim >= 2
            and state.get_array("positions").shape[1] >= 2
        )


# ---------------------------------------------------------------------------
# Built-in Invariant #4: Charge Conservation
# ---------------------------------------------------------------------------

class ChargeConservationInvariant:
    """Monitors total charge conservation: sum(q_i) = const.

    For particle-based simulations, the total charge is the sum of all
    particle charges. This should be exactly conserved in a closed system.

    Required arrays:
        - charges: (N,) array of particle charges
    """

    @property
    def name(self) -> str:
        return "Charge Conservation"

    @property
    def description(self) -> str:
        return "Total charge conservation (sum of all particle charges)"

    @property
    def default_threshold(self) -> float:
        return 1e-12

    def compute(self, state: SimulationState) -> float:
        """Compute total charge."""
        charges = state.get_array("charges")
        return float(np.sum(charges))

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        thr = threshold if threshold is not None else self.default_threshold
        return _standard_check(self.name, prev_value, curr_value, thr)

    def applicable(self, state: SimulationState) -> bool:
        return state.has_array("charges")


# ---------------------------------------------------------------------------
# Built-in Invariant #5: Particle Count
# ---------------------------------------------------------------------------

class ParticleCountInvariant:
    """Monitors particle count conservation.

    In a closed system, the number of particles should not change.
    Detects spurious particle creation or destruction at boundaries.

    Required arrays:
        - positions: (N, D) array (N = particle count) OR
        - velocities: (N, D) array (N = particle count) OR
        - particle_count metadata key
    """

    @property
    def name(self) -> str:
        return "Particle Count"

    @property
    def description(self) -> str:
        return "Particle number conservation (no spurious creation/destruction)"

    @property
    def default_threshold(self) -> float:
        # Particle count is integer-valued; any change is a violation
        return 0.5  # relative threshold: 1 particle in 2 is 50%
        # We use an absolute threshold approach internally

    def compute(self, state: SimulationState) -> float:
        """Compute the number of particles."""
        if "particle_count" in state.metadata:
            return float(state.metadata["particle_count"])
        if state.has_array("positions"):
            return float(state.get_array("positions").shape[0])
        if state.has_array("velocities"):
            return float(state.get_array("velocities").shape[0])
        raise KeyError(
            "Cannot determine particle count: no 'positions', 'velocities' "
            "array or 'particle_count' metadata found."
        )

    def check(
        self,
        prev_value: float,
        curr_value: float,
        threshold: float | None = None,
    ) -> Violation | None:
        # For particle count, use absolute comparison (any change is a violation)
        if prev_value == curr_value:
            return None

        absolute_error = abs(curr_value - prev_value)
        if abs(prev_value) > 0:
            relative_error = absolute_error / abs(prev_value)
        else:
            relative_error = absolute_error

        severity = classify_severity(relative_error, 0.01, curr_value)
        return Violation(
            invariant_name=self.name,
            timestep=0,
            time=0.0,
            expected_value=prev_value,
            actual_value=curr_value,
            relative_error=relative_error,
            absolute_error=absolute_error,
            severity=severity,
        )

    def applicable(self, state: SimulationState) -> bool:
        return (
            "particle_count" in state.metadata
            or state.has_array("positions")
            or state.has_array("velocities")
        )


# ---------------------------------------------------------------------------
# Default registry with all built-in invariants
# ---------------------------------------------------------------------------

def create_default_registry() -> InvariantRegistry:
    """Create a registry pre-populated with all built-in invariants."""
    registry = InvariantRegistry()
    registry.register(TotalEnergyInvariant())
    registry.register(LinearMomentumInvariant())
    registry.register(AngularMomentumInvariant())
    registry.register(ChargeConservationInvariant())
    registry.register(ParticleCountInvariant())
    return registry
