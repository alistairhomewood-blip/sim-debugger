"""Monitor: the orchestration engine that ties invariant checking together.

The Monitor is the central class that:
1. Maintains the invariant registry and active invariants
2. Tracks invariant values across timesteps
3. Detects violations and triggers localisation + explanation
4. Collects all violations for the final report
5. Optionally records history for trend analysis and export (Phase 3)
"""

from __future__ import annotations

import logging
import math

from sim_debugger.core.config import SimDebuggerConfig
from sim_debugger.core.history import ViolationHistory
from sim_debugger.core.invariants import (
    InvariantRegistry,
    create_default_registry,
)
from sim_debugger.core.state import SimulationState, StateHistory
from sim_debugger.core.violations import (
    LocalisationResult,
    Violation,
)
from sim_debugger.explain.generator import generate_explanation
from sim_debugger.localise.temporal import localise_temporal
from sim_debugger.parsec import (
    BorisEnergyInvariant,
    GaussLawInvariant,
    LorentzForceInvariant,
)

logger = logging.getLogger(__name__)


def _create_full_registry() -> InvariantRegistry:
    """Create a registry with all built-in + PARSEC invariants."""
    registry = create_default_registry()
    registry.register(BorisEnergyInvariant())
    registry.register(GaussLawInvariant())
    registry.register(LorentzForceInvariant())
    return registry


class Monitor:
    """Orchestrates invariant monitoring for a simulation.

    Usage::

        monitor = Monitor(invariants=["Total Energy", "Linear Momentum"])
        for t in range(num_timesteps):
            state = ... # update simulation
            violations = monitor.check(state)
            if violations:
                print(monitor.format_violations(violations))
        print(monitor.report())

    Args:
        invariants: List of invariant names to monitor. If None, auto-detect
                   from the first state.
        thresholds: Per-invariant threshold overrides.
        registry: Custom invariant registry. If None, uses full default.
        history_size: Number of states to keep in the ring buffer.
        check_interval: Check invariants every N timesteps (1 = every step).
        config: Optional SimDebuggerConfig for full configuration.
        record_history: If True, maintain a ViolationHistory for trend analysis.
    """

    def __init__(
        self,
        invariants: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        registry: InvariantRegistry | None = None,
        history_size: int = 100,
        check_interval: int = 1,
        config: SimDebuggerConfig | None = None,
        record_history: bool = False,
    ) -> None:
        self._registry = registry or _create_full_registry()
        self._requested_invariants = invariants
        self._thresholds = thresholds or {}
        self._history = StateHistory(max_size=history_size)
        self._check_interval = check_interval

        # Apply config overrides
        if config is not None:
            if config.monitor.invariants is not None and invariants is None:
                self._requested_invariants = config.monitor.invariants
            if config.thresholds.thresholds and not thresholds:
                self._thresholds = dict(config.thresholds.thresholds)
            self._check_interval = config.get_check_interval()
            self._history = StateHistory(max_size=config.monitor.history_size)

        # Violation history for trend analysis (Phase 3)
        self._violation_history: ViolationHistory | None = None
        if record_history:
            self._violation_history = ViolationHistory()

        # Populated on first check
        self._active_invariants: list[str] = []
        self._prev_values: dict[str, float] = {}
        self._initial_values: dict[str, float] = {}
        self._violations: list[Violation] = []
        self._step_count: int = 0
        self._initialised: bool = False

    @property
    def violations(self) -> list[Violation]:
        """All violations detected so far."""
        return list(self._violations)

    @property
    def active_invariants(self) -> list[str]:
        """Names of currently active invariant monitors."""
        return list(self._active_invariants)

    @property
    def step_count(self) -> int:
        """Number of timesteps monitored so far."""
        return self._step_count

    @property
    def violation_history(self) -> ViolationHistory | None:
        """The violation history tracker, if enabled."""
        return self._violation_history

    def _initialise(self, state: SimulationState) -> None:
        """Set up active invariants from the first state snapshot."""
        if self._requested_invariants is not None:
            self._active_invariants = list(self._requested_invariants)
        else:
            # Auto-detect applicable invariants
            applicable = self._registry.find_applicable(state)
            self._active_invariants = [inv.name for inv in applicable]

        # Validate that all requested invariants exist
        for name in self._active_invariants:
            try:
                self._registry.get(name)  # raises KeyError if not found
            except KeyError as e:
                logger.error("Invariant not found: %s", e)
                raise

        # Compute initial values
        for name in self._active_invariants:
            invariant = self._registry.get(name)
            try:
                value = invariant.compute(state)
                self._prev_values[name] = value
                self._initial_values[name] = value
            except (KeyError, ValueError) as e:
                logger.warning(
                    "Cannot compute invariant '%s' for initial state: %s",
                    name, e,
                )

        # Remove invariants we couldn't compute
        self._active_invariants = [
            name for name in self._active_invariants
            if name in self._prev_values
        ]

        self._initialised = True

    def check(self, state: SimulationState) -> list[Violation]:
        """Check all active invariants against the current state.

        This is the main entry point, called once per timestep. It:
        1. Computes current invariant values
        2. Checks for violations against previous values
        3. If violated, performs temporal localisation and explanation
        4. Stores state in history buffer
        5. Optionally records to violation history for trend analysis

        Args:
            state: The current simulation state snapshot.

        Returns:
            List of violations detected at this timestep (may be empty).
        """
        self._step_count += 1

        if not self._initialised:
            self._initialise(state)
            # Store initial state in history
            self._history.push(state, dict(self._prev_values))
            if self._violation_history is not None:
                self._violation_history.record_values(
                    state.timestep, dict(self._prev_values),
                )
            return []

        # Respect check interval
        if self._step_count % self._check_interval != 0:
            return []

        # Compute current invariant values
        current_values: dict[str, float] = {}
        step_violations: list[Violation] = []

        for name in self._active_invariants:
            invariant = self._registry.get(name)
            try:
                value = invariant.compute(state)
                current_values[name] = value
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(
                    "Cannot compute invariant '%s' at timestep %d: %s",
                    name, state.timestep, e,
                )
                continue
            except Exception as e:
                # Catch unexpected errors in invariant computation to
                # avoid crashing the simulation. Log and continue.
                logger.warning(
                    "Unexpected error computing invariant '%s' at "
                    "timestep %d: %s: %s",
                    name, state.timestep, type(e).__name__, e,
                )
                continue

            # Check for violation: both step-to-step and drift from initial
            threshold = self._thresholds.get(name, invariant.default_threshold)
            prev = self._prev_values.get(name)
            if prev is None:
                continue

            # Step-to-step check
            try:
                violation = invariant.check(prev, value, threshold)
            except Exception as e:
                logger.warning(
                    "Error in invariant check for '%s': %s: %s",
                    name, type(e).__name__, e,
                )
                violation = None

            # Drift-from-initial check: catches gradual accumulation
            # that is below threshold per step but significant overall.
            # Use sqrt(steps) scaling to tolerate round-off noise
            # accumulation while still catching linear/exponential drift.
            if violation is None:
                initial = self._initial_values.get(name)
                if initial is not None:
                    drift_threshold = threshold * max(10.0, math.sqrt(state.timestep + 1))
                    try:
                        violation = invariant.check(initial, value, drift_threshold)
                    except Exception:
                        violation = None

            if violation is not None:
                # Enrich the violation with timestep/time info
                violation = Violation(
                    invariant_name=violation.invariant_name,
                    timestep=state.timestep,
                    time=state.time,
                    expected_value=violation.expected_value,
                    actual_value=violation.actual_value,
                    relative_error=violation.relative_error,
                    absolute_error=violation.absolute_error,
                    severity=violation.severity,
                )

                # Temporal localisation
                try:
                    temporal_loc = localise_temporal(
                        name, self._history, threshold, state.timestep,
                    )
                except Exception as e:
                    logger.debug("Temporal localisation failed: %s", e)
                    temporal_loc = None

                if temporal_loc is not None:
                    localisation = LocalisationResult(temporal=temporal_loc)
                    pattern = temporal_loc.pattern
                    first_ts = temporal_loc.first_violation_timestep
                    dur = temporal_loc.duration
                else:
                    localisation = None
                    pattern = None
                    first_ts = None
                    dur = None

                # Generate explanation
                try:
                    explanation = generate_explanation(
                        violation,
                        pattern=pattern,
                        first_timestep=first_ts,
                        duration=dur,
                    )
                except Exception as e:
                    logger.debug("Explanation generation failed: %s", e)
                    explanation = (
                        f"{violation.invariant_name} violation at timestep "
                        f"{violation.timestep}: relative error "
                        f"{violation.relative_error:.2%}"
                    )

                # Create final violation with all enrichments
                violation = Violation(
                    invariant_name=violation.invariant_name,
                    timestep=state.timestep,
                    time=state.time,
                    expected_value=violation.expected_value,
                    actual_value=violation.actual_value,
                    relative_error=violation.relative_error,
                    absolute_error=violation.absolute_error,
                    severity=violation.severity,
                    localisation=localisation,
                    explanation=explanation,
                )

                step_violations.append(violation)
                self._violations.append(violation)

        # Update previous values and history
        for name, value in current_values.items():
            self._prev_values[name] = value

        self._history.push(state, current_values)

        # Record to violation history for trend analysis
        if self._violation_history is not None:
            self._violation_history.record_values(state.timestep, current_values)
            if step_violations:
                self._violation_history.record_violations(step_violations)

        return step_violations

    def get_current_values(self) -> dict[str, float]:
        """Return the most recent computed invariant values."""
        return dict(self._prev_values)

    def get_initial_values(self) -> dict[str, float]:
        """Return the initial invariant values (at first timestep)."""
        return dict(self._initial_values)

    def report(self) -> str:
        """Generate a summary report of all violations detected.

        Returns:
            Multi-line string with the violation summary.
        """
        lines = []
        lines.append("sim-debugger monitoring report")
        lines.append(f"{'=' * 60}")
        lines.append(f"Timesteps monitored: {self._step_count}")
        lines.append(f"Invariants monitored: {', '.join(self._active_invariants)}")
        lines.append(f"Total violations: {len(self._violations)}")
        lines.append("")

        if not self._violations:
            lines.append("No violations detected. All invariants conserved.")
        else:
            # Group by invariant
            by_invariant: dict[str, list[Violation]] = {}
            for v in self._violations:
                by_invariant.setdefault(v.invariant_name, []).append(v)

            for inv_name, violations in by_invariant.items():
                lines.append(f"--- {inv_name} ---")
                lines.append(f"  Violations: {len(violations)}")
                severities = [v.severity.value for v in violations]
                for sev in ["critical", "error", "warning"]:
                    count = severities.count(sev)
                    if count > 0:
                        lines.append(f"  {sev.upper()}: {count}")

                # Show first violation details
                first = violations[0]
                lines.append(f"  First violation at timestep {first.timestep}:")
                if first.explanation:
                    for exp_line in first.explanation.split("\n"):
                        lines.append(f"    {exp_line}")
                lines.append("")

        # Add trend summary if history is available
        if self._violation_history is not None:
            trends = self._violation_history.compute_all_trends()
            if trends:
                lines.append("Invariant trends:")
                for name, trend in sorted(trends.items()):
                    status = "STABLE" if trend.is_stable else "DRIFTING"
                    lines.append(
                        f"  {name}: {status} "
                        f"(drift={trend.relative_drift:.2e})"
                    )
                lines.append("")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def export_json(self, path: str) -> None:
        """Export the violation history to a JSON file.

        Requires record_history=True in the constructor.

        Args:
            path: Output file path.

        Raises:
            RuntimeError: If history recording was not enabled.
        """
        if self._violation_history is None:
            raise RuntimeError(
                "Violation history not enabled. Create the Monitor with "
                "record_history=True to enable JSON export."
            )
        self._violation_history.export_json(path)

    def export_hdf5(self, path: str) -> None:
        """Export the violation history to an HDF5 file.

        Requires record_history=True and h5py installed.

        Args:
            path: Output file path.
        """
        if self._violation_history is None:
            raise RuntimeError(
                "Violation history not enabled. Create the Monitor with "
                "record_history=True to enable HDF5 export."
            )
        self._violation_history.export_hdf5(path)

    def reset(self) -> None:
        """Reset the monitor to its initial state."""
        self._prev_values.clear()
        self._initial_values.clear()
        self._violations.clear()
        self._history.clear()
        self._step_count = 0
        self._initialised = False
        if self._violation_history is not None:
            self._violation_history = ViolationHistory()
