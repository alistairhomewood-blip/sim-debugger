"""SciPy backend adapter for sim-debugger.

Provides integration with SciPy's ODE solvers (solve_ivp and odeint),
allowing invariant monitoring at each internal solver step -- not just
at user-requested output points.

Key features:
    - Hook into solve_ivp via the events mechanism for step monitoring
    - Dense output callback for monitoring at solver internal steps
    - Wrapper around odeint that captures state at each output point
    - Support for monitoring during adaptive timestepping

Usage::

    from sim_debugger.backends.scipy_backend import SciPyBackend

    backend = SciPyBackend()

    # Wrap solve_ivp with monitoring
    result = backend.monitored_solve_ivp(
        fun=rhs_function,
        t_span=(0, 10),
        y0=initial_state,
        monitor=monitor,
        state_mapper=my_state_mapper,
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import Violation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SciPy availability check
# ---------------------------------------------------------------------------

_scipy_available: bool | None = None


def _check_scipy() -> bool:
    """Check if SciPy is importable."""
    global _scipy_available
    if _scipy_available is not None:
        return _scipy_available
    try:
        import scipy.integrate  # type: ignore[import-untyped]  # noqa: F401
        _scipy_available = True
    except ImportError:
        _scipy_available = False
    return _scipy_available


def _require_scipy() -> None:
    """Raise ImportError if SciPy is not available."""
    if not _check_scipy():
        raise ImportError(
            "SciPy is required for the SciPy backend but is not installed. "
            "Install it with: pip install scipy"
        )


# ---------------------------------------------------------------------------
# State mapper type
# ---------------------------------------------------------------------------

# A state mapper converts the ODE state vector y and time t into a
# SimulationState. Users provide this to bridge their simulation's
# representation with sim-debugger's state model.
StateMapper = Callable[[np.ndarray, float, int], SimulationState]


def default_state_mapper(y: np.ndarray, t: float, step: int) -> SimulationState:
    """Default state mapper: treats the ODE state as a flat array.

    Attempts to split the state vector into positions and velocities
    if the length is even, otherwise stores the full state as "y".

    Args:
        y: The ODE state vector.
        t: Current time.
        step: Step count.

    Returns:
        A SimulationState with arrays extracted from y.
    """
    arrays: dict[str, np.ndarray] = {"y": np.copy(y)}

    n = len(y)
    if n % 2 == 0:
        half = n // 2
        # Heuristic: first half is positions, second half is velocities
        arrays["positions"] = np.copy(y[:half].reshape(-1, 1))
        arrays["velocities"] = np.copy(y[half:].reshape(-1, 1))
        arrays["masses"] = np.ones(half)

    return SimulationState(
        timestep=step,
        time=t,
        arrays=arrays,
        metadata={"dt": 0.0},  # Unknown for adaptive solvers
    )


# ---------------------------------------------------------------------------
# Monitored event for solve_ivp
# ---------------------------------------------------------------------------

class _MonitorEvent:
    """Event function for solve_ivp that monitors invariants at each step.

    solve_ivp evaluates event functions at every internal step. We use
    this mechanism to check invariants, capturing violations at the
    solver's internal resolution rather than just at output points.

    The event function always returns a positive value (never triggers
    a zero-crossing termination) so the solver continues normally.
    """

    terminal = False
    direction = 0

    def __init__(
        self,
        monitor: Monitor,
        state_mapper: StateMapper,
        violations_out: list[Violation],
    ) -> None:
        self._monitor = monitor
        self._state_mapper = state_mapper
        self._violations = violations_out
        self._step_count = 0

    def __call__(self, t: float, y: np.ndarray) -> float:
        """Called by solve_ivp at each internal step.

        Returns a positive float to avoid triggering termination.
        The side effect is invariant checking.
        """
        self._step_count += 1
        state = self._state_mapper(y, t, self._step_count)
        violations = self._monitor.check(state)
        self._violations.extend(violations)

        # Return a positive value; we never want to terminate via event
        return 1.0


# ---------------------------------------------------------------------------
# Dense output callback for solve_ivp
# ---------------------------------------------------------------------------

class StepCallback:
    """Callback for monitoring at each solver step via solve_ivp's t_eval.

    For SciPy >= 1.12, solve_ivp does not have a direct step callback.
    Instead, this class wraps the right-hand side function to intercept
    every function evaluation and monitor at regular intervals.

    This is an alternative to the events mechanism that provides
    monitoring at every RHS evaluation (more frequent than events).
    """

    def __init__(
        self,
        rhs: Callable[[float, np.ndarray], np.ndarray],
        monitor: Monitor,
        state_mapper: StateMapper,
        violations_out: list[Violation],
        check_every: int = 1,
    ) -> None:
        self._rhs = rhs
        self._monitor = monitor
        self._state_mapper = state_mapper
        self._violations = violations_out
        self._check_every = check_every
        self._call_count = 0
        self._step_count = 0

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        """Wraps the RHS function, adding invariant monitoring.

        Args:
            t: Current time.
            y: Current state vector.

        Returns:
            The derivative dy/dt from the original RHS function.
        """
        self._call_count += 1

        if self._call_count % self._check_every == 0:
            self._step_count += 1
            state = self._state_mapper(y, t, self._step_count)
            violations = self._monitor.check(state)
            self._violations.extend(violations)

        return self._rhs(t, y)


# ---------------------------------------------------------------------------
# SciPy Backend
# ---------------------------------------------------------------------------

class SciPyBackend:
    """SciPy backend for ODE solver integration with invariant monitoring.

    Provides wrappers around scipy.integrate.solve_ivp and
    scipy.integrate.odeint that inject invariant monitoring at
    each solver step.
    """

    name: str = "scipy"

    @staticmethod
    def is_available() -> bool:
        """Check if SciPy is installed."""
        return _check_scipy()

    @staticmethod
    def detect_backend(source_code: str) -> bool:
        """Detect if the source code uses SciPy ODE solvers."""
        return (
            "scipy.integrate" in source_code
            or "from scipy.integrate" in source_code
            or "solve_ivp" in source_code
            or "odeint" in source_code
        )

    @staticmethod
    def monitored_solve_ivp(
        fun: Callable[[float, np.ndarray], np.ndarray],
        t_span: tuple[float, float],
        y0: np.ndarray,
        monitor: Monitor,
        state_mapper: StateMapper | None = None,
        method: str = "RK45",
        t_eval: np.ndarray | None = None,
        dense_output: bool = False,
        events: Any = None,
        use_rhs_wrapper: bool = False,
        rhs_check_every: int = 1,
        **solve_ivp_kwargs: Any,
    ) -> Any:
        """Solve an ODE with solve_ivp while monitoring invariants.

        Wraps scipy.integrate.solve_ivp and injects invariant monitoring
        at each solver step via the events mechanism or RHS wrapper.

        Args:
            fun: Right-hand side function dy/dt = fun(t, y).
            t_span: Integration interval (t_start, t_end).
            y0: Initial state vector.
            monitor: The Monitor instance for invariant checking.
            state_mapper: Converts (y, t, step) to SimulationState.
                         Uses default_state_mapper if None.
            method: Integration method (default "RK45").
            t_eval: Times at which to store the solution.
            dense_output: If True, compute dense output.
            events: Additional event functions (will be combined with
                    the monitoring event).
            use_rhs_wrapper: If True, monitor at every RHS evaluation
                            instead of using events. More frequent but
                            higher overhead.
            rhs_check_every: When using RHS wrapper, check every N calls.
            **solve_ivp_kwargs: Additional kwargs passed to solve_ivp.

        Returns:
            The solve_ivp OdeResult, with an additional `violations`
            attribute containing all detected violations.
        """
        _require_scipy()
        from scipy.integrate import solve_ivp

        mapper = state_mapper or default_state_mapper
        violations: list[Violation] = []

        # Monitor the initial state
        initial_state = mapper(np.asarray(y0), t_span[0], 0)
        init_violations = monitor.check(initial_state)
        violations.extend(init_violations)

        if use_rhs_wrapper:
            # Wrap the RHS function to monitor at every evaluation
            wrapped_fun = StepCallback(
                rhs=fun,
                monitor=monitor,
                state_mapper=mapper,
                violations_out=violations,
                check_every=rhs_check_every,
            )
            actual_fun: Any = wrapped_fun
            actual_events = events
        else:
            # Use the events mechanism for step-level monitoring
            monitor_event = _MonitorEvent(
                monitor=monitor,
                state_mapper=mapper,
                violations_out=violations,
            )
            actual_fun: Any = fun  # type: ignore[no-redef]

            # Combine with user events
            if events is not None:
                if callable(events):
                    actual_events = [events, monitor_event]
                else:
                    actual_events = list(events) + [monitor_event]
            else:
                actual_events = [monitor_event]

        result = solve_ivp(
            fun=actual_fun,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            dense_output=dense_output,
            events=actual_events,
            **solve_ivp_kwargs,
        )

        # Monitor the final state
        if result.success and result.t.size > 0:
            final_state = mapper(result.y[:, -1], result.t[-1], -1)
            final_violations = monitor.check(final_state)
            violations.extend(final_violations)

        # Attach violations to the result
        result.violations = violations  # type: ignore[attr-defined]
        return result

    @staticmethod
    def monitored_odeint(
        func: Callable[[np.ndarray, float], np.ndarray],
        y0: np.ndarray,
        t: np.ndarray,
        monitor: Monitor,
        state_mapper: StateMapper | None = None,
        **odeint_kwargs: Any,
    ) -> tuple[np.ndarray, list[Violation]]:
        """Solve an ODE with odeint while monitoring invariants.

        Wraps scipy.integrate.odeint and monitors invariants at each
        output point in the t array.

        Note: odeint does not support events, so monitoring happens at
        the user-requested output times, not at internal solver steps.
        For step-level monitoring, use monitored_solve_ivp instead.

        Args:
            func: Right-hand side function dy/dt = func(y, t).
                  Note the argument order is (y, t), not (t, y).
            y0: Initial state vector.
            t: Array of time points at which to solve.
            monitor: The Monitor instance.
            state_mapper: Converts (y, t, step) to SimulationState.
            **odeint_kwargs: Additional kwargs passed to odeint.

        Returns:
            Tuple of (solution_array, violations_list).
        """
        _require_scipy()
        from scipy.integrate import odeint

        mapper = state_mapper or default_state_mapper
        violations: list[Violation] = []

        # Run odeint
        solution = odeint(func, y0, t, **odeint_kwargs)

        # Monitor at each output point
        for step_idx in range(len(t)):
            state = mapper(solution[step_idx], t[step_idx], step_idx)
            step_violations = monitor.check(state)
            violations.extend(step_violations)

        return solution, violations

    @staticmethod
    def create_dense_monitor(
        monitor: Monitor,
        state_mapper: StateMapper | None = None,
    ) -> Callable[[float, np.ndarray], list[Violation]]:
        """Create a callback for manual integration loops.

        For users who call the solver step-by-step rather than using
        solve_ivp/odeint, this provides a callback they can invoke
        at each step.

        Args:
            monitor: The Monitor instance.
            state_mapper: Converts (y, t, step) to SimulationState.

        Returns:
            A callback function with signature (t, y) -> violations_list.
        """
        mapper = state_mapper or default_state_mapper
        step_counter = [0]

        def callback(t: float, y: np.ndarray) -> list[Violation]:
            step_counter[0] += 1
            state = mapper(y, t, step_counter[0])
            return monitor.check(state)

        return callback
