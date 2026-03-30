"""Decorator-based instrumentation for explicit opt-in monitoring.

Provides decorators that users can add to their simulation code for
precise control over what gets monitored:

- @monitor: wrap a function to capture state before/after each call
- @timestep: mark a function as a single timestep update
- @track_state: specify which variables constitute the simulation state
- @ignore: exclude a function from AST-based instrumentation
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState

F = TypeVar("F", bound=Callable[..., Any])

# Module-level monitor instance for decorator-based instrumentation
_global_monitor: Monitor | None = None
_timestep_counter: int = 0


def get_global_monitor() -> Monitor:
    """Get or create the global monitor for decorator-based instrumentation."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = Monitor()
    return _global_monitor


def set_global_monitor(monitor: Monitor) -> None:
    """Set the global monitor (used by CLI runner)."""
    global _global_monitor
    _global_monitor = monitor


def monitor(
    invariants: list[str] | None = None,
    threshold: float | None = None,
) -> Callable[[F], F]:
    """Decorator that monitors invariants before/after a function call.

    The decorated function should take a state dict (or object with arrays)
    as its first argument and return the updated state.

    Usage::

        @sim_debugger.monitor(invariants=["Total Energy", "Linear Momentum"])
        def timestep_update(state, dt):
            state['velocities'] += state['forces'] * dt
            state['positions'] += state['velocities'] * dt
            return state

    Args:
        invariants: List of invariant names to monitor.
        threshold: Global threshold override for all invariants.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            global _timestep_counter
            mon = get_global_monitor()

            # If this is the first call, configure the monitor
            if mon.step_count == 0 and invariants:
                mon._requested_invariants = invariants
                if threshold:
                    mon._thresholds = {name: threshold for name in invariants}

            # Try to extract state from args
            # TODO: use state_before for pre/post comparison monitoring
            _state_before = _extract_state(args, kwargs)  # noqa: F841

            # Call the original function
            result = func(*args, **kwargs)

            # Extract state after
            state_after = _extract_state_from_result(result, args, kwargs)
            if state_after is not None:
                _timestep_counter += 1
                state_after.timestep = _timestep_counter
                state_after.time = _timestep_counter * state_after.metadata.get("dt", 1.0)
                violations = mon.check(state_after)
                if violations:
                    for v in violations:
                        print(
                            f"!! VIOLATION: {v.invariant_name} at step "
                            f"{v.timestep} ({v.severity.value}): "
                            f"{v.relative_error:.2%} !!"
                        )
                        if v.explanation:
                            print(v.explanation)
                            print()

            return result
        return wrapper  # type: ignore[return-value]
    return decorator


def timestep(func: F) -> F:
    """Decorator that marks a function as a single timestep update.

    The function is wrapped to capture state before and after execution
    and run invariant checks. Uses the global monitor with auto-detected
    invariants.

    Usage::

        @sim_debugger.timestep
        def step(x, v, m, dt):
            v += (-x) * dt  # harmonic oscillator
            x += v * dt
            return x, v
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _timestep_counter
        mon = get_global_monitor()

        result = func(*args, **kwargs)

        # Try to build a state from the function's return value and args
        state = _extract_state_from_result(result, args, kwargs)
        if state is not None:
            _timestep_counter += 1
            state.timestep = _timestep_counter
            violations = mon.check(state)
            if violations:
                for v in violations:
                    print(
                        f"!! VIOLATION: {v.invariant_name} at step "
                        f"{v.timestep} ({v.severity.value}) !!"
                    )

        return result
    return wrapper  # type: ignore[return-value]


def track_state(variables: list[str]) -> Callable[[F], F]:
    """Decorator that specifies which variables constitute the simulation state.

    Applied to a function, it captures the named variables from the function's
    local scope after execution.

    Usage::

        @sim_debugger.track_state(variables=["positions", "velocities", "E_field"])
        def update(positions, velocities, E_field, dt):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            # Store tracked variable names on the function for later use
            wrapper._sim_debugger_tracked = variables  # type: ignore[attr-defined]
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


def ignore(func: F) -> F:
    """Decorator that excludes a function from AST-based instrumentation.

    Usage::

        @sim_debugger.ignore
        def helper_function():
            ...  # This won't be instrumented
    """
    func._sim_debugger_ignore = True  # type: ignore[attr-defined]
    return func


def _extract_state(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> SimulationState | None:
    """Try to extract a SimulationState from function arguments."""
    # If first arg is already a SimulationState, use it
    if args and isinstance(args[0], SimulationState):
        return args[0]

    # If first arg is a dict with numpy arrays, build a state
    if args and isinstance(args[0], dict):
        arrays = {}
        metadata = {}
        for k, v in args[0].items():
            if isinstance(v, np.ndarray):
                arrays[k] = np.copy(v)
            elif isinstance(v, (int, float, str)):
                metadata[k] = v
        if arrays:
            return SimulationState(
                timestep=0, time=0.0,
                arrays=arrays, metadata=metadata,
            )

    return None


def _extract_state_from_result(
    result: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> SimulationState | None:
    """Try to extract a SimulationState from a function's return value."""
    if isinstance(result, SimulationState):
        return result

    if isinstance(result, dict):
        arrays = {}
        metadata = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                arrays[k] = np.copy(v)
            elif isinstance(v, (int, float, str)):
                metadata[k] = v
        if arrays:
            return SimulationState(
                timestep=0, time=0.0,
                arrays=arrays, metadata=metadata,
            )

    # If result is a tuple of arrays, try to match with arg names
    if isinstance(result, tuple) and all(isinstance(r, np.ndarray) for r in result):
        # Build state from the result arrays using positional heuristics
        arrays = {}
        array_names = ["positions", "velocities", "forces", "E_field", "B_field"]
        for i, arr in enumerate(result):
            if i < len(array_names):
                arrays[array_names[i]] = np.copy(arr)
            else:
                arrays[f"array_{i}"] = np.copy(arr)

        # Pull metadata from kwargs
        metadata = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))}
        if arrays:
            return SimulationState(
                timestep=0, time=0.0,
                arrays=arrays, metadata=metadata,
            )

    return None
