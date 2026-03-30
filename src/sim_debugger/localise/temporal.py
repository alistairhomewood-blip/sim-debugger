"""Temporal localisation: find when a violation first appeared.

Given a violation detected at timestep T, uses the state history ring buffer
to determine:
- The earliest timestep T_0 where the invariant began to deviate
- The violation pattern (sudden, gradual, oscillatory, divergent)
- The trajectory of invariant values from T_0 to T
"""

from __future__ import annotations

from sim_debugger.core.state import StateHistory
from sim_debugger.core.violations import (
    TemporalLocalisation,
    ViolationPattern,
)


def localise_temporal(
    invariant_name: str,
    history: StateHistory,
    threshold: float,
    current_timestep: int,  # noqa: F841
) -> TemporalLocalisation | None:
    """Find the temporal origin and pattern of a violation.

    Uses the invariant value trajectory stored in the state history to
    binary-search for the first timestep where the invariant exceeded
    tolerance, and classifies the violation pattern.

    Args:
        invariant_name: Name of the violated invariant.
        history: The state history ring buffer.
        threshold: The invariant's violation threshold.
        current_timestep: The timestep at which the violation was detected.
            TODO: Use this as an upper bound for the search window and
            to handle cases where the history buffer does not extend to
            the current timestep.

    Returns:
        TemporalLocalisation describing when and how the violation appeared,
        or None if insufficient history is available.
    """
    # TODO: use current_timestep as upper bound for search
    _ = current_timestep
    trajectory = history.get_invariant_trajectory(invariant_name)
    if len(trajectory) < 2:
        return None

    # Find the reference value (first value in history, assumed correct)
    ref_timestep, ref_value = trajectory[0]

    # Build list of (timestep, value, relative_error) tuples
    errors: list[tuple[int, float, float]] = []
    for ts, val in trajectory:
        if abs(ref_value) > 1e-300:
            rel_err = abs(val - ref_value) / abs(ref_value)
        else:
            rel_err = abs(val - ref_value)
        errors.append((ts, val, rel_err))

    # Binary search for first timestep exceeding threshold
    first_violation_idx = _find_first_violation(errors, threshold)
    if first_violation_idx is None:
        # No violation found in the history
        return None

    first_violation_timestep = errors[first_violation_idx][0]

    # Extract the violation trajectory from first violation to current
    violation_traj = [
        (ts, val) for ts, val, _ in errors[first_violation_idx:]
    ]

    # Classify the pattern
    pattern = _classify_pattern(errors, first_violation_idx, threshold)

    return TemporalLocalisation(
        first_violation_timestep=first_violation_timestep,
        pattern=pattern,
        violation_trajectory=violation_traj,
    )


def _find_first_violation(
    errors: list[tuple[int, float, float]],
    threshold: float,
) -> int | None:
    """Binary search for the first index where relative error exceeds threshold.

    Args:
        errors: List of (timestep, value, relative_error) sorted by timestep.
        threshold: Violation threshold.

    Returns:
        Index of the first violating entry, or None.
    """
    lo, hi = 0, len(errors) - 1

    # Check that we actually have a violation at the end
    if errors[hi][2] <= threshold:
        return None

    # Check if violation exists from the start
    if errors[lo][2] > threshold:
        return 0

    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        if errors[mid][2] > threshold:
            hi = mid
        else:
            lo = mid + 1

    return lo


def _classify_pattern(
    errors: list[tuple[int, float, float]],
    first_violation_idx: int,
    threshold: float,
) -> ViolationPattern:
    """Classify the violation pattern from the error trajectory.

    Patterns:
    - SUDDEN: violation appears in a single timestep
    - GRADUAL: monotonic increase over many timesteps
    - OSCILLATORY: error oscillates with growing amplitude
    - DIVERGENT: exponential growth

    Args:
        errors: List of (timestep, value, relative_error).
        first_violation_idx: Index where violation first exceeded threshold.
        threshold: Violation threshold.

    Returns:
        The classified ViolationPattern.
    """
    # If violation is only in the last timestep, it's sudden
    if first_violation_idx >= len(errors) - 2:
        return ViolationPattern.SUDDEN

    # Extract the error values from first violation onward
    violation_errors = [e[2] for e in errors[first_violation_idx:]]
    n = len(violation_errors)

    if n < 3:
        return ViolationPattern.SUDDEN

    # Check for oscillatory pattern: count sign changes in the delta
    deltas = [violation_errors[i + 1] - violation_errors[i] for i in range(n - 1)]
    sign_changes = sum(
        1 for i in range(len(deltas) - 1)
        if deltas[i] * deltas[i + 1] < 0
    )

    if sign_changes > len(deltas) * 0.3:
        return ViolationPattern.OSCILLATORY

    # Check for divergent (exponential) growth:
    # If each step's error roughly doubles, it's divergent
    if n >= 4 and all(e > 0 for e in violation_errors):
        ratios = [
            violation_errors[i + 1] / violation_errors[i]
            for i in range(n - 1)
            if violation_errors[i] > 1e-300
        ]
        if ratios and all(r > 1.5 for r in ratios):
            return ViolationPattern.DIVERGENT

    # Check for gradual (monotonic increase)
    increasing = sum(1 for d in deltas if d > 0)
    if increasing > len(deltas) * 0.7:
        return ViolationPattern.GRADUAL

    return ViolationPattern.GRADUAL  # Default
