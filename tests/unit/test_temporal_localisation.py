"""Tests for temporal localisation."""


from sim_debugger.core.state import SimulationState, StateHistory
from sim_debugger.core.violations import ViolationPattern
from sim_debugger.localise.temporal import localise_temporal


def _build_history(values: list[float], invariant_name: str = "energy") -> StateHistory:
    """Build a state history with the given invariant values."""
    history = StateHistory(max_size=len(values) + 10)
    for i, val in enumerate(values):
        state = SimulationState(timestep=i, time=float(i))
        history.push(state, {invariant_name: val})
    return history


class TestTemporalLocalisation:
    def test_sudden_violation(self):
        """Single-timestep jump detected as SUDDEN."""
        values = [1.0] * 10 + [2.0]  # Jump at step 10
        history = _build_history(values)
        result = localise_temporal("energy", history, threshold=0.01, current_timestep=10)

        assert result is not None
        assert result.pattern == ViolationPattern.SUDDEN
        assert result.first_violation_timestep == 10

    def test_gradual_violation(self):
        """Slow drift detected as GRADUAL."""
        # Values slowly increase beyond threshold
        values = [1.0 + i * 0.001 for i in range(20)]
        history = _build_history(values)
        result = localise_temporal("energy", history, threshold=0.001, current_timestep=19)

        assert result is not None
        assert result.pattern == ViolationPattern.GRADUAL
        assert result.first_violation_timestep < 19

    def test_divergent_violation(self):
        """Exponential growth detected as DIVERGENT."""
        values = [1.0 * (2.0 ** i) for i in range(10)]
        history = _build_history(values)
        result = localise_temporal("energy", history, threshold=0.01, current_timestep=9)

        assert result is not None
        assert result.pattern == ViolationPattern.DIVERGENT

    def test_insufficient_history(self):
        """Returns None with insufficient history."""
        history = StateHistory(max_size=10)
        state = SimulationState(timestep=0, time=0.0)
        history.push(state, {"energy": 1.0})
        result = localise_temporal("energy", history, threshold=0.01, current_timestep=0)
        assert result is None

    def test_no_violation_in_history(self):
        """All values within threshold returns None."""
        values = [1.0] * 10
        history = _build_history(values)
        result = localise_temporal("energy", history, threshold=0.01, current_timestep=9)
        # All values are identical to reference, relative error is 0, which is within threshold
        assert result is None

    def test_violation_trajectory(self):
        """Violation trajectory captures the correct values."""
        values = [1.0] * 5 + [1.1, 1.2, 1.5, 2.0]
        history = _build_history(values)
        result = localise_temporal("energy", history, threshold=0.01, current_timestep=8)

        assert result is not None
        assert len(result.violation_trajectory) > 0
        # Trajectory should start from the first violating timestep
        first_ts = result.violation_trajectory[0][0]
        assert first_ts >= 5
