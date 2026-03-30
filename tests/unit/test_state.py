"""Tests for SimulationState and StateHistory."""

import numpy as np
import pytest

from sim_debugger.core.state import SimulationState, StateHistory


class TestSimulationState:
    def test_create_state(self):
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={"positions": np.array([[1.0, 0.0, 0.0]])},
            metadata={"dt": 0.01},
        )
        assert state.timestep == 0
        assert state.time == 0.0
        assert state.has_array("positions")
        assert not state.has_array("velocities")

    def test_get_array(self):
        arr = np.array([[1.0, 2.0, 3.0]])
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={"positions": arr},
        )
        result = state.get_array("positions")
        np.testing.assert_array_equal(result, arr)

    def test_get_array_missing(self):
        state = SimulationState(timestep=0, time=0.0)
        with pytest.raises(KeyError, match="positions"):
            state.get_array("positions")

    def test_copy_deep_copies_arrays(self):
        arr = np.array([[1.0, 2.0, 3.0]])
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={"positions": arr},
        )
        copied = state.copy()
        # Modify original
        arr[0, 0] = 999.0
        # Copy should not be affected
        assert copied.get_array("positions")[0, 0] == 1.0

    def test_copy_preserves_metadata(self):
        state = SimulationState(
            timestep=5, time=0.05,
            metadata={"dt": 0.01, "grid": [10, 10]},
        )
        copied = state.copy()
        assert copied.timestep == 5
        assert copied.time == 0.05
        assert copied.metadata["dt"] == 0.01


class TestStateHistory:
    def test_push_and_retrieve(self):
        history = StateHistory(max_size=10)
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={"x": np.array([1.0])},
        )
        history.push(state, {"energy": 1.0})
        assert len(history) == 1
        assert history.latest is not None
        assert history.latest.timestep == 0

    def test_ring_buffer_max_size(self):
        history = StateHistory(max_size=5)
        for i in range(10):
            state = SimulationState(timestep=i, time=float(i))
            history.push(state, {"energy": float(i)})

        assert len(history) == 5
        # Should keep the most recent 5
        assert history.latest.timestep == 9

    def test_invariant_trajectory(self):
        history = StateHistory(max_size=100)
        for i in range(10):
            state = SimulationState(timestep=i, time=float(i))
            history.push(state, {"energy": 1.0 + i * 0.001})

        traj = history.get_invariant_trajectory("energy")
        assert len(traj) == 10
        assert traj[0] == (0, 1.0)
        assert traj[9] == (9, pytest.approx(1.009))

    def test_get_state_at(self):
        history = StateHistory(max_size=100, full_copy_count=5)
        for i in range(10):
            state = SimulationState(
                timestep=i, time=float(i),
                arrays={"x": np.array([float(i)])},
            )
            history.push(state)

        # Recent states should have arrays
        recent = history.get_state_at(9)
        assert recent is not None
        assert recent.has_array("x")

    def test_get_recent_states(self):
        history = StateHistory(max_size=100)
        for i in range(20):
            state = SimulationState(timestep=i, time=float(i))
            history.push(state)

        recent = history.get_recent_states(5)
        assert len(recent) == 5
        assert recent[-1].timestep == 19

    def test_clear(self):
        history = StateHistory(max_size=10)
        state = SimulationState(timestep=0, time=0.0)
        history.push(state)
        assert len(history) == 1
        history.clear()
        assert len(history) == 0
        assert history.latest is None
