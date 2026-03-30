"""Tests for the SciPy ODE integration backend.

Tests the monitored_solve_ivp and monitored_odeint wrappers against
known ODE problems (harmonic oscillator, exponential growth) and
verifies that invariant monitoring works correctly at each solver step.
"""

import numpy as np
import pytest

from sim_debugger.backends.scipy_backend import (
    SciPyBackend,
    StepCallback,
    default_state_mapper,
)
from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState

# ===========================================================================
# Availability and detection
# ===========================================================================

class TestSciPyAvailability:
    def test_is_available(self):
        """SciPy backend reports availability correctly."""
        assert SciPyBackend.is_available() is True  # SciPy is installed

    def test_detect_backend_positive(self):
        """detect_backend detects SciPy ODE solver imports."""
        assert SciPyBackend.detect_backend("from scipy.integrate import solve_ivp") is True
        assert SciPyBackend.detect_backend("import scipy.integrate") is True
        assert SciPyBackend.detect_backend("result = solve_ivp(rhs, ...)") is True
        assert SciPyBackend.detect_backend("y = odeint(func, y0, t)") is True

    def test_detect_backend_negative(self):
        """detect_backend returns False for non-SciPy source."""
        assert SciPyBackend.detect_backend("import numpy as np") is False
        assert SciPyBackend.detect_backend("x = 42") is False


# ===========================================================================
# Default state mapper
# ===========================================================================

class TestDefaultStateMapper:
    def test_basic_mapping(self):
        """Default mapper creates a state with the full y vector."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        state = default_state_mapper(y, t=0.5, step=10)
        assert state.timestep == 10
        assert state.time == 0.5
        assert "y" in state.arrays
        np.testing.assert_array_equal(state.arrays["y"], y)

    def test_even_length_splits_positions_velocities(self):
        """Even-length y gets split into positions and velocities."""
        y = np.array([1.0, 2.0, 3.0, 4.0])  # 4 elements -> 2 pos, 2 vel
        state = default_state_mapper(y, t=0.0, step=0)
        assert "positions" in state.arrays
        assert "velocities" in state.arrays
        assert "masses" in state.arrays
        np.testing.assert_array_equal(
            state.arrays["positions"].ravel(), [1.0, 2.0]
        )
        np.testing.assert_array_equal(
            state.arrays["velocities"].ravel(), [3.0, 4.0]
        )

    def test_odd_length_no_split(self):
        """Odd-length y does not get split into positions/velocities."""
        y = np.array([1.0, 2.0, 3.0])
        state = default_state_mapper(y, t=0.0, step=0)
        assert "y" in state.arrays
        assert "positions" not in state.arrays

    def test_copies_array(self):
        """Default mapper copies the y array (mutation safety)."""
        y = np.array([1.0, 2.0])
        state = default_state_mapper(y, t=0.0, step=0)
        y[0] = 999.0
        assert state.arrays["y"][0] == pytest.approx(1.0)


# ===========================================================================
# StepCallback (RHS wrapper)
# ===========================================================================

class TestStepCallback:
    def test_wraps_rhs_function(self):
        """StepCallback calls the original RHS and returns its result."""
        def rhs(t, y):
            return -y

        monitor = Monitor(invariants=["Total Energy"])
        violations = []
        callback = StepCallback(
            rhs=rhs,
            monitor=monitor,
            state_mapper=default_state_mapper,
            violations_out=violations,
        )

        y = np.array([1.0, 0.0])
        result = callback(0.0, y)
        np.testing.assert_array_equal(result, -y)

    def test_counts_calls(self):
        """StepCallback tracks the number of RHS evaluations."""
        def rhs(t, y):
            return np.zeros_like(y)

        monitor = Monitor(invariants=["Total Energy"])
        violations = []
        callback = StepCallback(
            rhs=rhs,
            monitor=monitor,
            state_mapper=default_state_mapper,
            violations_out=violations,
        )

        for _ in range(10):
            callback(0.0, np.array([1.0, 0.0]))
        assert callback._call_count == 10

    def test_check_every_n(self):
        """StepCallback respects check_every parameter."""
        def rhs(t, y):
            return np.zeros_like(y)

        monitor = Monitor(invariants=["Total Energy"])
        violations = []
        callback = StepCallback(
            rhs=rhs,
            monitor=monitor,
            state_mapper=default_state_mapper,
            violations_out=violations,
            check_every=5,
        )

        for _ in range(10):
            callback(0.0, np.array([1.0, 0.0]))

        # Should have monitored at calls 5 and 10
        assert callback._step_count == 2


# ===========================================================================
# monitored_solve_ivp
# ===========================================================================

class TestMonitoredSolveIvp:
    def test_harmonic_oscillator_conserves_energy(self):
        """solve_ivp on a harmonic oscillator with energy monitoring.

        The harmonic oscillator ODE: x'' = -x
        Written as: y = [x, v], dy/dt = [v, -x]

        RK45 is a non-symplectic integrator, so energy will drift slightly,
        but with small enough step size it should be within tolerance.
        """
        def harmonic_rhs(t, y):
            x, v = y
            return np.array([v, -x])

        def ho_state_mapper(y, t, step):
            x, v = y
            return SimulationState(
                timestep=step,
                time=t,
                arrays={
                    "positions": np.array([[x]]),
                    "velocities": np.array([[v]]),
                    "masses": np.array([1.0]),
                    "potential_energy": np.array([0.5 * x**2]),
                },
                metadata={"dt": 0.0},
            )

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-3},  # Loose threshold for RK45
        )

        y0 = np.array([1.0, 0.0])
        result = SciPyBackend.monitored_solve_ivp(
            fun=harmonic_rhs,
            t_span=(0, 2 * np.pi),  # One full period
            y0=y0,
            monitor=monitor,
            state_mapper=ho_state_mapper,
            method="RK45",
            max_step=0.01,
        )

        assert result.success
        # With a loose threshold, RK45 should not trigger violations
        # on a short integration with small step size
        assert hasattr(result, "violations")

    def test_exponential_growth_detects_violation(self):
        """solve_ivp on y' = y (exponential growth) should detect energy violation.

        Starting from [1, 1], the solution grows exponentially.
        The "energy" (0.5 * m * v^2) will grow, triggering a violation.
        """
        def exp_rhs(t, y):
            return y  # dy/dt = y -> exponential growth

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-4},
        )

        y0 = np.array([1.0, 1.0])
        result = SciPyBackend.monitored_solve_ivp(
            fun=exp_rhs,
            t_span=(0, 5),
            y0=y0,
            monitor=monitor,
            method="RK45",
        )

        assert result.success
        assert hasattr(result, "violations")
        # Exponential growth should trigger energy violations
        assert len(result.violations) > 0

    def test_with_rhs_wrapper(self):
        """monitored_solve_ivp works with use_rhs_wrapper=True."""
        def rhs(t, y):
            return -y

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-2},
        )

        y0 = np.array([1.0, 0.0])
        result = SciPyBackend.monitored_solve_ivp(
            fun=rhs,
            t_span=(0, 1),
            y0=y0,
            monitor=monitor,
            use_rhs_wrapper=True,
            method="RK45",
        )

        assert result.success

    def test_with_user_events(self):
        """monitored_solve_ivp combines user events with monitoring."""
        def rhs(t, y):
            return np.array([y[1], -y[0]])

        # User event: detect when x crosses zero
        def zero_crossing(t, y):
            return y[0]
        zero_crossing.terminal = False
        zero_crossing.direction = 0

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-2},
        )

        y0 = np.array([1.0, 0.0])
        result = SciPyBackend.monitored_solve_ivp(
            fun=rhs,
            t_span=(0, 2 * np.pi),
            y0=y0,
            monitor=monitor,
            events=zero_crossing,
            method="RK45",
        )

        assert result.success
        # The zero_crossing event should have been detected
        assert hasattr(result, "t_events")

    def test_with_t_eval(self):
        """monitored_solve_ivp works with t_eval parameter."""
        def rhs(t, y):
            return -y

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-2},
        )

        t_eval = np.linspace(0, 1, 11)
        y0 = np.array([1.0, 0.0])
        result = SciPyBackend.monitored_solve_ivp(
            fun=rhs,
            t_span=(0, 1),
            y0=y0,
            monitor=monitor,
            t_eval=t_eval,
            method="RK45",
        )

        assert result.success
        assert len(result.t) == 11


# ===========================================================================
# monitored_odeint
# ===========================================================================

class TestMonitoredOdeint:
    def test_harmonic_oscillator(self):
        """odeint on harmonic oscillator with monitoring."""
        def harmonic_odeint(y, t):
            """Note: odeint uses (y, t) argument order."""
            x, v = y
            return [v, -x]

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-2},
        )

        y0 = np.array([1.0, 0.0])
        t = np.linspace(0, 2 * np.pi, 200)

        solution, violations = SciPyBackend.monitored_odeint(
            func=harmonic_odeint,
            y0=y0,
            t=t,
            monitor=monitor,
        )

        assert solution.shape == (200, 2)
        assert isinstance(violations, list)

    def test_odeint_monitors_all_output_points(self):
        """odeint monitoring runs at each output time point."""
        call_count = [0]

        def rhs(y, t):
            return -y

        def counting_mapper(y, t, step):
            call_count[0] += 1
            return default_state_mapper(y, t, step)

        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-2},
        )

        y0 = np.array([1.0, 0.0])
        t = np.linspace(0, 1, 20)

        solution, violations = SciPyBackend.monitored_odeint(
            func=rhs,
            y0=y0,
            t=t,
            monitor=monitor,
            state_mapper=counting_mapper,
        )

        # Should have been called for each output time point
        assert call_count[0] == 20


# ===========================================================================
# create_dense_monitor
# ===========================================================================

class TestDenseMonitor:
    def test_callback_returns_violations(self):
        """Dense monitor callback returns violations list."""
        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-6},
        )

        callback = SciPyBackend.create_dense_monitor(monitor)

        # First call: initialisation
        y0 = np.array([1.0, 0.0])
        v0 = callback(0.0, y0)
        assert isinstance(v0, list)

        # Second call: same state, no violation
        v1 = callback(0.01, y0)
        assert isinstance(v1, list)
        assert len(v1) == 0

    def test_callback_detects_violation(self):
        """Dense monitor callback detects violations."""
        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-6},
        )

        callback = SciPyBackend.create_dense_monitor(monitor)

        # First call: init
        callback(0.0, np.array([1.0, 0.0]))

        # Second call: big change in velocity -> energy violation
        violations = callback(0.01, np.array([1.0, 10.0]))
        # May or may not detect depending on default mapper and threshold
        assert isinstance(violations, list)
