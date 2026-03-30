"""Tests for the Monitor orchestration class."""

import numpy as np

from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState


def make_state(
    timestep: int,
    positions: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
    masses: np.ndarray | None = None,
    **extra,
) -> SimulationState:
    arrays = {}
    metadata = {"dt": 0.01}
    if positions is not None:
        arrays["positions"] = positions
    if velocities is not None:
        arrays["velocities"] = velocities
    if masses is not None:
        arrays["masses"] = masses
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            metadata[k] = v
    return SimulationState(
        timestep=timestep, time=timestep * 0.01,
        arrays=arrays, metadata=metadata,
    )


class TestMonitorBasic:
    def test_no_violations_on_conserved(self):
        """Monitor reports no violations when invariants are conserved."""
        monitor = Monitor(invariants=["Total Energy"])
        m = np.array([1.0])
        for t in range(10):
            state = make_state(
                timestep=t,
                velocities=np.array([[1.0, 0.0, 0.0]]),
                masses=m,
            )
            violations = monitor.check(state)
            assert violations == []

        assert len(monitor.violations) == 0

    def test_violation_detected(self):
        """Monitor detects energy violation when energy changes."""
        monitor = Monitor(invariants=["Total Energy"])
        m = np.array([1.0])

        # First state: E = 0.5
        state0 = make_state(
            timestep=0,
            velocities=np.array([[1.0, 0.0, 0.0]]),
            masses=m,
        )
        monitor.check(state0)

        # Second state: E = 2.0 (big jump)
        state1 = make_state(
            timestep=1,
            velocities=np.array([[2.0, 0.0, 0.0]]),
            masses=m,
        )
        violations = monitor.check(state1)

        assert len(violations) == 1
        assert violations[0].invariant_name == "Total Energy"
        assert violations[0].timestep == 1
        assert violations[0].explanation is not None

    def test_auto_detect_invariants(self):
        """Monitor auto-detects applicable invariants from state."""
        monitor = Monitor()
        state = make_state(
            timestep=0,
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.array([[0.0, 1.0, 0.0]]),
            masses=np.array([1.0]),
            charges=np.array([1.0]),
        )
        monitor.check(state)
        names = monitor.active_invariants
        assert "Total Energy" in names
        assert "Linear Momentum" in names

    def test_report_no_violations(self):
        monitor = Monitor(invariants=["Total Energy"])
        state = make_state(
            timestep=0,
            velocities=np.array([[1.0, 0.0, 0.0]]),
            masses=np.array([1.0]),
        )
        monitor.check(state)
        report = monitor.report()
        assert "No violations detected" in report

    def test_report_with_violations(self):
        monitor = Monitor(invariants=["Total Energy"])
        m = np.array([1.0])

        monitor.check(make_state(0, velocities=np.array([[1.0, 0.0, 0.0]]), masses=m))
        monitor.check(make_state(1, velocities=np.array([[10.0, 0.0, 0.0]]), masses=m))

        report = monitor.report()
        assert "Total Energy" in report
        assert "Violations:" in report or "Total violations:" in report

    def test_check_interval(self):
        """Check interval > 1 skips intermediate timesteps."""
        monitor = Monitor(invariants=["Total Energy"], check_interval=5)
        m = np.array([1.0])

        violations_count = 0
        for t in range(20):
            state = make_state(
                timestep=t,
                velocities=np.array([[1.0 + t * 0.1, 0.0, 0.0]]),
                masses=m,
            )
            violations = monitor.check(state)
            violations_count += len(violations)

        # Only checked at steps 0 (init), 5, 10, 15, 20
        # Violations would be detected at the check points
        assert monitor.step_count == 20

    def test_reset(self):
        """Monitor can be reset to initial state."""
        monitor = Monitor(invariants=["Total Energy"])
        m = np.array([1.0])
        monitor.check(make_state(0, velocities=np.array([[1.0, 0.0, 0.0]]), masses=m))
        monitor.check(make_state(1, velocities=np.array([[10.0, 0.0, 0.0]]), masses=m))
        assert len(monitor.violations) > 0

        monitor.reset()
        assert len(monitor.violations) == 0
        assert monitor.step_count == 0


class TestMonitorHarmonicOscillator:
    """Integration test: Monitor on a harmonic oscillator."""

    def test_leapfrog_conserves_energy(self):
        """Symplectic Euler integrator with small dt should conserve energy."""
        # Use a threshold that tolerates the O(dt) per-step energy error
        # of symplectic Euler but still catches actual bugs.
        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-3},
        )

        x = np.array([[1.0, 0.0, 0.0]])
        v = np.array([[0.0, 0.0, 0.0]])
        m = np.array([1.0])
        dt = 0.001  # Small dt for good conservation

        for t in range(200):
            # Symplectic Euler
            a = -x
            v = v + a * dt
            x = x + v * dt

            state = SimulationState(
                timestep=t, time=t * dt,
                arrays={
                    "positions": x.copy(),
                    "velocities": v.copy(),
                    "masses": m.copy(),
                    "potential_energy": np.array([0.5 * np.sum(x**2)]),
                },
                metadata={"dt": dt},
            )
            monitor.check(state)

        # Small dt + symplectic integrator should have no violations at 1e-3 threshold
        assert len(monitor.violations) == 0

    def test_euler_forward_violates_energy(self):
        """Forward Euler on harmonic oscillator causes energy growth."""
        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-4},
        )

        x = np.array([[1.0, 0.0, 0.0]])
        v = np.array([[0.0, 0.0, 0.0]])
        m = np.array([1.0])
        dt = 0.1  # Larger dt to make violation visible

        for t in range(200):
            a = -x
            # Forward Euler (non-symplectic): both update from old state
            x_new = x + v * dt
            v_new = v + a * dt
            x = x_new
            v = v_new

            state = SimulationState(
                timestep=t, time=t * dt,
                arrays={
                    "positions": x.copy(),
                    "velocities": v.copy(),
                    "masses": m.copy(),
                    "potential_energy": np.array([0.5 * np.sum(x**2)]),
                },
                metadata={"dt": dt},
            )
            monitor.check(state)

        # Forward Euler should cause energy growth -> violations
        assert len(monitor.violations) > 0
        # First violation should have an explanation
        v0 = monitor.violations[0]
        assert v0.explanation is not None
        assert "energy" in v0.explanation.lower() or "Energy" in v0.explanation
