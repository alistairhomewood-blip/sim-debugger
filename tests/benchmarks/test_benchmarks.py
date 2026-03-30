"""Benchmark test suite: verify zero false negatives on known-buggy simulations.

Each benchmark runs a simulation through the Monitor and verifies:
- Correct simulations: zero violations detected at default threshold
- Buggy simulations: violation detected within 10 timesteps of bug onset

This is the critical physics correctness test suite.
100% pass rate required before any release.
"""


from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState
from tests.benchmarks.simulations import (
    b01_harmonic_euler,
    b02_harmonic_large_dt,
    b03_boris_correct,
    b04_boris_large_dt,
    b05_boris_wrong_halfstep,
    b06_boris_wrong_sign,
    b09_lorentz_correct,
    b10_lorentz_wrong_sign,
    b11_nbody_correct,
    b12_nbody_asymmetric,
    b13_boundary_reflecting,
    b14_boundary_leak,
)


def _run_monitor(
    states: list[SimulationState],
    invariants: list[str],
    thresholds: dict[str, float] | None = None,
) -> Monitor:
    """Run a monitor over a sequence of states."""
    monitor = Monitor(invariants=invariants, thresholds=thresholds or {})
    for state in states:
        monitor.check(state)
    return monitor


# ===========================================================================
# B01: Forward Euler -> energy growth (MUST detect)
# ===========================================================================

class TestB01EulerEnergyGrowth:
    def test_violation_detected(self):
        """Forward Euler causes monotonic energy growth. Must be detected."""
        states = b01_harmonic_euler(num_steps=200, dt=0.05)
        monitor = _run_monitor(states, ["Total Energy"])
        assert len(monitor.violations) > 0, (
            "B01: Forward Euler energy growth NOT detected (false negative)"
        )

    def test_violation_has_explanation(self):
        states = b01_harmonic_euler(num_steps=200, dt=0.05)
        monitor = _run_monitor(states, ["Total Energy"])
        assert monitor.violations[0].explanation is not None


# ===========================================================================
# B02: Leapfrog large dt -> exponential energy growth (MUST detect)
# ===========================================================================

class TestB02LargeTimestep:
    def test_violation_detected(self):
        """Leapfrog with dt > stability limit causes divergence. Must detect."""
        states = b02_harmonic_large_dt(num_steps=50, dt=2.5)
        monitor = _run_monitor(states, ["Total Energy"])
        assert len(monitor.violations) > 0, (
            "B02: Large timestep energy divergence NOT detected (false negative)"
        )


# ===========================================================================
# B03: Boris correct -> no violation (MUST NOT detect)
# ===========================================================================

class TestB03BorisCorrect:
    def test_no_violation(self):
        """Correct Boris pusher (E=0) conserves energy exactly. No violation."""
        states = b03_boris_correct(num_steps=500, dt=0.01)
        monitor = _run_monitor(states, ["Boris Energy"])
        assert len(monitor.violations) == 0, (
            f"B03: Correct Boris flagged {len(monitor.violations)} violations "
            f"(false positive). First: {monitor.violations[0] if monitor.violations else 'N/A'}"
        )


# ===========================================================================
# B04: Boris dt too large -> energy instability (MUST detect)
# ===========================================================================

class TestB04BorisLargeDt:
    def test_violation_detected(self):
        """Boris with omega_c*dt > 2 is unstable. Must detect energy growth."""
        states = b04_boris_large_dt(num_steps=50, dt=2.5)
        monitor = _run_monitor(states, ["Boris Energy"])
        assert len(monitor.violations) > 0, (
            "B04: Boris large dt energy instability NOT detected (false negative)"
        )


# ===========================================================================
# B05: Boris wrong half-step -> O(dt) energy drift (MUST detect)
# ===========================================================================

class TestB05BorisWrongHalfstep:
    def test_violation_detected(self):
        """Wrong half-step structure causes O(dt) drift. Must detect."""
        states = b05_boris_wrong_halfstep(num_steps=300, dt=0.1)
        monitor = _run_monitor(states, ["Boris Energy"])
        assert len(monitor.violations) > 0, (
            "B05: Boris wrong half-step NOT detected (false negative)"
        )


# ===========================================================================
# B06: Boris wrong E-field sign -> immediate energy jump (MUST detect)
# ===========================================================================

class TestB06BorisWrongSign:
    def test_violation_detected(self):
        """Wrong sign on E-field causes immediate energy change. Must detect."""
        states = b06_boris_wrong_sign(num_steps=50, dt=0.01)
        monitor = _run_monitor(states, ["Boris Energy"])
        assert len(monitor.violations) > 0, (
            "B06: Boris wrong sign NOT detected (false negative)"
        )

    def test_detected_early(self):
        """Violation should be detected within the first 10 timesteps."""
        states = b06_boris_wrong_sign(num_steps=50, dt=0.01)
        monitor = _run_monitor(states, ["Boris Energy"])
        if monitor.violations:
            first_violation_ts = monitor.violations[0].timestep
            assert first_violation_ts < 10, (
                f"B06: Violation detected too late (timestep {first_violation_ts})"
            )


# ===========================================================================
# B09: Lorentz force correct -> no violation (MUST NOT detect)
# ===========================================================================

class TestB09LorentzCorrect:
    def test_no_violation(self):
        """Correct Lorentz force computation. No violation expected."""
        states = b09_lorentz_correct(num_steps=50, dt=0.01)
        monitor = _run_monitor(states, ["Lorentz Force"])
        assert len(monitor.violations) == 0, (
            f"B09: Correct Lorentz flagged {len(monitor.violations)} violations "
            f"(false positive)"
        )


# ===========================================================================
# B10: Lorentz force wrong sign -> force error (MUST detect)
# ===========================================================================

class TestB10LorentzWrongSign:
    def test_violation_detected(self):
        """Wrong sign on v x B in Lorentz force. Must detect."""
        states = b10_lorentz_wrong_sign(num_steps=50, dt=0.01)
        monitor = _run_monitor(states, ["Lorentz Force"])
        assert len(monitor.violations) > 0, (
            "B10: Lorentz force sign error NOT detected (false negative)"
        )


# ===========================================================================
# B11: N-body correct -> no momentum violation (MUST NOT detect)
# ===========================================================================

class TestB11NBodyCorrect:
    def test_no_momentum_violation(self):
        """Symmetric N-body forces conserve momentum. No violation."""
        states = b11_nbody_correct(num_steps=100, dt=0.001)
        monitor = _run_monitor(
            states,
            ["Linear Momentum"],
            thresholds={"Linear Momentum": 1e-4},
        )
        assert len(monitor.violations) == 0, (
            f"B11: Correct N-body flagged {len(monitor.violations)} momentum "
            f"violations (false positive)"
        )


# ===========================================================================
# B12: N-body asymmetric force -> momentum violation (MUST detect)
# ===========================================================================

class TestB12NBodyAsymmetric:
    def test_violation_detected(self):
        """Asymmetric forces (F_ij != -F_ji) violate momentum conservation.

        With a 10% force asymmetry and dt=0.001, the per-step momentum
        change is ~2e-5. Using a tight threshold of 1e-6 (default) ensures
        the step-to-step check catches it.
        """
        states = b12_nbody_asymmetric(num_steps=100, dt=0.001)
        monitor = _run_monitor(
            states,
            ["Linear Momentum"],
            # Use default threshold (1e-6), which is tight enough to catch
            # the ~2e-5 per-step momentum change from asymmetric forces
        )
        assert len(monitor.violations) > 0, (
            "B12: N-body asymmetric force NOT detected (false negative)"
        )


# ===========================================================================
# B13: Reflecting boundary -> no particle loss (MUST NOT detect)
# ===========================================================================

class TestB13BoundaryReflecting:
    def test_no_violation(self):
        """Reflecting boundaries conserve particle count. No violation."""
        states = b13_boundary_reflecting(num_steps=100, dt=0.01)
        monitor = _run_monitor(states, ["Particle Count"])
        assert len(monitor.violations) == 0, (
            f"B13: Reflecting boundary flagged {len(monitor.violations)} "
            f"particle count violations (false positive)"
        )


# ===========================================================================
# B14: Boundary leak -> particle count growth (MUST detect)
# ===========================================================================

class TestB14BoundaryLeak:
    def test_violation_detected(self):
        """Buggy boundary duplicates particles. Must detect count change."""
        states = b14_boundary_leak(num_steps=100, dt=0.01)
        monitor = _run_monitor(states, ["Particle Count"])
        assert len(monitor.violations) > 0, (
            "B14: Boundary particle leak NOT detected (false negative)"
        )
