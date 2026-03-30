"""Integration tests: end-to-end pipeline from instrumentation to report.

These tests verify the full pipeline: AST instrumentation -> state capture ->
invariant checking -> temporal localisation -> explanation generation.
"""

import subprocess
import sys
from pathlib import Path

from sim_debugger.core.monitor import Monitor
from sim_debugger.instrument.ast_rewriter import transform_source

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class TestEndToEndMonitor:
    """Test the full Monitor pipeline programmatically."""

    def test_boris_correct_no_violations(self):
        """Correct Boris pusher: full pipeline, no violations."""
        from tests.benchmarks.simulations import b03_boris_correct

        states = b03_boris_correct(num_steps=200)
        monitor = Monitor(invariants=["Boris Energy"])
        for s in states:
            monitor.check(s)

        assert len(monitor.violations) == 0
        report = monitor.report()
        assert "No violations detected" in report

    def test_euler_broken_detected_with_explanation(self):
        """Forward Euler: violation detected with physics explanation."""
        from tests.benchmarks.simulations import b01_harmonic_euler

        states = b01_harmonic_euler(num_steps=100)
        monitor = Monitor(invariants=["Total Energy"])
        for s in states:
            monitor.check(s)

        assert len(monitor.violations) > 0
        first_v = monitor.violations[0]
        assert first_v.explanation is not None
        # Explanation should mention energy
        assert "energy" in first_v.explanation.lower() or "Energy" in first_v.explanation
        # Later violations should have temporal localisation
        if len(monitor.violations) > 5:
            later_v = monitor.violations[5]
            assert later_v.localisation is not None

    def test_lorentz_wrong_sign_detected(self):
        """Lorentz force with wrong sign detected."""
        from tests.benchmarks.simulations import b10_lorentz_wrong_sign

        states = b10_lorentz_wrong_sign(num_steps=50)
        monitor = Monitor(invariants=["Lorentz Force"])
        for s in states:
            monitor.check(s)

        assert len(monitor.violations) > 0


class TestASTInstrumentation:
    """Test that AST instrumentation produces runnable code."""

    def test_simple_simulation_instrumented_runs(self):
        """Instrumented simple simulation runs without errors."""
        source = """
import numpy as np
positions = np.array([[1.0, 0.0, 0.0]])
velocities = np.array([[0.0, 1.0, 0.0]])
masses = np.array([1.0])
dt = 0.01
for t in range(10):
    velocities = velocities - positions * dt
    positions = positions + velocities * dt
"""
        transformed, transformer = transform_source(source, "test_sim.py")
        assert len(transformer.instrumented_loops) == 1

        # Execute the transformed code -- should not raise
        namespace: dict = {"__name__": "__main__"}
        exec(compile(transformed, "test_sim.py", "exec"), namespace)


class TestCLIIntegration:
    """Test CLI commands work end-to-end."""

    def test_list_invariants(self):
        """sim-debugger list-invariants should work."""
        result = subprocess.run(
            [sys.executable, "-m", "sim_debugger.cli.main", "list-invariants"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "Total Energy" in result.stdout
        assert "Boris Energy" in result.stdout
        assert "Lorentz Force" in result.stdout

    def test_check_command(self):
        """sim-debugger check should suggest invariants."""
        euler_path = str(EXAMPLES_DIR / "harmonic_oscillator" / "euler_broken.py")
        result = subprocess.run(
            [sys.executable, "-m", "sim_debugger.cli.main", "check", euler_path],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # Should suggest energy-related invariants
        assert "Total Energy" in result.stdout or "Suggested" in result.stdout

    def test_run_correct_boris(self):
        """sim-debugger run on correct Boris should complete with no violations."""
        boris_path = str(EXAMPLES_DIR / "boris_pusher" / "boris_correct.py")
        result = subprocess.run(
            [sys.executable, "-m", "sim_debugger.cli.main", "run", boris_path,
             "--invariants", "Boris Energy"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "No violations detected" in result.stdout

    def test_run_broken_euler(self):
        """sim-debugger run on broken Euler should detect violations."""
        euler_path = str(EXAMPLES_DIR / "harmonic_oscillator" / "euler_broken.py")
        result = subprocess.run(
            [sys.executable, "-m", "sim_debugger.cli.main", "run", euler_path,
             "--invariants", "Total Energy"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "VIOLATION" in result.stdout
