"""Tests for Monitor integration with ViolationHistory (Phase 3)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sim_debugger.core.config import SimDebuggerConfig
from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState


def _make_state(timestep: int, v_scale: float = 1.0) -> SimulationState:
    """Create a simple state for testing."""
    return SimulationState(
        timestep=timestep,
        time=float(timestep) * 0.01,
        arrays={
            "velocities": np.array([[v_scale, 0.0, 0.0]]),
            "masses": np.array([1.0]),
        },
    )


class TestMonitorWithHistory:
    def test_history_disabled_by_default(self):
        monitor = Monitor(invariants=["Total Energy"])
        assert monitor.violation_history is None

    def test_history_enabled(self):
        monitor = Monitor(
            invariants=["Total Energy"],
            record_history=True,
        )
        assert monitor.violation_history is not None

    def test_history_records_values(self):
        monitor = Monitor(
            invariants=["Total Energy"],
            record_history=True,
        )
        # Run a few steps
        for t in range(5):
            state = _make_state(t)
            monitor.check(state)

        series = monitor.violation_history.get_value_series("Total Energy")
        assert len(series) == 5

    def test_history_records_violations(self):
        monitor = Monitor(
            invariants=["Total Energy"],
            thresholds={"Total Energy": 1e-10},
            record_history=True,
        )
        # First step establishes baseline
        monitor.check(_make_state(0, v_scale=1.0))
        # Second step has different energy -> violation
        monitor.check(_make_state(1, v_scale=2.0))

        assert monitor.violation_history.total_violations > 0

    def test_export_json(self):
        monitor = Monitor(
            invariants=["Total Energy"],
            record_history=True,
        )
        for t in range(10):
            state = _make_state(t)
            monitor.check(state)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            monitor.export_json(path)
            data = json.loads(Path(path).read_text())
            assert "violations" in data
            assert "trends" in data
            assert "value_series" in data
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_without_history_raises(self):
        monitor = Monitor(invariants=["Total Energy"])
        with pytest.raises(RuntimeError, match="history not enabled"):
            monitor.export_json("/tmp/test.json")

    def test_report_includes_trends(self):
        monitor = Monitor(
            invariants=["Total Energy"],
            record_history=True,
        )
        for t in range(20):
            state = _make_state(t)
            monitor.check(state)

        report = monitor.report()
        assert "Invariant trends:" in report
        assert "STABLE" in report

    def test_reset_clears_history(self):
        monitor = Monitor(
            invariants=["Total Energy"],
            record_history=True,
        )
        for t in range(5):
            monitor.check(_make_state(t))

        assert monitor.violation_history.get_value_series("Total Energy")
        monitor.reset()
        assert len(monitor.violation_history.get_value_series("Total Energy")) == 0


class TestMonitorWithConfig:
    def test_config_sets_invariants(self):
        config = SimDebuggerConfig()
        config.monitor.invariants = ["Total Energy"]
        config.thresholds.thresholds = {"Total Energy": 1e-4}

        monitor = Monitor(config=config)
        # Initialise with a state
        monitor.check(_make_state(0))

        assert "Total Energy" in monitor.active_invariants

    def test_config_sets_check_interval(self):
        config = SimDebuggerConfig()
        config.monitor.mode = "lightweight"
        config.performance.lightweight_interval = 5

        monitor = Monitor(
            invariants=["Total Energy"],
            config=config,
        )
        assert monitor._check_interval == 5

    def test_cli_args_override_config(self):
        config = SimDebuggerConfig()
        config.monitor.invariants = ["Total Energy"]

        # CLI args should override
        monitor = Monitor(
            invariants=["Boris Energy"],
            config=config,
        )
        # The CLI arg takes precedence
        assert monitor._requested_invariants == ["Boris Energy"]
