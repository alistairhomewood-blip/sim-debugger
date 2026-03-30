"""Tests for violation history and trend analysis (Phase 3)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sim_debugger.core.history import ViolationHistory
from sim_debugger.core.violations import Violation, ViolationSeverity


def _make_violation(
    name: str = "Total Energy",
    timestep: int = 100,
    severity: ViolationSeverity = ViolationSeverity.WARNING,
    relative_error: float = 0.01,
) -> Violation:
    return Violation(
        invariant_name=name,
        timestep=timestep,
        time=float(timestep),
        expected_value=1.0,
        actual_value=1.0 + relative_error,
        relative_error=relative_error,
        absolute_error=relative_error,
        severity=severity,
    )


class TestViolationHistory:
    def test_record_and_query_values(self):
        history = ViolationHistory()
        history.record_values(1, {"Total Energy": 1.0})
        history.record_values(2, {"Total Energy": 1.001})
        history.record_values(3, {"Total Energy": 1.002})

        series = history.get_value_series("Total Energy")
        assert len(series) == 3
        assert series[0] == (1, 1.0)
        assert series[2] == (3, 1.002)

    def test_record_violations(self):
        history = ViolationHistory()
        v1 = _make_violation(timestep=100)
        v2 = _make_violation(timestep=200, severity=ViolationSeverity.ERROR)
        history.record_violation(v1)
        history.record_violation(v2)

        assert history.total_violations == 2
        violations = history.get_violations()
        assert len(violations) == 2

    def test_query_by_invariant_name(self):
        history = ViolationHistory()
        history.record_violation(_make_violation(name="Total Energy"))
        history.record_violation(_make_violation(name="Boris Energy"))
        history.record_violation(_make_violation(name="Total Energy"))

        energy = history.get_violations(invariant_name="Total Energy")
        assert len(energy) == 2
        boris = history.get_violations(invariant_name="Boris Energy")
        assert len(boris) == 1

    def test_query_by_severity(self):
        history = ViolationHistory()
        history.record_violation(
            _make_violation(severity=ViolationSeverity.WARNING)
        )
        history.record_violation(
            _make_violation(severity=ViolationSeverity.ERROR)
        )
        history.record_violation(
            _make_violation(severity=ViolationSeverity.CRITICAL)
        )

        errors_and_above = history.get_violations(
            severity=ViolationSeverity.ERROR
        )
        assert len(errors_and_above) == 2  # ERROR + CRITICAL

    def test_query_by_timestep_range(self):
        history = ViolationHistory()
        history.record_violation(_make_violation(timestep=50))
        history.record_violation(_make_violation(timestep=100))
        history.record_violation(_make_violation(timestep=200))

        results = history.get_violations(timestep_min=80, timestep_max=150)
        assert len(results) == 1
        assert results[0].timestep == 100

    def test_query_last_n(self):
        history = ViolationHistory()
        for i in range(10):
            history.record_violation(_make_violation(timestep=i * 10))

        results = history.get_violations(last_n=3)
        assert len(results) == 3
        assert results[0].timestep == 70

    def test_downsample_interval(self):
        history = ViolationHistory(downsample_interval=10)
        for i in range(100):
            history.record_values(i, {"Total Energy": 1.0 + i * 0.001})

        series = history.get_value_series("Total Energy")
        assert len(series) == 10  # 100 / 10

    def test_invariant_names(self):
        history = ViolationHistory()
        history.record_values(1, {"Total Energy": 1.0, "Boris Energy": 0.5})
        assert set(history.invariant_names) == {"Boris Energy", "Total Energy"}

    def test_record_violations_bulk(self):
        history = ViolationHistory()
        violations = [_make_violation(timestep=i) for i in range(5)]
        history.record_violations(violations)
        assert history.total_violations == 5


class TestTrendAnalysis:
    def test_stable_invariant(self):
        history = ViolationHistory()
        for i in range(100):
            history.record_values(i, {"Total Energy": 1.0})

        trend = history.compute_trend("Total Energy")
        assert trend.is_stable
        assert trend.num_samples == 100
        assert abs(trend.drift_rate) < 1e-10
        assert abs(trend.std) < 1e-10
        assert trend.mean == pytest.approx(1.0)

    def test_drifting_invariant(self):
        history = ViolationHistory()
        for i in range(100):
            history.record_values(i, {"Total Energy": 1.0 + i * 0.01})

        trend = history.compute_trend("Total Energy", stability_threshold=0.01)
        assert not trend.is_stable
        assert trend.drift_rate > 0
        assert trend.drift_total > 0
        assert trend.relative_drift > 0.01

    def test_oscillating_invariant(self):
        history = ViolationHistory()
        for i in range(100):
            value = 1.0 + 0.1 * np.sin(i * 0.5)
            history.record_values(i, {"Total Energy": value})

        trend = history.compute_trend("Total Energy")
        assert trend.oscillation_amplitude > 0.1

    def test_empty_trend(self):
        history = ViolationHistory()
        trend = history.compute_trend("Nonexistent")
        assert trend.num_samples == 0
        assert trend.is_stable  # Default

    def test_compute_all_trends(self):
        history = ViolationHistory()
        for i in range(50):
            history.record_values(i, {
                "Total Energy": 1.0,
                "Boris Energy": 0.5 + i * 0.001,
            })

        trends = history.compute_all_trends()
        assert "Total Energy" in trends
        assert "Boris Energy" in trends
        assert trends["Total Energy"].is_stable


class TestHistoryExport:
    def test_export_json(self):
        history = ViolationHistory()
        for i in range(10):
            history.record_values(i, {"Total Energy": 1.0 + i * 0.001})
        history.record_violation(_make_violation(timestep=5))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            history.export_json(path)
            data = json.loads(Path(path).read_text())

            assert data["version"] == "1.0"
            assert data["total_violations"] == 1
            assert len(data["violations"]) == 1
            assert "Total Energy" in data["trends"]
            assert "Total Energy" in data["value_series"]
            assert len(data["value_series"]["Total Energy"]) == 10
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_json_structure(self):
        history = ViolationHistory()
        history.record_violation(_make_violation(
            name="Boris Energy",
            timestep=42,
            severity=ViolationSeverity.ERROR,
        ))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            history.export_json(path)
            data = json.loads(Path(path).read_text())

            v = data["violations"][0]
            assert v["invariant_name"] == "Boris Energy"
            assert v["timestep"] == 42
            assert v["severity"] == "error"
        finally:
            Path(path).unlink(missing_ok=True)


class TestHistorySummary:
    def test_summary_with_violations(self):
        history = ViolationHistory()
        for i in range(50):
            history.record_values(i, {"Total Energy": 1.0 + i * 0.001})
        history.record_violation(
            _make_violation(severity=ViolationSeverity.WARNING)
        )
        history.record_violation(
            _make_violation(severity=ViolationSeverity.ERROR)
        )

        summary = history.summary()
        assert "Total violations: 2" in summary
        assert "Total Energy" in summary

    def test_summary_empty(self):
        history = ViolationHistory()
        summary = history.summary()
        assert "Total violations: 0" in summary
