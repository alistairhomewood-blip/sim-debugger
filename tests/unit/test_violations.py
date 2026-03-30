"""Tests for violation data models and severity classification."""


import pytest

from sim_debugger.core.violations import (
    TemporalLocalisation,
    Violation,
    ViolationPattern,
    ViolationSeverity,
    classify_severity,
)


class TestViolationSeverity:
    def test_classify_warning(self):
        """Error in [threshold, 10*threshold) -> WARNING."""
        assert classify_severity(1.5e-6, 1e-6) == ViolationSeverity.WARNING
        assert classify_severity(9e-6, 1e-6) == ViolationSeverity.WARNING

    def test_classify_error(self):
        """Error in [10*threshold, 100*threshold) -> ERROR."""
        assert classify_severity(1e-5, 1e-6) == ViolationSeverity.ERROR
        assert classify_severity(9e-5, 1e-6) == ViolationSeverity.ERROR

    def test_classify_critical(self):
        """Error >= 100*threshold -> CRITICAL."""
        assert classify_severity(1e-4, 1e-6) == ViolationSeverity.CRITICAL
        assert classify_severity(1.0, 1e-6) == ViolationSeverity.CRITICAL

    def test_classify_nan(self):
        """NaN or Inf -> CRITICAL."""
        assert classify_severity(float("nan"), 1e-6) == ViolationSeverity.CRITICAL
        assert classify_severity(float("inf"), 1e-6) == ViolationSeverity.CRITICAL

    def test_classify_nan_value(self):
        """NaN in value -> CRITICAL regardless of error."""
        assert classify_severity(1e-8, 1e-6, float("nan")) == ViolationSeverity.CRITICAL

    def test_classify_inf_value(self):
        """Inf in value -> CRITICAL regardless of error."""
        assert classify_severity(1e-8, 1e-6, float("inf")) == ViolationSeverity.CRITICAL


class TestViolation:
    def test_signed_relative_error_positive(self):
        v = Violation(
            invariant_name="test",
            timestep=10,
            time=1.0,
            expected_value=100.0,
            actual_value=103.0,
            relative_error=0.03,
            absolute_error=3.0,
            severity=ViolationSeverity.WARNING,
        )
        assert v.signed_relative_error == pytest.approx(0.03)

    def test_signed_relative_error_negative(self):
        v = Violation(
            invariant_name="test",
            timestep=10,
            time=1.0,
            expected_value=100.0,
            actual_value=97.0,
            relative_error=0.03,
            absolute_error=3.0,
            severity=ViolationSeverity.WARNING,
        )
        assert v.signed_relative_error == pytest.approx(-0.03)

    def test_signed_relative_error_near_zero(self):
        v = Violation(
            invariant_name="test",
            timestep=10,
            time=1.0,
            expected_value=0.0,
            actual_value=1e-10,
            relative_error=1e-10,
            absolute_error=1e-10,
            severity=ViolationSeverity.WARNING,
        )
        # When expected is zero, signed relative error is just the difference
        assert v.signed_relative_error == pytest.approx(1e-10)


class TestTemporalLocalisation:
    def test_duration_calculation(self):
        tl = TemporalLocalisation(
            first_violation_timestep=100,
            pattern=ViolationPattern.GRADUAL,
            violation_trajectory=[(100, 1.0), (105, 1.1), (110, 1.2)],
        )
        assert tl.duration == 10

    def test_duration_empty_trajectory(self):
        tl = TemporalLocalisation(
            first_violation_timestep=100,
            pattern=ViolationPattern.SUDDEN,
            violation_trajectory=[],
        )
        assert tl.duration == 0


class TestViolationPattern:
    def test_pattern_values(self):
        assert ViolationPattern.SUDDEN.value == "sudden"
        assert ViolationPattern.GRADUAL.value == "gradual"
        assert ViolationPattern.OSCILLATORY.value == "oscillatory"
        assert ViolationPattern.DIVERGENT.value == "divergent"
