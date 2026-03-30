"""Tests for the explanation generator and templates."""


from sim_debugger.core.violations import (
    Violation,
    ViolationPattern,
    ViolationSeverity,
)
from sim_debugger.explain.generator import generate_explanation
from sim_debugger.explain.templates import get_template


class TestTemplateRegistry:
    def test_energy_templates_exist(self):
        """All energy violation patterns have templates."""
        for pattern in ViolationPattern:
            template = get_template("Total Energy", pattern)
            assert template is not None, f"Missing template for Total Energy / {pattern}"

    def test_boris_templates_exist(self):
        """All Boris energy violation patterns have templates."""
        for pattern in ViolationPattern:
            template = get_template("Boris Energy", pattern)
            assert template is not None, f"Missing template for Boris Energy / {pattern}"

    def test_momentum_templates_exist(self):
        for pattern in ViolationPattern:
            template = get_template("Linear Momentum", pattern)
            assert template is not None

    def test_charge_templates_exist(self):
        for pattern in [ViolationPattern.SUDDEN, ViolationPattern.GRADUAL,
                        ViolationPattern.DIVERGENT]:
            template = get_template("Charge Conservation", pattern)
            assert template is not None

    def test_lorentz_template_exists(self):
        template = get_template("Lorentz Force", ViolationPattern.SUDDEN)
        assert template is not None

    def test_gauss_templates_exist(self):
        for pattern in [ViolationPattern.SUDDEN, ViolationPattern.GRADUAL,
                        ViolationPattern.DIVERGENT]:
            template = get_template("Gauss's Law", pattern)
            assert template is not None

    def test_nonexistent_template(self):
        template = get_template("Nonexistent", ViolationPattern.SUDDEN)
        assert template is None


class TestExplanationGenerator:
    def _make_violation(
        self,
        invariant_name: str = "Total Energy",
        expected: float = 1.0,
        actual: float = 1.05,
    ) -> Violation:
        return Violation(
            invariant_name=invariant_name,
            timestep=100,
            time=1.0,
            expected_value=expected,
            actual_value=actual,
            relative_error=(
                abs(actual - expected) / abs(expected) if expected
                else abs(actual - expected)
            ),
            absolute_error=abs(actual - expected),
            severity=ViolationSeverity.ERROR,
        )

    def test_energy_sudden_explanation(self):
        violation = self._make_violation()
        explanation = generate_explanation(
            violation, pattern=ViolationPattern.SUDDEN
        )
        assert "energy" in explanation.lower() or "Energy" in explanation
        assert "100" in explanation  # timestep
        assert "Suggested fix" in explanation

    def test_energy_gradual_explanation(self):
        violation = self._make_violation()
        explanation = generate_explanation(
            violation,
            pattern=ViolationPattern.GRADUAL,
            first_timestep=90,
            duration=10,
        )
        assert "drift" in explanation.lower() or "over" in explanation.lower()
        assert "90" in explanation

    def test_boris_explanation(self):
        violation = self._make_violation(invariant_name="Boris Energy")
        explanation = generate_explanation(
            violation, pattern=ViolationPattern.DIVERGENT
        )
        assert "Boris" in explanation or "boris" in explanation
        assert "omega_c" in explanation or "stability" in explanation.lower()

    def test_momentum_explanation(self):
        violation = self._make_violation(invariant_name="Linear Momentum")
        explanation = generate_explanation(
            violation, pattern=ViolationPattern.SUDDEN
        )
        assert "momentum" in explanation.lower()

    def test_charge_explanation(self):
        violation = self._make_violation(
            invariant_name="Charge Conservation",
            expected=0.0,
            actual=0.001,
        )
        explanation = generate_explanation(
            violation, pattern=ViolationPattern.SUDDEN
        )
        assert "charge" in explanation.lower()

    def test_fallback_for_unknown_combination(self):
        violation = self._make_violation(invariant_name="Custom Invariant")
        explanation = generate_explanation(
            violation, pattern=ViolationPattern.SUDDEN
        )
        # Should still produce some output
        assert "Custom Invariant" in explanation
        assert "violation" in explanation.lower()

    def test_explanation_includes_numeric_data(self):
        violation = self._make_violation(expected=1.0, actual=1.05)
        explanation = generate_explanation(
            violation, pattern=ViolationPattern.SUDDEN
        )
        # Should include the actual numeric values
        assert "5.00%" in explanation or "0.05" in explanation or "1.05" in explanation
