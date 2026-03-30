"""Tests for the AST rewriter and instrumentation system."""

import ast
import textwrap

from sim_debugger.instrument.ast_rewriter import (
    transform_source,
)


class TestSimDebugTransformer:
    def test_detects_for_range_loop(self):
        """Detects 'for t in range(N)' as a timestep loop."""
        source = textwrap.dedent("""\
            for t in range(100):
                x = x + v * dt
        """)
        _, transformer = transform_source(source)
        assert len(transformer.instrumented_loops) == 1
        assert transformer.instrumented_loops[0] == (1, 2)

    def test_detects_timestep_variable(self):
        """Detects loops with timestep-like variable names."""
        for var_name in ["t", "timestep", "step", "n"]:
            source = f"for {var_name} in range(100):\n    pass\n"
            _, transformer = transform_source(source)
            assert len(transformer.instrumented_loops) == 1, f"Failed for {var_name}"

    def test_ignores_spatial_loop(self):
        """Does not instrument loops with spatial-like variable names
        that are not also timestep names."""
        source = textwrap.dedent("""\
            for j in [1, 2, 3]:
                pass
        """)
        _, transformer = transform_source(source)
        assert len(transformer.instrumented_loops) == 0

    def test_instruments_only_outermost(self):
        """Only the outermost timestep loop is instrumented."""
        source = textwrap.dedent("""\
            for t in range(100):
                for step in range(10):
                    pass
        """)
        _, transformer = transform_source(source)
        # Should only instrument the outer loop
        assert len(transformer.instrumented_loops) == 1

    def test_while_loop_detection(self):
        """Detects 'while t < T_max:' as a timestep loop."""
        source = textwrap.dedent("""\
            t = 0
            while t < 100:
                x = x + dt
                t += 1
        """)
        _, transformer = transform_source(source)
        assert len(transformer.instrumented_loops) == 1

    def test_transformed_code_is_valid_python(self):
        """Transformed source code is syntactically valid Python."""
        source = textwrap.dedent("""\
            import numpy as np
            x = np.array([1.0])
            v = np.array([0.0])
            m = np.array([1.0])
            dt = 0.01
            for t in range(100):
                v = v - x * dt
                x = x + v * dt
        """)
        transformed, _ = transform_source(source)
        # Should parse without errors
        ast.parse(transformed)

    def test_injected_code_structure(self):
        """Verify the structure of injected monitoring code."""
        source = textwrap.dedent("""\
            for t in range(10):
                x = x + 1
        """)
        transformed, _ = transform_source(source)

        # Should contain monitor initialisation
        assert "_sim_debugger_monitor" in transformed
        # Should contain state capture
        assert "capture_state" in transformed
        # Should contain report
        assert "report" in transformed

    def test_preserves_original_code(self):
        """Original simulation logic is preserved in the transformed code."""
        source = textwrap.dedent("""\
            for t in range(10):
                x = x + v * dt
                v = v - k * x * dt
        """)
        transformed, _ = transform_source(source)
        assert "x = x + v * dt" in transformed
        assert "v = v - k * x * dt" in transformed

    def test_no_instrumentation_without_loop(self):
        """Code without loops is not instrumented."""
        source = "x = 1\ny = 2\nz = x + y\n"
        _, transformer = transform_source(source)
        assert len(transformer.instrumented_loops) == 0

    def test_custom_invariants_in_transform(self):
        """Custom invariant list is passed through to the monitor."""
        source = "for t in range(10):\n    pass\n"
        transformed, _ = transform_source(
            source, invariants=["Total Energy"]
        )
        assert "'Total Energy'" in transformed

    def test_custom_thresholds_in_transform(self):
        """Custom thresholds are passed through to the monitor."""
        source = "for t in range(10):\n    pass\n"
        transformed, _ = transform_source(
            source, thresholds={"Total Energy": 1e-4}
        )
        assert "1e-04" in transformed or "0.0001" in transformed
