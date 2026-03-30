"""Tests for source-code localisation (Phase 3).

Tests the AST source mapping, Boris sub-step identification,
source-code localisation, and bisection planning.
"""

import textwrap

from sim_debugger.localise.source import (
    SourceMap,
    build_source_map,
    identify_boris_substeps,
    localise_source,
    plan_bisection,
)

# ===========================================================================
# Source map construction
# ===========================================================================

class TestBuildSourceMap:
    def test_extracts_function_ranges(self):
        source = textwrap.dedent("""\
            def step(x, v, dt):
                v += -x * dt
                x += v * dt
                return x, v

            def helper():
                pass
        """)
        sm = build_source_map(source, "test.py")
        assert "step" in sm.function_ranges
        assert "helper" in sm.function_ranges
        assert sm.function_ranges["step"] == (1, 4)
        assert sm.function_ranges["helper"] == (6, 7)
        assert sm.original_file == "test.py"

    def test_handles_syntax_error(self):
        source = "def broken(\n"
        sm = build_source_map(source, "bad.py")
        assert sm.function_ranges == {}
        assert sm.original_file == "bad.py"

    def test_empty_source(self):
        sm = build_source_map("", "empty.py")
        assert sm.function_ranges == {}


# ===========================================================================
# Boris sub-step identification
# ===========================================================================

class TestBorisSubstepIdentification:
    def test_detects_half_push_and_rotation(self):
        source = textwrap.dedent("""\
            # Half E-push
            v_minus = v + q * dt / (2 * m) * E_field
            # B-rotation
            t_vec = q * B * dt / (2 * m)
            v_rot = v_minus + np.cross(v_minus, t_vec)
            # Second half E-push
            v_plus = v_rot + q * dt / (2 * m) * E_field
        """)
        substeps = identify_boris_substeps(source)
        assert "half_e_push_1" in substeps
        assert "b_rotation" in substeps
        assert "half_e_push_2" in substeps

    def test_rotation_detected_by_cross_product(self):
        source = textwrap.dedent("""\
            v_minus = v + qdt_2m * E
            v_prime = v_minus + np.cross(v_minus, t_vec)
            s_vec = 2 * t_vec / (1 + t_vec**2)
            v_plus = v_minus + np.cross(v_prime, s_vec)
            v_new = v_plus + qdt_2m * E
        """)
        substeps = identify_boris_substeps(source)
        assert "b_rotation" in substeps

    def test_no_boris_in_simple_code(self):
        source = textwrap.dedent("""\
            x = x + v * dt
            v = v - x * dt
        """)
        substeps = identify_boris_substeps(source)
        # Should not detect Boris sub-steps in non-Boris code
        assert "b_rotation" not in substeps


# ===========================================================================
# Source localisation
# ===========================================================================

class TestSourceLocalisation:
    def test_localise_to_function(self):
        sm = SourceMap(
            original_file="sim.py",
            loop_ranges=[(10, 30)],
            function_ranges={
                "main": (1, 50),
                "timestep": (8, 35),
            },
        )
        result = localise_source("Total Energy", sm, violation_timestep=100)
        assert result is not None
        assert result.file == "sim.py"
        assert result.function_name == "timestep"  # Smallest enclosing

    def test_localise_boris_with_substep_values(self):
        sm = SourceMap(
            original_file="boris.py",
            substep_ranges={
                "half_e_push_1": (5, 6),
                "b_rotation": (8, 12),
                "half_e_push_2": (14, 15),
            },
        )
        # Rotation step introduces error
        substep_values = {
            "half_e_push_1": 1.0,
            "b_rotation": 1.5,  # Should be 1.0 -- violation!
            "half_e_push_2": 1.5,
        }
        result = localise_source(
            "Boris Energy", sm, violation_timestep=100,
            substep_values=substep_values,
        )
        assert result is not None
        assert result.sub_step == "b_rotation"
        assert result.line_start == 8

    def test_localise_boris_without_substep_values(self):
        sm = SourceMap(
            original_file="boris.py",
            substep_ranges={
                "half_e_push_1": (5, 6),
                "b_rotation": (8, 12),
                "half_e_push_2": (14, 15),
            },
        )
        result = localise_source("Boris Energy", sm, violation_timestep=100)
        assert result is not None
        assert result.sub_step is None  # Full range, no narrowing
        assert result.line_start == 5
        assert result.line_end == 15

    def test_localise_returns_none_for_empty_map(self):
        sm = SourceMap(original_file="empty.py")
        result = localise_source("Total Energy", sm, violation_timestep=100)
        assert result is None


# ===========================================================================
# Bisection planning
# ===========================================================================

class TestBisectionPlanning:
    def test_splits_loop_body(self):
        source = textwrap.dedent("""\
            for t in range(100):
                a = compute_a()
                b = compute_b()
                c = compute_c()
                d = compute_d()
        """)
        halves = plan_bisection(source, loop_start=1, loop_end=5)
        assert len(halves) == 2
        # Each half should be a (start, end) tuple
        assert halves[0][0] < halves[1][0]

    def test_single_statement_cannot_split(self):
        source = textwrap.dedent("""\
            for t in range(100):
                compute()
        """)
        halves = plan_bisection(source, loop_start=1, loop_end=2)
        assert len(halves) == 1  # Cannot split single statement

    def test_syntax_error_returns_full_range(self):
        source = "invalid python{{{"
        halves = plan_bisection(source, loop_start=1, loop_end=5)
        assert len(halves) == 1
        assert halves[0] == (1, 5)
