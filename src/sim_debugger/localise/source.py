"""Source-code localisation: identify which code block caused a violation.

Given a conservation law violation, narrows down the responsible code by:

1. AST source mapping: maps injected monitoring nodes back to original
   source line numbers so that violations reference the user's code.
2. Sub-step instrumentation: for known integrator patterns (Boris pusher),
   instruments each sub-step individually so violations can be attributed
   to a specific operation (half E-push, B-rotation, etc.).
3. Bisection approach: when sub-step data is insufficient, offers to re-run
   with finer instrumentation that splits the timestep body in half and
   tests each half, recursively narrowing to the offending code block.

Usage::

    from sim_debugger.localise.source import (
        build_source_map,
        localise_source,
        identify_boris_substeps,
    )

    source_map = build_source_map(transformer)
    loc = localise_source(violation, source_map, source_code)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from sim_debugger.core.violations import SourceLocalisation

# ---------------------------------------------------------------------------
# Source map: maps injected AST node lines to original source lines
# ---------------------------------------------------------------------------

@dataclass
class SourceMap:
    """Mapping from instrumented code locations to original source locations.

    The AST rewriter injects monitoring calls that do not exist in the
    original source. This map records the relationship so that violation
    reports can reference the user's code, not the injected code.

    Attributes:
        original_file: Path to the original source file.
        loop_ranges: List of (start_line, end_line) for instrumented loops.
        function_ranges: Mapping from function_name -> (start_line, end_line).
        substep_ranges: Mapping from substep_label -> (start_line, end_line).
            Used for Boris pusher sub-step attribution.
        line_mapping: Mapping from instrumented_line -> original_line.
    """

    original_file: str = ""
    loop_ranges: list[tuple[int, int]] = field(default_factory=list)
    function_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    substep_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    line_mapping: dict[int, int] = field(default_factory=dict)


def build_source_map(
    source: str,
    filename: str = "<unknown>",
) -> SourceMap:
    """Build a source map from the original source code.

    Parses the source to extract:
    - Function definitions and their line ranges
    - Timestep loop locations
    - Boris pusher sub-step patterns

    Args:
        source: The original (uninstrumented) source code.
        filename: Path to the source file.

    Returns:
        A SourceMap containing extracted location information.
    """
    source_map = SourceMap(original_file=filename)

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return source_map

    for node in ast.walk(tree):
        # Record function definitions
        if isinstance(node, ast.FunctionDef):
            end_line = node.end_lineno or node.lineno
            source_map.function_ranges[node.name] = (node.lineno, end_line)

        # Record class methods
        if isinstance(node, ast.AsyncFunctionDef):
            end_line = node.end_lineno or node.lineno
            source_map.function_ranges[node.name] = (node.lineno, end_line)

    # Detect Boris pusher sub-steps by pattern matching
    substeps = identify_boris_substeps(source)
    source_map.substep_ranges = substeps

    return source_map


# ---------------------------------------------------------------------------
# Boris pusher sub-step identification
# ---------------------------------------------------------------------------

# Patterns for identifying Boris pusher sub-steps in source code
_BORIS_HALF_E_PUSH_PATTERNS = [
    # v_minus = v + (q*dt)/(2*m) * E
    re.compile(
        r"v\w*\s*[\+\-]?=\s*.*(?:q|charge)\s*\*\s*(?:dt|delta_t).*(?:E|E_field)",
        re.IGNORECASE,
    ),
    # Half-push assignment patterns
    re.compile(
        r"v_?(?:minus|half|m)\s*=\s*v\w*\s*\+",
        re.IGNORECASE,
    ),
    re.compile(
        r"v_?(?:plus|p)\s*=\s*v_?(?:plus|rot)\w*\s*\+",
        re.IGNORECASE,
    ),
]

_BORIS_ROTATION_PATTERNS = [
    # Cross product patterns indicating B-field rotation
    re.compile(r"(?:np\.)?cross\s*\(", re.IGNORECASE),
    # Rotation-specific variable names
    re.compile(r"(?:v_?rot|v_?plus|t_vec|s_vec)\s*=", re.IGNORECASE),
    # tan(theta/2) pattern from Boris rotation formula
    re.compile(r"tan\s*\(", re.IGNORECASE),
]


def identify_boris_substeps(
    source: str,
) -> dict[str, tuple[int, int]]:
    """Identify Boris pusher sub-steps in source code by pattern matching.

    Scans the source for patterns characteristic of the three Boris steps:
    1. Half E-field push (first)
    2. B-field rotation
    3. Half E-field push (second)

    Args:
        source: The source code to analyse.

    Returns:
        Dict mapping sub-step label to (start_line, end_line).
        Labels: "half_e_push_1", "b_rotation", "half_e_push_2".
    """
    substeps: dict[str, tuple[int, int]] = {}
    lines = source.split("\n")

    e_push_lines: list[int] = []
    rotation_lines: list[int] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        for pattern in _BORIS_HALF_E_PUSH_PATTERNS:
            if pattern.search(stripped):
                e_push_lines.append(i)
                break

        for pattern in _BORIS_ROTATION_PATTERNS:
            if pattern.search(stripped):
                rotation_lines.append(i)
                break

    # Assign sub-steps based on ordering
    if e_push_lines and rotation_lines:
        # First E-push: lines before the first rotation
        first_rot = min(rotation_lines)
        pre_rot_pushes = [ln for ln in e_push_lines if ln < first_rot]
        post_rot_pushes = [ln for ln in e_push_lines if ln > first_rot]

        if pre_rot_pushes:
            substeps["half_e_push_1"] = (min(pre_rot_pushes), max(pre_rot_pushes))

        if rotation_lines:
            substeps["b_rotation"] = (min(rotation_lines), max(rotation_lines))

        if post_rot_pushes:
            substeps["half_e_push_2"] = (min(post_rot_pushes), max(post_rot_pushes))

    return substeps


# ---------------------------------------------------------------------------
# Source-code localisation
# ---------------------------------------------------------------------------

def localise_source(
    invariant_name: str,
    source_map: SourceMap,
    violation_timestep: int,  # noqa: F841
    substep_values: dict[str, float] | None = None,
) -> SourceLocalisation | None:
    """Localise a violation to a specific source code region.

    Uses the source map to determine which function or code block is
    responsible for the violation. For Boris pusher violations, uses
    sub-step invariant values to attribute the violation to a specific
    sub-step.

    Args:
        invariant_name: Name of the violated invariant.
        source_map: The source map built from the original code.
        violation_timestep: Timestep at which the violation occurred.
            TODO: Use this to correlate with temporal localisation data
            for finer-grained source attribution within the timestep loop.
        substep_values: Optional dict of substep_label -> invariant_value
            after that substep executed. Used to narrow Boris violations.

    Returns:
        A SourceLocalisation, or None if localisation is not possible.
    """
    # TODO: use violation_timestep to correlate with temporal data
    _ = violation_timestep
    # For Boris-specific invariants, try sub-step attribution
    if invariant_name in ("Boris Energy", "Total Energy") and source_map.substep_ranges:
        return _localise_boris_substep(source_map, substep_values)

    # For other invariants, try to find the enclosing function
    if source_map.loop_ranges:
        loop_start, loop_end = source_map.loop_ranges[0]
        # Find the function containing the loop
        enclosing_fn = _find_enclosing_function(
            loop_start, source_map.function_ranges,
        )
        fn_name = enclosing_fn or "<module>"

        return SourceLocalisation(
            file=source_map.original_file,
            line_start=loop_start,
            line_end=loop_end,
            function_name=fn_name,
        )

    # Fall back to the first function range available
    if source_map.function_ranges:
        first_fn = next(iter(source_map.function_ranges))
        start, end = source_map.function_ranges[first_fn]
        return SourceLocalisation(
            file=source_map.original_file,
            line_start=start,
            line_end=end,
            function_name=first_fn,
        )

    return None


def _localise_boris_substep(
    source_map: SourceMap,
    substep_values: dict[str, float] | None,
) -> SourceLocalisation | None:
    """Localise a Boris pusher violation to a specific sub-step.

    The Boris pusher has three sub-steps:
    1. Half E-push: v_minus = v + (q*dt)/(2*m) * E
    2. B-rotation: v_plus = rotate(v_minus, B)
    3. Half E-push: v_{n+1} = v_plus + (q*dt)/(2*m) * E

    If sub-step invariant values are available, determine which sub-step
    introduced the violation. Otherwise, report the full Boris range.

    Args:
        source_map: Source map with substep_ranges populated.
        substep_values: Dict of substep_label -> kinetic energy after step.
            Expected keys: "half_e_push_1", "b_rotation", "half_e_push_2".

    Returns:
        SourceLocalisation pointing to the offending sub-step.
    """
    substeps = source_map.substep_ranges
    if not substeps:
        return None

    # If we have sub-step values, find which step caused the deviation
    if substep_values and len(substep_values) >= 2:
        # Check B-rotation: should preserve |v|^2 exactly
        ke_before_rot = substep_values.get("half_e_push_1")
        ke_after_rot = substep_values.get("b_rotation")

        if ke_before_rot is not None and ke_after_rot is not None:
            if abs(ke_before_rot) > 1e-300:
                rot_error = abs(ke_after_rot - ke_before_rot) / abs(ke_before_rot)
                if rot_error > 1e-10:
                    # Violation in the B-rotation step
                    if "b_rotation" in substeps:
                        s, e = substeps["b_rotation"]
                        return SourceLocalisation(
                            file=source_map.original_file,
                            line_start=s,
                            line_end=e,
                            function_name="boris_pusher",
                            sub_step="b_rotation",
                        )

        # Check second half-push
        ke_after_rot_val = substep_values.get("b_rotation")
        ke_final = substep_values.get("half_e_push_2")
        if ke_after_rot_val is not None and ke_final is not None:
            if abs(ke_after_rot_val) > 1e-300:
                push2_error = abs(ke_final - ke_after_rot_val) / abs(ke_after_rot_val)
                if push2_error > 1e-6:
                    if "half_e_push_2" in substeps:
                        s, e = substeps["half_e_push_2"]
                        return SourceLocalisation(
                            file=source_map.original_file,
                            line_start=s,
                            line_end=e,
                            function_name="boris_pusher",
                            sub_step="half_e_push_2",
                        )

    # Fall back to full Boris range
    all_lines: list[int] = []
    for label, (s, e) in substeps.items():
        all_lines.extend(range(s, e + 1))

    if all_lines:
        return SourceLocalisation(
            file=source_map.original_file,
            line_start=min(all_lines),
            line_end=max(all_lines),
            function_name="boris_pusher",
            sub_step=None,
        )

    return None


def _find_enclosing_function(
    line: int,
    function_ranges: dict[str, tuple[int, int]],
) -> str | None:
    """Find the function that contains the given line number.

    Returns the name of the smallest enclosing function, or None if
    the line is at module level.
    """
    best_match: str | None = None
    best_span = float("inf")

    for fn_name, (start, end) in function_ranges.items():
        if start <= line <= end:
            span = end - start
            if span < best_span:
                best_span = span
                best_match = fn_name

    return best_match


# ---------------------------------------------------------------------------
# Bisection-based source localisation
# ---------------------------------------------------------------------------

@dataclass
class BisectionResult:
    """Result of bisection-based source localisation.

    When sub-step data is insufficient, we can re-run the simulation with
    progressively finer instrumentation to narrow down the violating code.

    Attributes:
        narrowed_range: (start_line, end_line) of the narrowed code region.
        bisection_depth: Number of bisection iterations performed.
        identified_statements: List of AST statement types in the range.
    """

    narrowed_range: tuple[int, int]
    bisection_depth: int
    identified_statements: list[str] = field(default_factory=list)


def plan_bisection(
    source: str,
    loop_start: int,
    loop_end: int,
) -> list[tuple[int, int]]:
    """Plan a bisection of the timestep loop body.

    Splits the loop body into two halves at the statement boundary
    closest to the midpoint. Returns the two halves as (start, end) ranges.

    This is used to plan finer instrumentation: insert a monitoring
    check between the two halves and re-run to determine which half
    contains the violation.

    Args:
        source: The original source code.
        loop_start: Start line of the loop body.
        loop_end: End line of the loop body.

    Returns:
        List of (start_line, end_line) pairs for each half.
        Returns the full range if the body cannot be split.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [(loop_start, loop_end)]

    # Find the loop node
    loop_body_stmts: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            if hasattr(node, "lineno") and node.lineno == loop_start:
                loop_body_stmts = node.body
                break

    if len(loop_body_stmts) < 2:
        return [(loop_start, loop_end)]

    # Split at the midpoint
    mid = len(loop_body_stmts) // 2
    first_half = loop_body_stmts[:mid]
    second_half = loop_body_stmts[mid:]

    first_start = first_half[0].lineno
    first_end = first_half[-1].end_lineno or first_half[-1].lineno
    second_start = second_half[0].lineno
    second_end = second_half[-1].end_lineno or second_half[-1].lineno

    return [(first_start, first_end), (second_start, second_end)]
