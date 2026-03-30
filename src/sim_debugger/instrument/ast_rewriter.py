"""AST-based instrumentation for simulation code.

The SimDebugTransformer identifies timestep loops in user simulation code
and injects monitoring hooks before and after each iteration. This enables
zero-modification instrumentation: the user's code runs normally with
monitoring calls woven in at the AST level.

Timestep loop detection heuristics:
1. Variable name matching: loops over 't', 'timestep', 'step', 'iter', 'n'
2. Range pattern: for t in range(num_timesteps)
3. While pattern: while t < T_max with t += dt in body
4. Only the outermost matching loop is instrumented (avoid spatial loops)
"""

from __future__ import annotations

import ast
import textwrap

# Names that suggest a timestep loop variable
_TIMESTEP_NAMES = {
    "t", "timestep", "step", "iter", "n", "i_step",
    "time_step", "tstep", "nstep", "istep", "it",
}

# Names that suggest the loop is a spatial loop (not timestep)
_SPATIAL_NAMES = {
    "i", "j", "k", "ix", "iy", "iz", "cell", "particle",
    "idx", "index", "elem",
}


class SimDebugTransformer(ast.NodeTransformer):
    """AST transformer that instruments timestep loops with monitoring hooks.

    The transformer:
    1. Finds the outermost for/while loop that looks like a timestep loop
    2. Injects a monitoring initialisation call before the loop
    3. Injects state-capture + invariant-check calls at the end of each iteration
    4. Injects a report call after the loop

    The injected code references a `_sim_debugger_monitor` object that is
    created by the import hook or runner before execution.

    Attributes:
        instrumented_loops: List of (line_start, line_end) for instrumented loops.
        source_map: Mapping from injected node line -> original source line.
    """

    def __init__(
        self,
        invariants: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.invariants = invariants
        self.thresholds = thresholds or {}
        self.instrumented_loops: list[tuple[int, int]] = []
        self.source_map: dict[int, int] = {}
        self._depth = 0  # Track nesting to only instrument outermost
        self._instrumented = False

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.AST]:
        """Check if a for-loop is a timestep loop and instrument it."""
        if self._instrumented:
            # Already found and instrumented a timestep loop
            self.generic_visit(node)
            return node

        if self._is_timestep_for_loop(node):
            return self._instrument_for_loop(node)

        # Not a timestep loop; visit children
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        return node

    def visit_While(self, node: ast.While) -> ast.AST | list[ast.AST]:
        """Check if a while-loop is a timestep loop and instrument it."""
        if self._instrumented:
            self.generic_visit(node)
            return node

        if self._is_timestep_while_loop(node):
            return self._instrument_while_loop(node)

        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        return node

    def _is_timestep_for_loop(self, node: ast.For) -> bool:
        """Heuristic: does this for-loop look like a timestep loop?

        Matches patterns like:
            for t in range(num_timesteps):
            for step in range(0, T, dt):
            for n in range(N):
        """
        # Check loop variable name
        target = node.target
        if isinstance(target, ast.Name):
            var_name = target.id.lower()
            if var_name in _SPATIAL_NAMES and var_name not in _TIMESTEP_NAMES:
                return False
            if var_name in _TIMESTEP_NAMES:
                return True
            # Check if iterating over range()
            if isinstance(node.iter, ast.Call):
                func = node.iter.func
                if isinstance(func, ast.Name) and func.id == "range":
                    # range() with a timestep-like variable is likely a timestep loop
                    return True

        return False

    def _is_timestep_while_loop(self, node: ast.While) -> bool:
        """Heuristic: does this while-loop look like a timestep loop?

        Matches patterns like:
            while t < T_max:
            while time < end_time:
        """
        test = node.test
        if isinstance(test, ast.Compare):
            if len(test.comparators) == 1:
                left = test.ops[0]
                if isinstance(left, (ast.Lt, ast.LtE)):
                    # Check if left operand is a timestep variable
                    if isinstance(test.left, ast.Name):
                        var_name = test.left.id.lower()
                        if var_name in _TIMESTEP_NAMES or var_name in {
                            "time", "current_time", "sim_time"
                        }:
                            return True
        return False

    def _instrument_for_loop(self, node: ast.For) -> list[ast.AST]:
        """Instrument a for-loop by injecting monitoring code.

        Returns a list of AST nodes: [init, instrumented_loop, report].
        """
        self._instrumented = True
        loop_start = node.lineno
        loop_end = node.end_lineno or loop_start
        self.instrumented_loops.append((loop_start, loop_end))

        # Get loop variable name for state capture
        loop_var = node.target.id if isinstance(node.target, ast.Name) else "_t"

        # Visit children of the loop body (don't instrument nested loops)
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

        # Build the instrumented code
        init_code = self._make_init_code()
        check_code = self._make_check_code(loop_var)
        report_code = self._make_report_code()

        # Inject check at the end of each loop iteration
        node.body.extend(check_code)

        return [*init_code, node, *report_code]

    def _instrument_while_loop(self, node: ast.While) -> list[ast.AST]:
        """Instrument a while-loop by injecting monitoring code."""
        self._instrumented = True
        loop_start = node.lineno
        loop_end = node.end_lineno or loop_start
        self.instrumented_loops.append((loop_start, loop_end))

        # For while loops, we don't have a clean loop variable.
        # Use a counter.
        loop_var = "_sim_debugger_step"

        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

        init_code = self._make_init_code()
        counter_init = ast.parse(f"{loop_var} = 0").body
        counter_incr = ast.parse(f"{loop_var} += 1").body
        check_code = self._make_check_code(loop_var)
        report_code = self._make_report_code()

        # Inject counter increment and check at end of each iteration
        node.body.extend(counter_incr)
        node.body.extend(check_code)

        return [*init_code, *counter_init, node, *report_code]

    def _make_init_code(self) -> list[ast.stmt]:
        """Generate the monitoring initialisation code."""
        invariants_str = repr(self.invariants) if self.invariants else "None"
        thresholds_str = repr(self.thresholds) if self.thresholds else "{}"

        code = textwrap.dedent(f"""\
            import sim_debugger.core.monitor as _sdm
            import sim_debugger.core.state as _sds
            import sim_debugger.backends.numpy_backend as _sdb
            import numpy as _np
            _sim_debugger_monitor = _sdm.Monitor(
                invariants={invariants_str},
                thresholds={thresholds_str},
            )
        """)
        return ast.parse(code).body

    def _make_check_code(self, loop_var: str) -> list[ast.stmt]:
        """Generate the per-iteration state capture and check code."""
        code = textwrap.dedent(f"""\
            _sim_debugger_state = _sdb.NumPyBackend.capture_state(
                {{**locals()}},
                timestep=int({loop_var}),
                time=float({loop_var}) * locals().get('dt', 1.0),
            )
            _sim_debugger_violations = _sim_debugger_monitor.check(_sim_debugger_state)
            if _sim_debugger_violations:
                for _v in _sim_debugger_violations:
                    print(f"!! VIOLATION: {{_v.invariant_name}} at timestep {{_v.timestep}} "
                          f"({{_v.severity.value}}): {{_v.relative_error:.2%}} !!")
                    if _v.explanation:
                        print(_v.explanation)
                        print()
        """)
        return ast.parse(code).body

    def _make_report_code(self) -> list[ast.stmt]:
        """Generate the final report code."""
        code = textwrap.dedent("""\
            print()
            print(_sim_debugger_monitor.report())
        """)
        return ast.parse(code).body


def transform_source(
    source: str,
    filename: str = "<unknown>",
    invariants: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, SimDebugTransformer]:
    """Parse, transform, and unparse a simulation source file.

    Args:
        source: The source code to instrument.
        filename: The filename (for error reporting).
        invariants: List of invariant names to monitor.
        thresholds: Per-invariant threshold overrides.

    Returns:
        Tuple of (transformed_source, transformer) where the transformer
        contains metadata about what was instrumented.
    """
    tree = ast.parse(source, filename=filename)
    transformer = SimDebugTransformer(
        invariants=invariants,
        thresholds=thresholds,
    )
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_source = ast.unparse(new_tree)
    return new_source, transformer


def instrument_file(
    filepath: str,
    invariants: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, SimDebugTransformer]:
    """Read a Python file and return the instrumented source.

    Args:
        filepath: Path to the Python source file.
        invariants: List of invariant names to monitor.
        thresholds: Per-invariant threshold overrides.

    Returns:
        Tuple of (instrumented_source, transformer).
    """
    with open(filepath) as f:
        source = f.read()
    return transform_source(source, filename=filepath,
                            invariants=invariants, thresholds=thresholds)
