"""Custom import hook for zero-modification instrumentation.

Installs a sys.meta_path finder/loader that intercepts module loading
for the target simulation script and applies AST rewriting before
compilation. This enables `sim-debugger run my_simulation.py` to work
without any modifications to the user's code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types
from typing import Any

from sim_debugger.instrument.ast_rewriter import transform_source


class SimDebugFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that intercepts loading of target simulation modules.

    Only intercepts modules whose source files are in the target paths.
    All other modules are loaded normally.
    """

    def __init__(
        self,
        target_paths: list[str],
        invariants: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.target_paths = [os.path.abspath(p) for p in target_paths]
        self.invariants = invariants
        self.thresholds = thresholds or {}

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Find the module spec, intercepting if it's in our target paths."""
        # Try to find the module normally first
        # Temporarily remove ourselves from meta_path to avoid recursion
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        except (ModuleNotFoundError, ValueError):
            spec = None
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.origin is None:
            return None

        # Check if the module's source is in our target paths
        source_path = os.path.abspath(spec.origin)
        if not any(source_path.startswith(tp) for tp in self.target_paths):
            return None

        # Only instrument .py files
        if not source_path.endswith(".py"):
            return None

        # Create a spec with our custom loader
        loader = SimDebugLoader(
            source_path,
            invariants=self.invariants,
            thresholds=self.thresholds,
        )
        return importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=source_path,
        )


class SimDebugLoader(importlib.abc.Loader):
    """Custom loader that applies AST rewriting before compilation."""

    def __init__(
        self,
        source_path: str,
        invariants: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.source_path = source_path
        self.invariants = invariants
        self.thresholds = thresholds or {}

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return None  # Use default module creation

    def exec_module(self, module: types.ModuleType) -> None:
        """Execute the module with AST instrumentation applied."""
        with open(self.source_path) as f:
            source = f.read()

        transformed, transformer = transform_source(
            source,
            filename=self.source_path,
            invariants=self.invariants,
            thresholds=self.thresholds,
        )

        code = compile(transformed, self.source_path, "exec")
        # SECURITY: exec() is used by design to load AST-transformed
        # simulation modules during import hooking. The transformed code
        # originates from the user's own source files. Only instrument
        # trusted code. See README for details.
        exec(code, module.__dict__)  # noqa: S102


def install_hook(
    target_paths: list[str],
    invariants: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> SimDebugFinder:
    """Install the sim-debugger import hook on sys.meta_path.

    Args:
        target_paths: List of directories/files to instrument.
        invariants: Invariant names to monitor.
        thresholds: Per-invariant thresholds.

    Returns:
        The installed finder (for later removal).
    """
    finder = SimDebugFinder(target_paths, invariants, thresholds)
    sys.meta_path.insert(0, finder)
    return finder


def remove_hook(finder: SimDebugFinder) -> None:
    """Remove a previously installed import hook."""
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass
