"""Tests for the plugin system (Phase 3)."""

import tempfile
from pathlib import Path

from sim_debugger.core.invariants import create_default_registry
from sim_debugger.core.plugins import (
    discover_plugins,
    load_plugins,
)


def _create_plugin_file(plugin_dir: Path, filename: str, content: str) -> Path:
    """Create a plugin Python file in the given directory."""
    filepath = plugin_dir / filename
    filepath.write_text(content)
    return filepath


VALID_PLUGIN = """\
import math
import numpy as np

class SymplecticVolumeInvariant:
    \"\"\"Custom invariant that checks symplectic volume preservation.\"\"\"

    @property
    def name(self):
        return "Symplectic Volume"

    @property
    def description(self):
        return "Phase-space volume conservation (Liouville theorem)"

    @property
    def default_threshold(self):
        return 1e-10

    def compute(self, state):
        if state.has_array("positions") and state.has_array("velocities"):
            x = state.get_array("positions")
            v = state.get_array("velocities")
            return float(np.sum(x * x) + np.sum(v * v))
        return 0.0

    def check(self, prev_value, curr_value, threshold=None):
        thr = threshold if threshold is not None else self.default_threshold
        if abs(prev_value) > 1e-300:
            rel = abs(curr_value - prev_value) / abs(prev_value)
            if rel > thr:
                return None  # Simplified for testing
        return None

    def applicable(self, state):
        return state.has_array("positions") and state.has_array("velocities")
"""

INVALID_PLUGIN = """\
class NotAnInvariant:
    \"\"\"This class doesn't implement the Invariant protocol.\"\"\"
    def hello(self):
        return "world"
"""

PARTIAL_PLUGIN = """\
class PartialInvariant:
    \"\"\"Has some but not all required methods.\"\"\"

    @property
    def name(self):
        return "Partial"

    @property
    def description(self):
        return "Incomplete invariant"

    # Missing: default_threshold, compute, check, applicable
"""


class TestDiscoverPlugins:
    def test_discovers_valid_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "symplectic.py", VALID_PLUGIN)

            plugins = discover_plugins([plugin_dir])
            assert len(plugins) == 1
            assert plugins[0].name == "Symplectic Volume"
            assert plugins[0].class_name == "SymplecticVolumeInvariant"

    def test_ignores_invalid_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "invalid.py", INVALID_PLUGIN)

            plugins = discover_plugins([plugin_dir])
            assert len(plugins) == 0

    def test_ignores_partial_plugin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "partial.py", PARTIAL_PLUGIN)

            plugins = discover_plugins([plugin_dir])
            assert len(plugins) == 0

    def test_skips_underscore_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "__init__.py", "")
            _create_plugin_file(plugin_dir, "_private.py", VALID_PLUGIN)

            plugins = discover_plugins([plugin_dir])
            assert len(plugins) == 0

    def test_nonexistent_directory(self):
        plugins = discover_plugins(["/nonexistent/path"])
        assert len(plugins) == 0

    def test_enabled_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "symplectic.py", VALID_PLUGIN)

            # Enable a name that doesn't match
            plugins = discover_plugins(
                [plugin_dir], enabled=["Other Invariant"]
            )
            assert len(plugins) == 0

            # Enable the correct name
            plugins = discover_plugins(
                [plugin_dir], enabled=["Symplectic Volume"]
            )
            assert len(plugins) == 1

    def test_disabled_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "symplectic.py", VALID_PLUGIN)

            plugins = discover_plugins(
                [plugin_dir], disabled=["Symplectic Volume"]
            )
            assert len(plugins) == 0


class TestLoadPlugins:
    def test_registers_in_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "symplectic.py", VALID_PLUGIN)

            registry = create_default_registry()
            initial_count = len(registry.list_all())

            loaded = load_plugins(registry, [plugin_dir])
            assert len(loaded) == 1
            assert len(registry.list_all()) == initial_count + 1

            # The invariant should be retrievable by name
            inv = registry.get("Symplectic Volume")
            assert inv.description == "Phase-space volume conservation (Liouville theorem)"

    def test_handles_duplicate_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            _create_plugin_file(plugin_dir, "symplectic.py", VALID_PLUGIN)

            registry = create_default_registry()
            # Load once
            load_plugins(registry, [plugin_dir])
            # Load again -- should warn but not crash
            loaded = load_plugins(registry, [plugin_dir])
            assert len(loaded) == 0  # Duplicate skipped
