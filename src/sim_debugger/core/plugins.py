"""Plugin system for custom invariant monitors.

Allows users to extend sim-debugger with their own invariant monitors
by placing Python modules in plugin directories. Plugins are discovered
and loaded at runtime, and their invariants are registered in the global
invariant registry.

Plugin structure::

    my_plugin/
        __init__.py        # Optional
        my_invariant.py    # Must contain classes implementing Invariant protocol

Each Python file in a plugin directory is scanned for classes that
implement the Invariant protocol (have name, description, default_threshold
properties, and compute/check/applicable methods).

Configuration in `.sim-debugger.toml`::

    [plugins]
    paths = ["./my_invariants", "~/shared_invariants"]
    enabled = ["MyCustomInvariant"]
    disabled = ["ExperimentalInvariant"]

Usage::

    from sim_debugger.core.plugins import discover_plugins, load_plugins

    # Discover available plugins
    plugins = discover_plugins(["./my_invariants"])

    # Load and register
    registry = create_default_registry()
    load_plugins(registry, ["./my_invariants"])
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim_debugger.core.invariants import InvariantRegistry

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a discovered plugin invariant.

    Attributes:
        name: The invariant name (from its .name property).
        description: The invariant description.
        module_path: Path to the Python module containing the invariant.
        class_name: Name of the class implementing the invariant.
        instance: The instantiated invariant object.
    """

    name: str
    description: str
    module_path: str
    class_name: str
    instance: Any


def discover_plugins(
    plugin_paths: Sequence[str | Path],
    enabled: list[str] | None = None,
    disabled: list[str] | None = None,
) -> list[PluginInfo]:
    """Discover invariant plugins in the specified directories.

    Scans each directory for Python files, imports them, and looks for
    classes that implement the Invariant protocol.

    Args:
        plugin_paths: Directories to search for plugin modules.
        enabled: If set, only load invariants with these names.
        disabled: Names of invariants to skip.

    Returns:
        List of PluginInfo for each discovered invariant.
    """
    plugins: list[PluginInfo] = []
    disabled_set = set(disabled or [])

    for plugin_dir in plugin_paths:
        path = Path(plugin_dir).expanduser().resolve()
        if not path.is_dir():
            logger.warning("Plugin directory not found: %s", path)
            continue

        # Scan for Python files
        for py_file in sorted(path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            try:
                discovered = _load_module_invariants(py_file)
            except Exception as e:
                logger.warning(
                    "Error loading plugin %s: %s", py_file, e
                )
                continue

            for info in discovered:
                # Apply enabled/disabled filters
                if info.name in disabled_set:
                    logger.debug("Plugin '%s' is disabled, skipping", info.name)
                    continue
                if enabled is not None and info.name not in enabled:
                    logger.debug(
                        "Plugin '%s' not in enabled list, skipping", info.name
                    )
                    continue
                plugins.append(info)

    return plugins


def load_plugins(
    registry: InvariantRegistry,
    plugin_paths: Sequence[str | Path],
    enabled: list[str] | None = None,
    disabled: list[str] | None = None,
) -> list[PluginInfo]:
    """Discover and register plugin invariants.

    Discovers plugins in the given directories and registers them in
    the invariant registry. Skips plugins whose names conflict with
    already-registered invariants (warns but does not error).

    Args:
        registry: The invariant registry to add plugins to.
        plugin_paths: Directories to search.
        enabled: If set, only load these named invariants.
        disabled: Names to skip.

    Returns:
        List of successfully loaded PluginInfo.
    """
    discovered = discover_plugins(plugin_paths, enabled, disabled)
    loaded: list[PluginInfo] = []

    for info in discovered:
        try:
            registry.register(info.instance)
            loaded.append(info)
            logger.info(
                "Loaded plugin invariant '%s' from %s",
                info.name,
                info.module_path,
            )
        except ValueError as e:
            # Name already registered
            logger.warning(
                "Skipping plugin '%s': %s", info.name, e
            )

    return loaded


def _load_module_invariants(py_file: Path) -> list[PluginInfo]:
    """Import a Python file and extract Invariant implementations.

    Args:
        py_file: Path to the Python source file.

    Returns:
        List of PluginInfo for each invariant class found.
    """
    module_name = f"sim_debugger_plugin_{py_file.stem}"

    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)

    # Temporarily add the plugin directory to sys.path so relative
    # imports within the plugin can work
    plugin_dir = str(py_file.parent)
    path_added = False
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
        path_added = True

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.warning("Failed to execute plugin module %s: %s", py_file, e)
        return []
    finally:
        if path_added:
            try:
                sys.path.remove(plugin_dir)
            except ValueError:
                pass

    plugins: list[PluginInfo] = []

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue

        obj = getattr(module, attr_name)

        # Must be a class (not an instance)
        if not inspect.isclass(obj):
            continue

        # Check if it looks like an Invariant implementation
        if not _is_invariant_class(obj):
            continue

        # Try to instantiate
        try:
            instance = obj()
        except Exception as e:
            logger.warning(
                "Failed to instantiate %s from %s: %s",
                attr_name,
                py_file,
                e,
            )
            continue

        # Verify the instance has the required properties
        try:
            name = instance.name
            desc = instance.description
            _ = instance.default_threshold
        except Exception:
            continue

        plugins.append(PluginInfo(
            name=name,
            description=desc,
            module_path=str(py_file),
            class_name=attr_name,
            instance=instance,
        ))

    return plugins


def _is_invariant_class(cls: type) -> bool:
    """Check if a class implements the Invariant protocol.

    Checks for the required methods and properties without instantiating.
    """
    required_attrs = ["name", "description", "default_threshold", "compute", "check", "applicable"]

    for attr in required_attrs:
        if not hasattr(cls, attr):
            return False

    # Check that compute and check are callable
    if not callable(getattr(cls, "compute", None)):
        return False
    if not callable(getattr(cls, "check", None)):
        return False
    if not callable(getattr(cls, "applicable", None)):
        return False

    return True
