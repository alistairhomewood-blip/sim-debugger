"""Configuration file support for sim-debugger.

Loads settings from `.sim-debugger.toml` files, with a hierarchical
lookup: project directory -> user home -> built-in defaults. Uses
Python's built-in tomllib (3.11+) for TOML parsing.

Example `.sim-debugger.toml`::

    [monitor]
    invariants = ["Total Energy", "Boris Energy"]
    check_interval = 1
    history_size = 200
    mode = "default"

    [thresholds]
    "Total Energy" = 1e-6
    "Boris Energy" = 1e-8
    "Charge Conservation" = 1e-12

    [output]
    format = "text"
    log_file = "violations.log"
    json_file = "violations.json"

    [performance]
    lightweight_interval = 100
    state_copy_mode = "view"  # "copy" or "view"

    [plugins]
    paths = ["./my_invariants"]

Usage::

    from sim_debugger.core.config import load_config, SimDebuggerConfig

    config = load_config()  # Searches CWD, then home, then defaults
    config = load_config("/path/to/.sim-debugger.toml")  # Explicit path
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MonitorConfig:
    """Configuration for the invariant monitor."""

    invariants: list[str] | None = None
    check_interval: int = 1
    history_size: int = 100
    mode: str = "default"  # "lightweight", "default", "full"


@dataclass
class ThresholdConfig:
    """Per-invariant threshold overrides."""

    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Configuration for output formats."""

    format: str = "text"  # "text", "json"
    log_file: str | None = None
    json_file: str | None = None
    hdf5_file: str | None = None
    verbose: bool = False


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""

    lightweight_interval: int = 100
    state_copy_mode: str = "copy"  # "copy" or "view"
    max_memory_mb: int | None = None


@dataclass
class PluginConfig:
    """Plugin system configuration."""

    paths: list[str] = field(default_factory=list)
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)


@dataclass
class SimDebuggerConfig:
    """Complete sim-debugger configuration.

    Aggregates all configuration sections. Can be loaded from TOML
    files or constructed programmatically.
    """

    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    config_file: str | None = None

    def get_check_interval(self) -> int:
        """Get the effective check interval based on mode."""
        if self.monitor.mode == "lightweight":
            return self.performance.lightweight_interval
        return self.monitor.check_interval

    def get_thresholds(self) -> dict[str, float]:
        """Get the threshold overrides dict."""
        return dict(self.thresholds.thresholds)


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_CONFIG_FILENAMES = [
    ".sim-debugger.toml",
    "sim-debugger.toml",
    ".sim_debugger.toml",
]


def find_config_file(start_dir: str | None = None) -> Path | None:
    """Search for a config file in the project directory and user home.

    Search order:
    1. start_dir (or CWD if None)
    2. Parent directories up to the filesystem root
    3. User home directory (~/)

    Args:
        start_dir: Directory to start searching from. Defaults to CWD.

    Returns:
        Path to the config file, or None if not found.
    """
    if start_dir is None:
        start_dir = os.getcwd()

    # Search upward from start_dir
    current = Path(start_dir).resolve()
    while True:
        for filename in _CONFIG_FILENAMES:
            candidate = current / filename
            if candidate.is_file():
                return candidate

        parent = current.parent
        if parent == current:
            break
        current = parent

    # Check user home
    home = Path.home()
    for filename in _CONFIG_FILENAMES:
        candidate = home / filename
        if candidate.is_file():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(
    path: str | Path | None = None,
    start_dir: str | None = None,
) -> SimDebuggerConfig:
    """Load sim-debugger configuration from a TOML file.

    If no path is specified, searches for a config file automatically.
    If no config file is found, returns default configuration.

    Args:
        path: Explicit path to a config file.
        start_dir: Directory to start searching from (if path is None).

    Returns:
        A SimDebuggerConfig with values from the file (or defaults).

    Raises:
        FileNotFoundError: If an explicit path is given but does not exist.
        ValueError: If the config file has invalid syntax or values.
    """
    config_path: Path | None
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        config_path = find_config_file(start_dir)

    if config_path is None:
        return SimDebuggerConfig()

    return _parse_config_file(config_path)


def _parse_monitor_section(mon: dict[str, Any], config: SimDebuggerConfig) -> None:
    """Parse the [monitor] section of the config file."""
    if "invariants" in mon:
        config.monitor.invariants = list(mon["invariants"])
    if "check_interval" in mon:
        config.monitor.check_interval = int(mon["check_interval"])
    if "history_size" in mon:
        config.monitor.history_size = int(mon["history_size"])
    if "mode" in mon:
        mode = str(mon["mode"])
        if mode not in ("lightweight", "default", "full"):
            raise ValueError(
                f"Invalid monitor.mode: '{mode}'. "
                f"Must be 'lightweight', 'default', or 'full'."
            )
        config.monitor.mode = mode


def _parse_thresholds_section(
    thresholds: dict[str, Any], config: SimDebuggerConfig,
) -> None:
    """Parse the [thresholds] section of the config file."""
    for name, value in thresholds.items():
        try:
            config.thresholds.thresholds[name] = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid threshold value for '{name}': {value}"
            ) from exc


def _parse_output_section(out: dict[str, Any], config: SimDebuggerConfig) -> None:
    """Parse the [output] section of the config file."""
    if "format" in out:
        fmt = str(out["format"])
        if fmt not in ("text", "json"):
            raise ValueError(
                f"Invalid output.format: '{fmt}'. Must be 'text' or 'json'."
            )
        config.output.format = fmt
    if "log_file" in out:
        config.output.log_file = str(out["log_file"])
    if "json_file" in out:
        config.output.json_file = str(out["json_file"])
    if "hdf5_file" in out:
        config.output.hdf5_file = str(out["hdf5_file"])
    if "verbose" in out:
        config.output.verbose = bool(out["verbose"])


def _parse_performance_section(
    perf: dict[str, Any], config: SimDebuggerConfig,
) -> None:
    """Parse the [performance] section of the config file."""
    if "lightweight_interval" in perf:
        config.performance.lightweight_interval = int(
            perf["lightweight_interval"]
        )
    if "state_copy_mode" in perf:
        mode = str(perf["state_copy_mode"])
        if mode not in ("copy", "view"):
            raise ValueError(
                f"Invalid performance.state_copy_mode: '{mode}'. "
                f"Must be 'copy' or 'view'."
            )
        config.performance.state_copy_mode = mode
    if "max_memory_mb" in perf:
        config.performance.max_memory_mb = int(perf["max_memory_mb"])


def _parse_plugins_section(
    plug: dict[str, Any], config: SimDebuggerConfig,
) -> None:
    """Parse the [plugins] section of the config file."""
    if "paths" in plug:
        config.plugins.paths = [str(p) for p in plug["paths"]]
    if "enabled" in plug:
        config.plugins.enabled = [str(p) for p in plug["enabled"]]
    if "disabled" in plug:
        config.plugins.disabled = [str(p) for p in plug["disabled"]]


# Dispatch dict mapping config section names to their parsers
_SECTION_PARSERS: dict[str, Any] = {
    "monitor": _parse_monitor_section,
    "thresholds": _parse_thresholds_section,
    "output": _parse_output_section,
    "performance": _parse_performance_section,
    "plugins": _parse_plugins_section,
}


def _parse_config_file(path: Path) -> SimDebuggerConfig:
    """Parse a TOML config file into a SimDebuggerConfig.

    Uses a dispatch-based approach with one parser per config section
    to keep cyclomatic complexity low.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed configuration.

    Raises:
        ValueError: If the file has invalid syntax or values.
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ValueError(f"Error parsing config file {path}: {e}") from e

    config = SimDebuggerConfig(config_file=str(path))

    for section_name, parser in _SECTION_PARSERS.items():
        if section_name in data:
            parser(data[section_name], config)

    return config


def save_default_config(path: str | Path) -> None:
    """Write a default configuration file with documented options.

    Useful for `sim-debugger init` to create a starter config file.

    Args:
        path: Where to write the config file.
    """
    template = '''\
# sim-debugger configuration
# See: https://github.com/your-org/sim-debugger

[monitor]
# Invariant names to monitor. Comment out for auto-detection.
# invariants = ["Total Energy", "Boris Energy"]
check_interval = 1       # Check every N timesteps (1 = every step)
history_size = 100       # Ring buffer size for temporal localisation
mode = "default"         # "lightweight", "default", or "full"

[thresholds]
# Per-invariant relative tolerance overrides.
# Uncomment and adjust as needed.
# "Total Energy" = 1e-6
# "Linear Momentum" = 1e-6
# "Boris Energy" = 1e-8
# "Charge Conservation" = 1e-12

[output]
format = "text"          # "text" or "json"
# log_file = "violations.log"
# json_file = "violations.json"
verbose = false

[performance]
lightweight_interval = 100  # Check interval in lightweight mode
state_copy_mode = "copy"    # "copy" (safe) or "view" (fast, risk of mutation)
# max_memory_mb = 1024      # Memory limit for state history

[plugins]
# paths = ["./my_invariants"]  # Directories to search for plugin invariants
# enabled = []                 # Explicitly enable these plugins
# disabled = []                # Explicitly disable these plugins
'''
    Path(path).write_text(template)
