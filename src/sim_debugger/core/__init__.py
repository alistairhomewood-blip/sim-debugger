"""Core module: invariant definitions, violation detection engine, data models."""

from sim_debugger.core.auto_detect import InvariantSuggestion, auto_detect_invariants
from sim_debugger.core.config import SimDebuggerConfig, load_config, save_default_config
from sim_debugger.core.history import InvariantTrend, ViolationHistory
from sim_debugger.core.invariants import (
    AngularMomentumInvariant,
    ChargeConservationInvariant,
    Invariant,
    InvariantRegistry,
    LinearMomentumInvariant,
    ParticleCountInvariant,
    TotalEnergyInvariant,
    create_default_registry,
)
from sim_debugger.core.monitor import Monitor
from sim_debugger.core.plugins import PluginInfo, discover_plugins, load_plugins
from sim_debugger.core.state import SimulationState, StateHistory
from sim_debugger.core.violations import (
    LocalisationResult,
    SourceLocalisation,
    SpatialLocalisation,
    TemporalLocalisation,
    Violation,
    ViolationPattern,
    ViolationSeverity,
    classify_severity,
)

__all__ = [
    # State
    "SimulationState",
    "StateHistory",
    # Violations
    "Violation",
    "ViolationSeverity",
    "ViolationPattern",
    "LocalisationResult",
    "TemporalLocalisation",
    "SpatialLocalisation",
    "SourceLocalisation",
    "classify_severity",
    # Invariants
    "Invariant",
    "InvariantRegistry",
    "TotalEnergyInvariant",
    "LinearMomentumInvariant",
    "AngularMomentumInvariant",
    "ChargeConservationInvariant",
    "ParticleCountInvariant",
    "create_default_registry",
    # Monitor
    "Monitor",
    # Phase 3: Config
    "SimDebuggerConfig",
    "load_config",
    "save_default_config",
    # Phase 3: Auto-detection
    "auto_detect_invariants",
    "InvariantSuggestion",
    # Phase 3: History
    "ViolationHistory",
    "InvariantTrend",
    # Phase 3: Plugins
    "discover_plugins",
    "load_plugins",
    "PluginInfo",
]
