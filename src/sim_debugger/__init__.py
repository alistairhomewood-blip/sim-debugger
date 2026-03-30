"""sim-debugger: Instrument simulations, monitor physical invariants, explain failures.

Public API:

    Monitor: the main orchestration class for invariant monitoring.
    SimulationState: snapshot of simulation state at a timestep.

    Decorators:
        @sim_debugger.monitor(): wrap a function for invariant monitoring.
        @sim_debugger.timestep: mark a function as a timestep update.
        @sim_debugger.track_state(): specify tracked variables.
        @sim_debugger.ignore: exclude from AST instrumentation.

    Functions:
        list_invariants(): list all registered invariant names.
        get_invariant(name): look up an invariant by name.
        register_invariant(inv): register a custom invariant.
        instrument(source): instrument source code (returns transformed source).
        run(script, invariants): instrument and run a script.

    Phase 3:
        auto_detect_invariants(): heuristically detect applicable invariants.
        load_config(): load .sim-debugger.toml configuration.
        ViolationHistory: track violations and trends across a full run.
        discover_plugins(): find custom invariant plugins in directories.
"""

from sim_debugger.core.auto_detect import auto_detect_invariants
from sim_debugger.core.config import SimDebuggerConfig, load_config
from sim_debugger.core.history import ViolationHistory
from sim_debugger.core.monitor import Monitor
from sim_debugger.core.plugins import discover_plugins, load_plugins
from sim_debugger.core.state import SimulationState, StateHistory
from sim_debugger.core.violations import Violation, ViolationSeverity
from sim_debugger.instrument.decorators import (
    ignore,
    monitor,
    timestep,
    track_state,
)

__version__ = "0.2.0"

__all__ = [
    "Monitor",
    "SimulationState",
    "StateHistory",
    "Violation",
    "ViolationSeverity",
    "monitor",
    "timestep",
    "track_state",
    "ignore",
    # Phase 3
    "auto_detect_invariants",
    "load_config",
    "SimDebuggerConfig",
    "ViolationHistory",
    "discover_plugins",
    "load_plugins",
    "__version__",
]
