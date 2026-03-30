"""NumPy backend adapter for sim-debugger.

Provides array operations via NumPy and state capture utilities.
This is the primary backend for the MVP (Phase 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim_debugger.core.state import SimulationState


class NumPyBackend:
    """NumPy backend for array operations and state capture.

    Wraps NumPy operations used by invariant monitors, and provides
    state-capture utilities for extracting simulation state from
    user code's local variables.
    """

    name: str = "numpy"

    @staticmethod
    def is_array(obj: Any) -> bool:
        """Check if an object is a NumPy array."""
        return isinstance(obj, np.ndarray)

    @staticmethod
    def to_numpy(arr: Any) -> np.ndarray:
        """Convert to NumPy array (no-op for NumPy arrays)."""
        if isinstance(arr, np.ndarray):
            return arr
        return np.asarray(arr)

    @staticmethod
    def copy_array(arr: np.ndarray) -> np.ndarray:
        """Create a deep copy of an array."""
        return np.copy(arr)

    @staticmethod
    def capture_state(
        local_vars: dict[str, Any],
        timestep: int = 0,
        time: float = 0.0,
        source_file: str = "",
        source_line: int = 0,
        array_names: list[str] | None = None,
        metadata_names: list[str] | None = None,
    ) -> SimulationState:
        """Capture a SimulationState from a frame's local variables.

        Inspects local_vars for numpy arrays and known simulation parameters.
        Copies arrays to prevent mutation issues.

        Args:
            local_vars: The local variables dict (e.g. from frame.f_locals).
            timestep: Current timestep index.
            time: Current simulation time.
            source_file: Source file being monitored.
            source_line: Current line in source.
            array_names: If provided, only capture these named arrays.
                         If None, capture all numpy arrays found.
            metadata_names: If provided, capture these as metadata.
                           If None, capture known scalar/string variables.

        Returns:
            A SimulationState with copied arrays and metadata.
        """
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}

        # Known array name mappings (user variable -> state key)
        ARRAY_ALIASES: dict[str, str] = {
            "x": "positions",
            "pos": "positions",
            "positions": "positions",
            "v": "velocities",
            "vel": "velocities",
            "velocities": "velocities",
            "m": "masses",
            "mass": "masses",
            "masses": "masses",
            "q": "charges",
            "charge": "charges",
            "charges": "charges",
            "E": "E_field",
            "E_field": "E_field",
            "B": "B_field",
            "B_field": "B_field",
            "rho": "charge_density",
            "charge_density": "charge_density",
            "F": "applied_force",
            "force": "applied_force",
            "applied_force": "applied_force",
            "E_at_particles": "E_at_particles",
            "B_at_particles": "B_at_particles",
            "potential_energy": "potential_energy",
        }

        # Known metadata names
        METADATA_NAMES: set[str] = {
            "dt", "dx", "dy", "dz", "eps_0", "mu_0",
            "num_particles", "particle_count",
            "grid_size", "domain_size",
            "q_over_m", "omega_c",
        }

        if array_names is not None:
            # Capture only specified arrays
            for name in array_names:
                if name in local_vars and isinstance(local_vars[name], np.ndarray):
                    key = ARRAY_ALIASES.get(name, name)
                    arrays[key] = np.copy(local_vars[name])
        else:
            # Auto-discover arrays
            for var_name, var_val in local_vars.items():
                if isinstance(var_val, np.ndarray):
                    key = ARRAY_ALIASES.get(var_name, var_name)
                    arrays[key] = np.copy(var_val)

        if metadata_names is not None:
            for name in metadata_names:
                if name in local_vars:
                    metadata[name] = local_vars[name]
        else:
            for var_name, var_val in local_vars.items():
                if var_name in METADATA_NAMES:
                    metadata[var_name] = var_val

        return SimulationState(
            timestep=timestep,
            time=time,
            arrays=arrays,
            metadata=metadata,
            source_file=source_file,
            source_line=source_line,
        )

    @staticmethod
    def detect_backend(source_code: str) -> bool:
        """Detect if the source code uses NumPy.

        Simple heuristic: look for numpy imports.
        """
        return "import numpy" in source_code or "from numpy" in source_code
