"""Simulation state snapshot and history models.

Defines SimulationState for capturing the full simulation state at a timestep,
and StateHistory as a ring buffer for temporal localisation lookback.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationState:
    """Snapshot of a simulation's state at a single timestep.

    This is the fundamental data structure that invariant monitors operate on.
    It holds named arrays (positions, velocities, fields, etc.) plus metadata
    about the simulation parameters.

    Attributes:
        timestep: The integer timestep index.
        time: The physical simulation time.
        arrays: Named state arrays. Keys are user-defined (e.g. "positions",
                "velocities", "E_field", "B_field", "charge_density").
        metadata: Simulation parameters (dt, grid spacing, physical constants, etc.).
        source_file: Path to the source file being monitored.
        source_line: Current line number in the source (for source localisation).
    """

    timestep: int
    time: float
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_line: int = 0

    def copy(self) -> SimulationState:
        """Create a deep copy of this state (copies all arrays)."""
        return SimulationState(
            timestep=self.timestep,
            time=self.time,
            arrays={k: np.copy(v) for k, v in self.arrays.items()},
            metadata=copy.deepcopy(self.metadata),
            source_file=self.source_file,
            source_line=self.source_line,
        )

    def has_array(self, name: str) -> bool:
        """Check if the state contains a named array."""
        return name in self.arrays

    def get_array(self, name: str) -> np.ndarray:
        """Get a named array, raising KeyError if not present."""
        if name not in self.arrays:
            raise KeyError(
                f"State does not contain array '{name}'. "
                f"Available arrays: {list(self.arrays.keys())}"
            )
        return self.arrays[name]


class StateHistory:
    """Ring buffer of simulation state snapshots for temporal localisation.

    Stores the most recent `max_size` state snapshots. Full copies are kept
    for the most recent `full_copy_count` states; older states store only
    the invariant values (not the full arrays) to save memory.

    Attributes:
        max_size: Maximum number of entries in the ring buffer.
        full_copy_count: Number of most-recent states to keep as full copies.
    """

    def __init__(self, max_size: int = 100, full_copy_count: int = 10) -> None:
        self.max_size = max_size
        self.full_copy_count = full_copy_count
        self._states: deque[SimulationState] = deque(maxlen=max_size)
        self._invariant_values: deque[dict[str, float]] = deque(maxlen=max_size)

    def push(
        self,
        state: SimulationState,
        invariant_values: dict[str, float] | None = None,
    ) -> None:
        """Add a state snapshot to the history.

        The state is deep-copied to prevent mutation issues. If the buffer
        is beyond the full_copy_count, older states have their arrays cleared.

        Args:
            state: The simulation state to store.
            invariant_values: Optional dict of invariant_name -> computed value.
        """
        self._states.append(state.copy())
        self._invariant_values.append(invariant_values or {})

        # Trim arrays from old states to save memory
        if len(self._states) > self.full_copy_count:
            old_idx = len(self._states) - self.full_copy_count - 1
            if old_idx >= 0:
                old_state = self._states[old_idx]
                # Replace arrays with empty dict to free memory
                # but keep timestep/time/metadata for localisation
                old_state.arrays = {}

    def get_invariant_trajectory(
        self, invariant_name: str
    ) -> list[tuple[int, float]]:
        """Get the time series of an invariant's values over the history.

        Args:
            invariant_name: Name of the invariant.

        Returns:
            List of (timestep, value) pairs in chronological order.
        """
        trajectory = []
        for state, values in zip(self._states, self._invariant_values):
            if invariant_name in values:
                trajectory.append((state.timestep, values[invariant_name]))
        return trajectory

    def get_state_at(self, timestep: int) -> SimulationState | None:
        """Retrieve the state snapshot for a specific timestep, if available.

        Returns None if the timestep is not in the buffer or its arrays
        have been trimmed.
        """
        for state in self._states:
            if state.timestep == timestep and state.arrays:
                return state
        return None

    def get_recent_states(self, count: int) -> list[SimulationState]:
        """Get the most recent `count` state snapshots."""
        states = list(self._states)
        return states[-count:]

    @property
    def latest(self) -> SimulationState | None:
        """The most recently added state, or None if empty."""
        return self._states[-1] if self._states else None

    def __len__(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        """Clear all stored states."""
        self._states.clear()
        self._invariant_values.clear()
