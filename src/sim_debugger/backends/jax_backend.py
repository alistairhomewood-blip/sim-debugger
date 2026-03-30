"""JAX backend adapter for sim-debugger.

Provides array operations and state capture for simulations using JAX.
Handles JAX's functional paradigm (no in-place mutation), JIT compilation
boundaries, and device placement (GPU/TPU arrays).

Strategy:
    - Uses jax.experimental.io_callback for state extraction during JIT
      compilation, allowing monitoring inside jit-compiled functions.
    - Converts JAX arrays to NumPy for invariant computation (invariants
      are always computed on the host CPU).
    - Detects whether JAX is available at import time and provides graceful
      fallback when it is not installed.

Usage::

    from sim_debugger.backends.jax_backend import JAXBackend

    if JAXBackend.is_available():
        backend = JAXBackend()
        state = backend.capture_state(local_vars, timestep=t)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from sim_debugger.core.state import SimulationState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JAX availability detection
# ---------------------------------------------------------------------------

_jax_available: bool | None = None
_jax: Any = None
_jnp: Any = None


def _check_jax() -> bool:
    """Lazily check if JAX is importable and cache the result."""
    global _jax_available, _jax, _jnp
    if _jax_available is not None:
        return _jax_available
    try:
        import jax
        import jax.numpy as jnp

        _jax = jax
        _jnp = jnp
        _jax_available = True
        logger.debug("JAX backend available: version %s", jax.__version__)
    except ImportError:
        _jax_available = False
        logger.debug("JAX not available; JAX backend disabled.")
    return _jax_available


def _require_jax() -> None:
    """Raise ImportError if JAX is not available."""
    if not _check_jax():
        raise ImportError(
            "JAX is required for the JAX backend but is not installed. "
            "Install it with: pip install 'jax[cpu]' or 'jax[cuda]'."
        )


# ---------------------------------------------------------------------------
# JAX Backend
# ---------------------------------------------------------------------------

class JAXBackend:
    """JAX backend for array operations and state capture.

    Wraps JAX-specific operations used by invariant monitors, and provides
    state-capture utilities that handle JAX's functional paradigm.

    Key design decisions:
    - Invariant computation always happens on CPU (NumPy). JAX arrays are
      transferred to host before computation to avoid device placement issues.
    - The io_callback mechanism allows state extraction inside JIT-compiled
      functions without breaking the compilation trace.
    - No in-place mutation: all state capture creates new arrays (copies).
    """

    name: str = "jax"

    @staticmethod
    def is_available() -> bool:
        """Check if JAX is installed and importable."""
        return _check_jax()

    @staticmethod
    def is_jax_array(obj: Any) -> bool:
        """Check if an object is a JAX array.

        Returns False if JAX is not available, rather than raising.
        """
        if not _check_jax():
            return False
        return isinstance(obj, _jax.Array)

    @staticmethod
    def to_numpy(arr: Any) -> np.ndarray:
        """Convert a JAX array to a NumPy array on the host.

        Handles device transfer transparently. If the input is already
        a NumPy array, returns it unchanged.

        Args:
            arr: A JAX array, NumPy array, or Python scalar.

        Returns:
            A NumPy ndarray on the host CPU.
        """
        _require_jax()

        if isinstance(arr, np.ndarray):
            return arr

        if isinstance(arr, _jax.Array):
            # Transfer from device (GPU/TPU) to host CPU
            return np.asarray(arr)

        # Fallback: try converting via np.asarray
        return np.asarray(arr)

    @staticmethod
    def copy_array(arr: Any) -> np.ndarray:
        """Create a NumPy copy of a JAX or NumPy array.

        Always returns a contiguous NumPy array on the host.
        """
        _require_jax()

        if isinstance(arr, _jax.Array):
            return np.array(arr)
        if isinstance(arr, np.ndarray):
            return np.copy(arr)
        return np.array(arr)

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
        """Capture a SimulationState from variables, converting JAX arrays.

        Inspects local_vars for JAX arrays and known simulation parameters.
        All JAX arrays are converted to NumPy (host CPU) for invariant
        computation. This transfer has overhead but ensures compatibility
        with all invariant monitors.

        Args:
            local_vars: The local variables dict (e.g. from frame.f_locals).
            timestep: Current timestep index.
            time: Current simulation time.
            source_file: Source file being monitored.
            source_line: Current line in source.
            array_names: If provided, only capture these named arrays.
            metadata_names: If provided, capture these as metadata.

        Returns:
            A SimulationState with NumPy arrays (transferred from device).
        """
        _require_jax()

        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}

        # Same alias map as NumPy backend for consistency
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

        METADATA_NAMES: set[str] = {
            "dt", "dx", "dy", "dz", "eps_0", "mu_0",
            "num_particles", "particle_count",
            "grid_size", "domain_size",
            "q_over_m", "omega_c",
        }

        def _is_array(obj: Any) -> bool:
            """Check if obj is a JAX or NumPy array."""
            if isinstance(obj, np.ndarray):
                return True
            if isinstance(obj, _jax.Array):
                return True
            return False

        def _to_numpy(obj: Any) -> np.ndarray:
            """Convert JAX/NumPy array to NumPy copy."""
            if isinstance(obj, _jax.Array):
                return np.array(obj)
            return np.copy(obj)

        if array_names is not None:
            for name in array_names:
                if name in local_vars and _is_array(local_vars[name]):
                    key = ARRAY_ALIASES.get(name, name)
                    arrays[key] = _to_numpy(local_vars[name])
        else:
            for var_name, var_val in local_vars.items():
                if _is_array(var_val):
                    key = ARRAY_ALIASES.get(var_name, var_name)
                    arrays[key] = _to_numpy(var_val)

        if metadata_names is not None:
            for name in metadata_names:
                if name in local_vars:
                    val = local_vars[name]
                    # Convert JAX scalars to Python floats
                    if isinstance(val, _jax.Array) and val.ndim == 0:
                        val = float(val)
                    metadata[name] = val
        else:
            for var_name, var_val in local_vars.items():
                if var_name in METADATA_NAMES:
                    if isinstance(var_val, _jax.Array) and var_val.ndim == 0:
                        var_val = float(var_val)
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
        """Detect if the source code uses JAX.

        Simple heuristic: look for JAX imports.
        """
        return (
            "import jax" in source_code
            or "from jax" in source_code
            or "jax.numpy" in source_code
        )

    @staticmethod
    def create_io_callback(
        callback_fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Create a JAX io_callback wrapper for state extraction during JIT.

        Uses jax.experimental.io_callback to inject a Python callback
        into a JIT-compiled function trace. This allows state extraction
        at specific points without breaking the compilation.

        The callback receives JAX arrays and can perform side effects
        (like logging or state capture) but must return a value with
        the same structure and dtype as the input.

        Args:
            callback_fn: A Python function to call during JIT execution.
                        It receives JAX arrays and should return arrays
                        with the same shape/dtype.

        Returns:
            A wrapped function suitable for use inside jax.jit.

        Example::

            def monitor_step(velocities):
                # Extract state for monitoring (side effect)
                state = backend.capture_state({"velocities": velocities}, timestep=t)
                monitor.check(state)
                return velocities  # Must return same-shaped output

            jit_safe_monitor = backend.create_io_callback(monitor_step)

            @jax.jit
            def simulation_step(v, dt):
                v_new = v + force * dt
                v_new = jit_safe_monitor(v_new)  # Monitoring happens here
                return v_new
        """
        _require_jax()

        def wrapper(*args: Any) -> Any:
            """io_callback wrapper that calls the user function."""
            result = callback_fn(*args)
            return result

        def jit_compatible(*args: Any) -> Any:
            """Wrapper that uses io_callback for JIT compatibility."""
            if len(args) == 1:
                result_shape = _jax.ShapeDtypeStruct(
                    args[0].shape, args[0].dtype,
                )
                return _jax.experimental.io_callback(
                    wrapper,
                    result_shape,
                    args[0],
                    ordered=True,
                )
            else:
                result_shapes = tuple(
                    _jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args
                )
                return _jax.experimental.io_callback(
                    wrapper,
                    result_shapes,
                    *args,
                    ordered=True,
                )

        return jit_compatible

    @staticmethod
    def ensure_numpy_for_invariant(arr: Any) -> np.ndarray:
        """Ensure an array is NumPy for invariant computation.

        Invariant monitors expect NumPy arrays. This method handles
        the conversion from JAX arrays, including device transfer.

        Args:
            arr: A JAX array, NumPy array, or Python scalar.

        Returns:
            A NumPy ndarray.
        """
        if not _check_jax():
            if isinstance(arr, np.ndarray):
                return arr
            return np.asarray(arr)

        if isinstance(arr, _jax.Array):
            return np.asarray(arr)
        if isinstance(arr, np.ndarray):
            return arr
        return np.asarray(arr)
