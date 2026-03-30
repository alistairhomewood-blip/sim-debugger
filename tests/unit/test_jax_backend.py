"""Tests for the JAX backend adapter.

JAX may or may not be installed. Tests are structured to:
1. Always test the availability detection and graceful fallback
2. Skip JAX-specific functionality tests if JAX is not installed
"""

import numpy as np
import pytest

from sim_debugger.backends.jax_backend import JAXBackend, _check_jax
from sim_debugger.core.state import SimulationState

# Check if JAX is available for conditional test skipping
_jax_installed = _check_jax()

skip_without_jax = pytest.mark.skipif(
    not _jax_installed, reason="JAX is not installed"
)


# ===========================================================================
# Availability detection
# ===========================================================================

class TestJAXAvailability:
    def test_is_available_returns_bool(self):
        """is_available() returns a boolean."""
        result = JAXBackend.is_available()
        assert isinstance(result, bool)

    def test_is_available_is_consistent(self):
        """Calling is_available() twice returns the same result."""
        assert JAXBackend.is_available() == JAXBackend.is_available()


# ===========================================================================
# Graceful fallback when JAX is not available
# ===========================================================================

class TestJAXGracefulFallback:
    def test_is_jax_array_numpy(self):
        """is_jax_array returns False for NumPy arrays regardless of JAX."""
        arr = np.array([1.0, 2.0, 3.0])
        # If JAX is not available, should return False
        # If JAX is available, should also return False (it's numpy, not jax)
        assert JAXBackend.is_jax_array(arr) is False

    def test_is_jax_array_scalar(self):
        """is_jax_array returns False for Python scalars."""
        assert JAXBackend.is_jax_array(42) is False
        assert JAXBackend.is_jax_array(3.14) is False

    def test_is_jax_array_string(self):
        """is_jax_array returns False for non-array types."""
        assert JAXBackend.is_jax_array("hello") is False

    def test_detect_backend_positive(self):
        """detect_backend detects JAX imports in source code."""
        assert JAXBackend.detect_backend("import jax") is True
        assert JAXBackend.detect_backend("import jax.numpy as jnp") is True
        assert JAXBackend.detect_backend("from jax import lax") is True
        assert JAXBackend.detect_backend("x = jax.numpy.array([1])") is True

    def test_detect_backend_negative(self):
        """detect_backend returns False for non-JAX source code."""
        assert JAXBackend.detect_backend("import numpy as np") is False
        assert JAXBackend.detect_backend("x = 42") is False

    def test_ensure_numpy_works_with_numpy(self):
        """ensure_numpy_for_invariant works with plain NumPy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        result = JAXBackend.ensure_numpy_for_invariant(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_ensure_numpy_works_with_list(self):
        """ensure_numpy_for_invariant works with Python lists."""
        result = JAXBackend.ensure_numpy_for_invariant([1.0, 2.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])


# ===========================================================================
# Tests requiring JAX to be installed
# ===========================================================================

@skip_without_jax
class TestJAXWithJAX:
    """Tests that require JAX to be installed."""

    def test_is_jax_array_true(self):
        """is_jax_array returns True for JAX arrays."""
        import jax.numpy as jnp
        arr = jnp.array([1.0, 2.0, 3.0])
        assert JAXBackend.is_jax_array(arr) is True

    def test_to_numpy(self):
        """to_numpy converts JAX arrays to NumPy."""
        import jax.numpy as jnp
        jax_arr = jnp.array([1.0, 2.0, 3.0])
        np_arr = JAXBackend.to_numpy(jax_arr)
        assert isinstance(np_arr, np.ndarray)
        np.testing.assert_array_almost_equal(np_arr, [1.0, 2.0, 3.0])

    def test_to_numpy_preserves_numpy(self):
        """to_numpy returns NumPy arrays unchanged."""
        arr = np.array([1.0, 2.0])
        result = JAXBackend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_numpy_2d(self):
        """to_numpy handles multi-dimensional JAX arrays."""
        import jax.numpy as jnp
        jax_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        np_arr = JAXBackend.to_numpy(jax_arr)
        assert np_arr.shape == (2, 2)
        assert np_arr[1, 1] == pytest.approx(4.0)

    def test_copy_array_jax(self):
        """copy_array creates a new NumPy array from JAX input."""
        import jax.numpy as jnp
        jax_arr = jnp.array([10.0, 20.0])
        result = JAXBackend.copy_array(jax_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0])

    def test_copy_array_numpy(self):
        """copy_array creates a copy of NumPy arrays."""
        arr = np.array([1.0, 2.0])
        result = JAXBackend.copy_array(arr)
        assert isinstance(result, np.ndarray)
        # Modify original; copy should be unaffected
        arr[0] = 999.0
        assert result[0] == pytest.approx(1.0)

    def test_capture_state_basic(self):
        """capture_state converts JAX arrays to NumPy in the state."""
        import jax.numpy as jnp
        local_vars = {
            "velocities": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "masses": jnp.array([1.0, 2.0]),
            "dt": 0.01,
        }
        state = JAXBackend.capture_state(
            local_vars, timestep=5, time=0.05,
        )
        assert isinstance(state, SimulationState)
        assert state.timestep == 5
        assert state.time == 0.05
        assert "velocities" in state.arrays
        assert isinstance(state.arrays["velocities"], np.ndarray)
        assert state.arrays["velocities"].shape == (2, 3)
        assert state.metadata.get("dt") == pytest.approx(0.01)

    def test_capture_state_with_aliases(self):
        """capture_state maps JAX variable names to standard state keys."""
        import jax.numpy as jnp
        local_vars = {
            "v": jnp.array([[3.0, 4.0, 0.0]]),
            "m": jnp.array([2.0]),
            "x": jnp.array([[1.0, 0.0, 0.0]]),
        }
        state = JAXBackend.capture_state(local_vars)
        assert "velocities" in state.arrays
        assert "masses" in state.arrays
        assert "positions" in state.arrays

    def test_capture_state_explicit_arrays(self):
        """capture_state with explicit array_names filters correctly."""
        import jax.numpy as jnp
        local_vars = {
            "velocities": jnp.array([[1.0, 0.0, 0.0]]),
            "masses": jnp.array([1.0]),
            "unrelated": jnp.array([99.0]),
        }
        state = JAXBackend.capture_state(
            local_vars, array_names=["velocities", "masses"],
        )
        assert "velocities" in state.arrays
        assert "masses" in state.arrays
        assert "unrelated" not in state.arrays

    def test_capture_state_scalar_metadata(self):
        """capture_state converts JAX scalar metadata to Python float."""
        import jax.numpy as jnp
        local_vars = {
            "velocities": jnp.array([[1.0, 0.0, 0.0]]),
            "dt": jnp.float32(0.001),
        }
        state = JAXBackend.capture_state(local_vars)
        assert isinstance(state.metadata.get("dt"), float)
        assert state.metadata["dt"] == pytest.approx(0.001, abs=1e-5)

    def test_ensure_numpy_for_invariant_jax(self):
        """ensure_numpy_for_invariant converts JAX arrays."""
        import jax.numpy as jnp
        jax_arr = jnp.array([1.0, 2.0, 3.0])
        result = JAXBackend.ensure_numpy_for_invariant(jax_arr)
        assert isinstance(result, np.ndarray)

    def test_create_io_callback(self):
        """create_io_callback returns a callable."""
        import jax
        import jax.numpy as jnp

        captured_values = []

        def my_callback(x):
            captured_values.append(float(jnp.sum(x)))
            return x

        jit_callback = JAXBackend.create_io_callback(my_callback)

        # Use it inside a jit-compiled function
        @jax.jit
        def f(x):
            return jit_callback(x)

        x = jnp.array([1.0, 2.0, 3.0])
        result = f(x)
        np.testing.assert_array_almost_equal(np.array(result), [1.0, 2.0, 3.0])
        assert len(captured_values) == 1
        assert captured_values[0] == pytest.approx(6.0)


# ===========================================================================
# Tests for when JAX is NOT available
# ===========================================================================

class TestJAXNotAvailableGuards:
    def test_require_jax_raises_without_jax(self):
        """If JAX is not installed, methods that require it should raise."""
        if _jax_installed:
            pytest.skip("JAX is installed; skipping not-available test")

        with pytest.raises(ImportError, match="JAX is required"):
            JAXBackend.to_numpy(np.array([1.0]))

        with pytest.raises(ImportError, match="JAX is required"):
            JAXBackend.capture_state({})
