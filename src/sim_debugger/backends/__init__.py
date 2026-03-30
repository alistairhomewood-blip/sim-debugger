"""Backend adapters: abstract array operations for invariant monitors."""

from sim_debugger.backends.numpy_backend import NumPyBackend

__all__ = ["NumPyBackend"]

# Optional backends are imported lazily to avoid hard dependencies.
# Use JAXBackend.is_available() and SciPyBackend.is_available() to check.
