# tests/benchmarks/__init__.py
# Placeholder only. No implementation code.
# Benchmark suite: known-buggy simulations with known violations.
# Used to verify zero false-negative rate.
# Each benchmark is a simulation with:
#   - A known bug (e.g., wrong sign in force, missing factor of 2)
#   - The expected violation type, magnitude, and timestep
#   - A PASS criterion: sim-debugger must detect the violation
