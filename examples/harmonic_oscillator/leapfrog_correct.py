"""Correct harmonic oscillator using symplectic Euler (leapfrog variant).

Symplectic Euler: v_{n+1} = v_n + a_n * dt, x_{n+1} = x_n + v_{n+1} * dt
This is a first-order symplectic integrator that preserves the
Hamiltonian structure and has bounded energy error.

sim-debugger should report no energy violations at default threshold.

Run with:
    sim-debugger run examples/harmonic_oscillator/leapfrog_correct.py
"""

import numpy as np

# Initial conditions
positions = np.array([[1.0, 0.0, 0.0]])
velocities = np.array([[0.0, 0.0, 0.0]])
masses = np.array([1.0])
dt = 0.001

# Symplectic Euler integration (energy-conserving to O(dt))
for t in range(10000):
    forces = -positions  # Spring force F = -kx, k=1

    # Symplectic Euler: update v first, then x with new v
    velocities = velocities + forces * dt
    positions = positions + velocities * dt

    # Compute potential energy for total energy tracking
    potential_energy = 0.5 * np.sum(positions**2, axis=-1, keepdims=True)
