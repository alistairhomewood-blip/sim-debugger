"""Deliberately broken harmonic oscillator using Forward Euler.

Forward Euler is non-symplectic and causes monotonic energy growth.
sim-debugger should detect this and explain that the energy drift
is consistent with a non-symplectic integrator.

Run with:
    sim-debugger run examples/harmonic_oscillator/euler_broken.py
"""

import numpy as np

# Initial conditions
positions = np.array([[1.0, 0.0, 0.0]])
velocities = np.array([[0.0, 0.0, 0.0]])
masses = np.array([1.0])
dt = 0.05

# Forward Euler integration (non-symplectic -> energy grows)
for t in range(500):
    forces = -positions  # Spring force F = -kx, k=1

    # BUG: Forward Euler updates both from old state (non-symplectic)
    positions_new = positions + velocities * dt
    velocities_new = velocities + forces * dt

    positions = positions_new
    velocities = velocities_new

    # Compute potential energy for total energy tracking
    potential_energy = 0.5 * np.sum(positions**2, axis=-1, keepdims=True)
