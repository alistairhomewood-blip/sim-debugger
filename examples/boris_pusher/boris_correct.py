"""Correct Boris pusher implementation.

Boris algorithm for a charged particle in a magnetic field.
In B-only mode (E=0), the Boris rotation preserves |v| exactly.

sim-debugger should report zero energy violations.

Run with:
    sim-debugger run examples/boris_pusher/boris_correct.py --invariants "Boris Energy"
"""

import numpy as np

# Physical setup: particle gyrating in uniform B field
positions = np.array([[1.0, 0.0, 0.0]])
velocities = np.array([[0.0, 1.0, 0.0]])
masses = np.array([1.0])
charges = np.array([1.0])

E = np.array([0.0, 0.0, 0.0])  # No electric field
B = np.array([0.0, 0.0, 1.0])  # Uniform B in z-direction
dt = 0.01
q = 1.0
m = 1.0

for t in range(1000):
    x = positions[0]
    v = velocities[0]

    qdt_2m = q * dt / (2.0 * m)

    # Half E-push
    v_minus = v + qdt_2m * E

    # B-rotation (Boris rotation formula)
    t_vec = (q / m) * B * dt / 2.0
    t_mag_sq = np.dot(t_vec, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Half E-push
    v_new = v_plus + qdt_2m * E

    # Position update
    x_new = x + v_new * dt

    positions[0] = x_new
    velocities[0] = v_new
