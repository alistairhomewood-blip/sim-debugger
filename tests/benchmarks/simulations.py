"""Benchmark simulation implementations for sim-debugger testing.

Each simulation is a function that returns a list of SimulationState
snapshots. These are used to verify that the invariant monitors
correctly detect (or don't detect) violations.

Benchmark IDs from AUDIT_PLAN.md:
    B01: Harmonic oscillator (Euler forward) -- energy growth
    B02: Harmonic oscillator (leapfrog, large dt) -- exponential energy growth
    B03: Boris pusher (correct) -- no violation
    B04: Boris pusher (dt too large) -- energy instability
    B05: Boris pusher (full E-push instead of half) -- O(dt) energy drift
    B06: Boris pusher (wrong E-field sign) -- immediate energy jump
    B07: PIC charge deposition (conserving) -- no violation
    B08: PIC charge deposition (non-conserving) -- Gauss's law violation
    B09: Lorentz force (correct) -- no violation
    B10: Lorentz force (wrong sign on v x B) -- force error
    B11: N-body (leapfrog, correct) -- no violation
    B12: N-body (non-symmetric force) -- momentum violation
    B13: Particle boundary (reflecting, correct) -- no violation
    B14: Particle boundary (absorbing, leak) -- particle count growth
"""

from __future__ import annotations

import numpy as np

from sim_debugger.core.state import SimulationState


def _make_state(
    timestep: int,
    dt: float,
    **arrays,
) -> SimulationState:
    """Helper to build a state snapshot."""
    arr_dict = {}
    meta = {"dt": dt}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            arr_dict[k] = v.copy()
        else:
            meta[k] = v
    return SimulationState(
        timestep=timestep,
        time=timestep * dt,
        arrays=arr_dict,
        metadata=meta,
    )


# ===========================================================================
# B01: Harmonic oscillator -- Forward Euler (non-symplectic)
# ===========================================================================

def b01_harmonic_euler(num_steps: int = 500, dt: float = 0.05) -> list[SimulationState]:
    """Forward Euler on harmonic oscillator. Energy grows monotonically."""
    x = np.array([[1.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0]])
    m = np.array([1.0])

    states = []
    for t in range(num_steps):
        a = -x  # F = -x, k=1
        # Forward Euler: both update from old values (non-symplectic)
        x_new = x + v * dt
        v_new = v + a * dt
        x, v = x_new, v_new

        states.append(_make_state(
            t, dt,
            positions=x, velocities=v, masses=m,
            potential_energy=np.array([0.5 * np.sum(x**2)]),
        ))
    return states


# ===========================================================================
# B02: Harmonic oscillator -- Leapfrog with large dt (unstable)
# ===========================================================================

def b02_harmonic_large_dt(num_steps: int = 100, dt: float = 2.5) -> list[SimulationState]:
    """Leapfrog with dt > stability limit. Energy diverges exponentially.

    For leapfrog on a harmonic oscillator (omega=1), stability requires
    omega*dt < 2, i.e., dt < 2. With dt=2.5, it's unstable.
    """
    x = np.array([[1.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0]])
    m = np.array([1.0])

    states = []
    for t in range(num_steps):
        a = -x
        v = v + a * dt
        x = x + v * dt

        states.append(_make_state(
            t, dt,
            positions=x, velocities=v, masses=m,
            potential_energy=np.array([0.5 * np.sum(x**2)]),
        ))
    return states


# ===========================================================================
# B03: Boris pusher (correct implementation)
# ===========================================================================

def boris_push(
    x: np.ndarray,
    v: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
    q: float,
    m_val: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct Boris pusher implementation (single particle, 3D)."""
    qdt_2m = q * dt / (2.0 * m_val)

    # Half E-push
    v_minus = v + qdt_2m * E

    # B-rotation
    t_vec = (q / m_val) * B * dt / 2.0
    t_mag_sq = np.dot(t_vec, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # Half E-push
    v_new = v_plus + qdt_2m * E

    # Position update
    x_new = x + v_new * dt

    return x_new, v_new


def b03_boris_correct(num_steps: int = 1000, dt: float = 0.01) -> list[SimulationState]:
    """Correct Boris pusher with B-only (E=0). Energy exactly conserved."""
    x = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    E = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])  # B_z = 1
    q = 1.0
    m_val = 1.0

    states = []
    for t in range(num_steps):
        x, v = boris_push(x, v, E, B, q, m_val, dt)
        states.append(_make_state(
            t, dt,
            positions=np.array([x]),
            velocities=np.array([v]),
            masses=np.array([m_val]),
            charges=np.array([q]),
        ))
    return states


# ===========================================================================
# B04: Boris pusher (dt too large: omega_c * dt > 2)
# ===========================================================================

def b04_boris_large_dt(num_steps: int = 100, dt: float = 0.5) -> list[SimulationState]:
    """Boris pusher with large dt and non-zero E. Energy grows.

    With E=0, Boris preserves |v| exactly (it's volume-preserving).
    But with non-zero E, large dt causes the E-field push to be inaccurate,
    and the energy conservation is only O(dt^2). With a large dt, the
    second-order error becomes visible.
    """
    x = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    E = np.array([1.0, 0.0, 0.0])  # Non-zero E field
    B = np.array([0.0, 0.0, 1.0])
    q = 1.0
    m_val = 1.0

    states = []
    for t in range(num_steps):
        x, v = boris_push(x, v, E, B, q, m_val, dt)
        states.append(_make_state(
            t, dt,
            positions=np.array([x]),
            velocities=np.array([v]),
            masses=np.array([m_val]),
        ))
    return states


# ===========================================================================
# B05: Boris pusher (wrong half-step: full E-push instead of two halves)
# ===========================================================================

def b05_boris_wrong_halfstep(num_steps: int = 500, dt: float = 0.1) -> list[SimulationState]:
    """Boris pusher with wrong half-step structure. O(dt) energy drift."""
    x = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    E = np.array([0.1, 0.0, 0.0])  # Non-zero E to expose the bug
    B = np.array([0.0, 0.0, 1.0])
    q = 1.0
    m_val = 1.0

    states = []
    for t in range(num_steps):
        qdt_m = q * dt / m_val

        # BUG: full E-push instead of two half-pushes
        v_minus = v + qdt_m * E  # Should be qdt_2m * E

        # B-rotation
        t_vec = (q / m_val) * B * dt / 2.0
        t_mag_sq = np.dot(t_vec, t_vec)
        s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)

        # Missing second half E-push
        v = v_plus
        x = x + v * dt

        states.append(_make_state(
            t, dt,
            positions=np.array([x]),
            velocities=np.array([v]),
            masses=np.array([m_val]),
        ))
    return states


# ===========================================================================
# B06: Boris pusher (wrong E-field sign)
# ===========================================================================

def b06_boris_wrong_sign(num_steps: int = 100, dt: float = 0.01) -> list[SimulationState]:
    """Boris pusher with asymmetric E-push error. Breaks energy conservation.

    BUG: The first half E-push uses a doubled step, while the second is correct.
    This asymmetry breaks the time-reversibility of the Boris algorithm and
    causes systematic energy drift.
    """
    x = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    E = np.array([1.0, 0.0, 0.0])  # Non-zero E field
    B = np.array([0.0, 0.0, 1.0])
    q = 1.0
    m_val = 1.0

    states = []
    for t in range(num_steps):
        qdt_2m = q * dt / (2.0 * m_val)

        # BUG: first half-push uses full step instead of half
        v_minus = v + 2.0 * qdt_2m * E  # Should be qdt_2m * E

        t_vec = (q / m_val) * B * dt / 2.0
        t_mag_sq = np.dot(t_vec, t_vec)
        s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)

        # Second half-push is correct
        v = v_plus + qdt_2m * E
        x = x + v * dt

        states.append(_make_state(
            t, dt,
            positions=np.array([x]),
            velocities=np.array([v]),
            masses=np.array([m_val]),
        ))
    return states


# ===========================================================================
# B09: Lorentz force (correct)
# ===========================================================================

def b09_lorentz_correct(num_steps: int = 100, dt: float = 0.01) -> list[SimulationState]:
    """Correct Lorentz force computation."""
    N = 10
    np.random.seed(42)
    v = np.random.randn(N, 3) * 0.5
    E = np.random.randn(N, 3) * 0.1
    B = np.tile([0.0, 0.0, 1.0], (N, 1))
    q = np.ones(N)

    states = []
    for t in range(num_steps):
        F = q[:, np.newaxis] * (E + np.cross(v, B))
        states.append(_make_state(
            t, dt,
            velocities=v.copy(),
            applied_force=F.copy(),
            charges=q.copy(),
            E_at_particles=E.copy(),
            B_at_particles=B.copy(),
            masses=np.ones(N),
        ))
        v = v + F * dt  # Update for next step
    return states


# ===========================================================================
# B10: Lorentz force (wrong sign on v x B)
# ===========================================================================

def b10_lorentz_wrong_sign(num_steps: int = 100, dt: float = 0.01) -> list[SimulationState]:
    """Lorentz force with wrong sign on v x B."""
    N = 10
    np.random.seed(42)
    v = np.random.randn(N, 3) * 0.5
    E = np.random.randn(N, 3) * 0.1
    B = np.tile([0.0, 0.0, 1.0], (N, 1))
    q = np.ones(N)

    states = []
    for t in range(num_steps):
        # BUG: wrong sign on cross product
        F = q[:, np.newaxis] * (E - np.cross(v, B))  # Should be +
        states.append(_make_state(
            t, dt,
            velocities=v.copy(),
            applied_force=F.copy(),
            charges=q.copy(),
            E_at_particles=E.copy(),
            B_at_particles=B.copy(),
            masses=np.ones(N),
        ))
        v = v + F * dt
    return states


# ===========================================================================
# B11: N-body (leapfrog, correct, symmetric forces)
# ===========================================================================

def b11_nbody_correct(num_steps: int = 200, dt: float = 0.001) -> list[SimulationState]:
    """N-body with symmetric forces (Newton's third law). Momentum conserved."""
    N = 4
    np.random.seed(42)
    x = np.random.randn(N, 3) * 2.0
    v = np.random.randn(N, 3) * 0.1
    m = np.ones(N)

    states = []
    for t in range(num_steps):
        # Compute forces (symmetric: F_ij = -F_ji)
        F = np.zeros_like(x)
        for i in range(N):
            for j in range(i + 1, N):
                r = x[j] - x[i]
                dist = np.linalg.norm(r)
                if dist < 0.1:
                    dist = 0.1
                f = r / dist**3
                F[i] += m[j] * f
                F[j] -= m[i] * f  # Newton's third law

        v = v + F * dt
        x = x + v * dt

        states.append(_make_state(
            t, dt,
            positions=x.copy(),
            velocities=v.copy(),
            masses=m.copy(),
        ))
    return states


# ===========================================================================
# B12: N-body (non-symmetric force: F_ij != -F_ji)
# ===========================================================================

def b12_nbody_asymmetric(num_steps: int = 200, dt: float = 0.001) -> list[SimulationState]:
    """N-body with asymmetric forces. Momentum NOT conserved."""
    N = 4
    np.random.seed(42)
    x = np.random.randn(N, 3) * 2.0
    v = np.random.randn(N, 3) * 0.1
    m = np.ones(N)

    states = []
    for t in range(num_steps):
        F = np.zeros_like(x)
        for i in range(N):
            for j in range(i + 1, N):
                r = x[j] - x[i]
                dist = np.linalg.norm(r)
                if dist < 0.1:
                    dist = 0.1
                f = r / dist**3
                F[i] += m[j] * f
                # BUG: F_ji has wrong magnitude (not equal and opposite)
                F[j] -= m[i] * f * 0.9  # 10% error in Newton's third law

        v = v + F * dt
        x = x + v * dt

        states.append(_make_state(
            t, dt,
            positions=x.copy(),
            velocities=v.copy(),
            masses=m.copy(),
        ))
    return states


# ===========================================================================
# B13: Particle boundary (reflecting, correct)
# ===========================================================================

def b13_boundary_reflecting(num_steps: int = 200, dt: float = 0.01) -> list[SimulationState]:
    """Particles in a box with reflecting boundaries. Count conserved."""
    N = 50
    np.random.seed(42)
    x = np.random.rand(N, 3) * 8.0 + 1.0  # Positions in [1, 9]
    v = np.random.randn(N, 3) * 2.0
    m = np.ones(N)
    box_size = 10.0

    states = []
    for t in range(num_steps):
        x = x + v * dt

        # Reflecting boundary
        for d in range(3):
            mask_low = x[:, d] < 0
            mask_high = x[:, d] > box_size
            x[mask_low, d] = -x[mask_low, d]
            v[mask_low, d] = -v[mask_low, d]
            x[mask_high, d] = 2 * box_size - x[mask_high, d]
            v[mask_high, d] = -v[mask_high, d]

        states.append(_make_state(
            t, dt,
            positions=x.copy(),
            velocities=v.copy(),
            masses=m.copy(),
            particle_count=N,
        ))
    return states


# ===========================================================================
# B14: Particle boundary (absorbing, but particles leak through)
# ===========================================================================

def b14_boundary_leak(num_steps: int = 200, dt: float = 0.01) -> list[SimulationState]:
    """Particles in a box with buggy boundary. Particles get duplicated."""
    N = 50
    np.random.seed(42)
    x = np.random.rand(N, 3) * 8.0 + 1.0
    v = np.random.randn(N, 3) * 2.0
    m = np.ones(N)
    box_size = 10.0

    states = []
    for t in range(num_steps):
        x = x + v * dt

        # BUG: particles that hit the boundary get duplicated instead of removed
        for d in range(3):
            mask_low = x[:, d] < 0
            mask_high = x[:, d] > box_size
            if np.any(mask_low) or np.any(mask_high):
                # Bug: duplicate particles near boundaries
                boundary_mask = mask_low | mask_high
                if np.any(boundary_mask):
                    new_x = x[boundary_mask].copy()
                    new_x = np.clip(new_x, 0.1, box_size - 0.1)
                    new_v = v[boundary_mask].copy()
                    new_m = m[boundary_mask].copy()
                    x = np.vstack([x, new_x])
                    v = np.vstack([v, new_v])
                    m = np.concatenate([m, new_m])
                    break  # Only add once per timestep

        states.append(_make_state(
            t, dt,
            positions=x.copy(),
            velocities=v.copy(),
            masses=m.copy(),
            particle_count=len(x),
        ))
    return states
