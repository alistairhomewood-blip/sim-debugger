"""Tests for spatial localisation.

Tests per-particle and per-cell contribution analysis, top-N contributor
identification, bounding box computation, and the main localise_spatial
dispatch function.
"""

import numpy as np
import pytest

from sim_debugger.core.state import SimulationState
from sim_debugger.localise.spatial import (
    compute_bounding_box,
    compute_contribution_changes,
    compute_field_energy_contributions,
    compute_kinetic_energy_contributions,
    compute_momentum_contributions,
    find_top_contributors,
    localise_spatial,
)

# ===========================================================================
# Helpers
# ===========================================================================

def make_state(
    timestep: int = 0,
    positions: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
    masses: np.ndarray | None = None,
    charges: np.ndarray | None = None,
    **kwargs,
) -> SimulationState:
    """Helper to create a SimulationState for testing."""
    arrays = {}
    metadata = {}
    if positions is not None:
        arrays["positions"] = positions
    if velocities is not None:
        arrays["velocities"] = velocities
    if masses is not None:
        arrays["masses"] = masses
    if charges is not None:
        arrays["charges"] = charges
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            metadata[k] = v
    return SimulationState(
        timestep=timestep, time=float(timestep),
        arrays=arrays, metadata=metadata,
    )


# ===========================================================================
# Per-particle kinetic energy contributions
# ===========================================================================

class TestKineticEnergyContributions:
    def test_single_particle(self):
        """KE of a single particle: 0.5 * m * |v|^2."""
        state = make_state(
            velocities=np.array([[3.0, 4.0, 0.0]]),
            masses=np.array([2.0]),
        )
        ke = compute_kinetic_energy_contributions(state)
        # 0.5 * 2.0 * (9 + 16) = 25.0
        assert ke[0] == pytest.approx(25.0)

    def test_multiple_particles(self):
        """KE of multiple particles computed independently."""
        state = make_state(
            velocities=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            masses=np.array([1.0, 3.0]),
        )
        ke = compute_kinetic_energy_contributions(state)
        assert len(ke) == 2
        assert ke[0] == pytest.approx(0.5)    # 0.5 * 1 * 1
        assert ke[1] == pytest.approx(6.0)    # 0.5 * 3 * 4

    def test_scalar_mass(self):
        """KE with uniform (scalar) mass."""
        state = make_state(
            velocities=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            masses=np.float64(2.0),
        )
        ke = compute_kinetic_energy_contributions(state)
        assert ke[0] == pytest.approx(1.0)
        assert ke[1] == pytest.approx(1.0)

    def test_1d_velocities(self):
        """KE with 1D velocity array."""
        state = make_state(
            velocities=np.array([3.0, 4.0]),
            masses=np.array([1.0, 1.0]),
        )
        ke = compute_kinetic_energy_contributions(state)
        assert ke[0] == pytest.approx(4.5)    # 0.5 * 1 * 9
        assert ke[1] == pytest.approx(8.0)    # 0.5 * 1 * 16

    def test_missing_arrays_raises(self):
        """Missing velocities or masses raises KeyError."""
        state = make_state(velocities=np.array([[1.0]]))
        with pytest.raises(KeyError):
            compute_kinetic_energy_contributions(state)


# ===========================================================================
# Per-particle momentum contributions
# ===========================================================================

class TestMomentumContributions:
    def test_single_particle_3d(self):
        """Momentum vector: m * v (returns vectors, not magnitudes)."""
        state = make_state(
            velocities=np.array([[3.0, 4.0, 0.0]]),
            masses=np.array([2.0]),
        )
        p = compute_momentum_contributions(state)
        # 2 * (3, 4, 0) = (6, 8, 0); magnitude = 10
        np.testing.assert_allclose(p[0], [6.0, 8.0, 0.0])
        assert np.linalg.norm(p[0]) == pytest.approx(10.0)

    def test_multiple_particles(self):
        """Per-particle momentum vectors for multiple particles."""
        state = make_state(
            velocities=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            masses=np.array([1.0, 3.0]),
        )
        p = compute_momentum_contributions(state)
        np.testing.assert_allclose(p[0], [1.0, 0.0, 0.0])  # 1 * (1,0,0)
        np.testing.assert_allclose(p[1], [0.0, 6.0, 0.0])  # 3 * (0,2,0)

    def test_1d_velocities(self):
        """Momentum with 1D velocity array (returns signed scalars)."""
        state = make_state(
            velocities=np.array([3.0, -4.0]),
            masses=np.array([2.0, 1.0]),
        )
        p = compute_momentum_contributions(state)
        assert p[0] == pytest.approx(6.0)    # 2 * 3 = 6
        assert p[1] == pytest.approx(-4.0)   # 1 * -4 = -4


# ===========================================================================
# Per-cell field energy contributions
# ===========================================================================

class TestFieldEnergyContributions:
    def test_uniform_field(self):
        """Uniform field has uniform energy density."""
        E = np.ones((10, 3)) * 2.0  # 10 cells, 3D
        state = make_state(E_field=E, eps_0=1.0)
        energy = compute_field_energy_contributions(state)
        # 0.5 * 1.0 * (4 + 4 + 4) = 6.0 per cell
        assert energy.shape == (10,)
        np.testing.assert_array_almost_equal(energy, 6.0)

    def test_single_hot_cell(self):
        """One cell has a strong field; others are zero."""
        E = np.zeros((5, 3))
        E[2] = [10.0, 0.0, 0.0]
        state = make_state(E_field=E, eps_0=1.0)
        energy = compute_field_energy_contributions(state)
        assert energy[2] == pytest.approx(50.0)  # 0.5 * 1.0 * 100
        assert energy[0] == pytest.approx(0.0)
        assert energy[4] == pytest.approx(0.0)


# ===========================================================================
# Contribution change analysis
# ===========================================================================

class TestContributionChanges:
    def test_equal_contributions(self):
        """Zero change when contributions are equal."""
        prev = np.array([1.0, 2.0, 3.0])
        curr = np.array([1.0, 2.0, 3.0])
        changes = compute_contribution_changes(prev, curr)
        np.testing.assert_array_almost_equal(changes, 0.0)

    def test_positive_change(self):
        """Positive change when current exceeds previous."""
        prev = np.array([1.0, 2.0, 3.0])
        curr = np.array([2.0, 3.0, 5.0])
        changes = compute_contribution_changes(prev, curr)
        np.testing.assert_array_equal(changes, [1.0, 1.0, 2.0])

    def test_mixed_changes(self):
        """Mixed positive and negative changes."""
        prev = np.array([5.0, 2.0, 3.0])
        curr = np.array([3.0, 4.0, 3.0])
        changes = compute_contribution_changes(prev, curr)
        np.testing.assert_array_equal(changes, [-2.0, 2.0, 0.0])

    def test_different_lengths(self):
        """Handles arrays of different lengths (particle count change)."""
        prev = np.array([1.0, 2.0, 3.0])
        curr = np.array([1.0, 5.0])
        changes = compute_contribution_changes(prev, curr)
        # Only compares the overlapping portion
        assert len(changes) == 2
        np.testing.assert_array_equal(changes, [0.0, 3.0])


# ===========================================================================
# Top contributors
# ===========================================================================

class TestFindTopContributors:
    def test_single_dominant_contributor(self):
        """Identifies the single largest contributor."""
        changes = np.array([0.1, 0.2, 10.0, 0.3, 0.1])
        indices, top = find_top_contributors(changes, top_n=1)
        assert top[0][0] == 2  # Index 2 has the largest change
        assert top[0][1] == pytest.approx(10.0)

    def test_top_n(self):
        """Returns correct number of top contributors."""
        changes = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        indices, top = find_top_contributors(changes, top_n=3)
        assert len(top) == 3
        # Top 3: index 1 (5.0), index 4 (4.0), index 2 (3.0)
        top_indices = [t[0] for t in top]
        assert 1 in top_indices
        assert 4 in top_indices
        assert 2 in top_indices

    def test_negative_changes(self):
        """Negative changes are handled by absolute value."""
        changes = np.array([0.1, -10.0, 0.5])
        indices, top = find_top_contributors(changes, top_n=1)
        assert top[0][0] == 1  # -10.0 has the largest |change|
        assert top[0][1] == pytest.approx(-10.0)  # Original signed value

    def test_top_n_exceeds_length(self):
        """top_n larger than array length returns all elements."""
        changes = np.array([1.0, 2.0])
        indices, top = find_top_contributors(changes, top_n=10)
        assert len(top) == 2

    def test_sorted_indices(self):
        """Returned indices are sorted by descending |change|."""
        changes = np.array([1.0, 5.0, 3.0])
        indices, _ = find_top_contributors(changes, top_n=3)
        # indices[0] should be the index of the largest |change|
        assert indices[0] == 1
        assert indices[1] == 2
        assert indices[2] == 0


# ===========================================================================
# Bounding box computation
# ===========================================================================

class TestBoundingBox:
    def test_2d_particles(self):
        """Bounding box of 2D particles."""
        state = make_state(
            positions=np.array([
                [1.0, 2.0],
                [3.0, 5.0],
                [0.5, 1.0],
                [4.0, 3.0],
            ]),
        )
        indices = np.array([0, 1, 2])
        bbox = compute_bounding_box(state, indices)
        assert bbox is not None
        # xmin=0.5, xmax=3.0, ymin=1.0, ymax=5.0
        assert bbox == pytest.approx((0.5, 3.0, 1.0, 5.0))

    def test_3d_particles(self):
        """Bounding box of 3D particles."""
        state = make_state(
            positions=np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
        )
        indices = np.array([0, 1])
        bbox = compute_bounding_box(state, indices)
        assert bbox is not None
        assert len(bbox) == 6  # xmin, xmax, ymin, ymax, zmin, zmax
        assert bbox == pytest.approx((1.0, 4.0, 2.0, 5.0, 3.0, 6.0))

    def test_no_positions(self):
        """Returns None when positions are not available."""
        state = make_state(velocities=np.array([[1.0, 0.0, 0.0]]))
        bbox = compute_bounding_box(state, np.array([0]))
        assert bbox is None

    def test_empty_indices(self):
        """Returns None for empty index array."""
        state = make_state(
            positions=np.array([[1.0, 2.0]]),
        )
        bbox = compute_bounding_box(state, np.array([], dtype=int))
        assert bbox is None

    def test_out_of_range_indices(self):
        """Handles indices beyond array length."""
        state = make_state(
            positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        indices = np.array([0, 1, 5, 10])  # 5 and 10 are out of range
        bbox = compute_bounding_box(state, indices)
        assert bbox is not None
        # Only indices 0 and 1 are valid
        assert bbox == pytest.approx((1.0, 3.0, 2.0, 4.0))


# ===========================================================================
# Main localise_spatial function
# ===========================================================================

class TestLocaliseSpatial:
    def test_energy_localisation(self):
        """Localise energy violation to the particle that gained energy."""
        prev = make_state(
            timestep=0,
            velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0, 1.0]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )
        curr = make_state(
            timestep=1,
            # Particle 1 (middle) has a huge velocity jump
            velocities=np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0, 1.0]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )

        result = localise_spatial("Total Energy", prev, curr, top_n=1)
        assert result is not None
        assert result.region_type == "particles"
        # The top contributor should be particle 1
        assert 1 in result.indices

    def test_momentum_localisation(self):
        """Localise momentum violation to the contributing particle."""
        prev = make_state(
            timestep=0,
            velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        )
        curr = make_state(
            timestep=1,
            velocities=np.array([[1.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        )

        result = localise_spatial("Linear Momentum", prev, curr, top_n=1)
        assert result is not None
        assert result.region_type == "particles"
        assert 1 in result.indices

    def test_boris_energy_localisation(self):
        """Boris Energy localisation uses kinetic energy contributions."""
        prev = make_state(
            timestep=0,
            velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
        )
        curr = make_state(
            timestep=1,
            velocities=np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
        )

        result = localise_spatial("Boris Energy", prev, curr, top_n=1)
        assert result is not None
        assert 1 in result.indices

    def test_charge_localisation(self):
        """Localise charge violation to the particle with changed charge."""
        prev = make_state(
            timestep=0,
            charges=np.array([1.0, -1.0, 1.0]),
        )
        curr = make_state(
            timestep=1,
            charges=np.array([1.0, -1.0, 5.0]),  # Particle 2 changed
        )

        result = localise_spatial("Charge Conservation", prev, curr, top_n=1)
        assert result is not None
        assert result.region_type == "particles"
        assert 2 in result.indices

    def test_field_localisation(self):
        """Localise Gauss's law violation to the cell with changed field."""
        E_prev = np.zeros((5, 3))
        E_curr = np.zeros((5, 3))
        E_curr[3] = [100.0, 0.0, 0.0]  # Cell 3 has a big field change

        prev = make_state(timestep=0, E_field=E_prev, eps_0=1.0)
        curr = make_state(timestep=1, E_field=E_curr, eps_0=1.0)

        result = localise_spatial("Gauss's Law", prev, curr, top_n=1)
        assert result is not None
        assert result.region_type == "cells"
        assert 3 in result.indices

    def test_unsupported_invariant(self):
        """Unknown invariant returns None."""
        state = make_state(timestep=0)
        result = localise_spatial("Unknown Invariant", state, state)
        assert result is None

    def test_missing_arrays_returns_none(self):
        """Missing required arrays returns None instead of raising."""
        prev = make_state(timestep=0)  # No arrays at all
        curr = make_state(timestep=1)
        result = localise_spatial("Total Energy", prev, curr)
        assert result is None

    def test_localisation_includes_bounding_box(self):
        """Energy localisation includes a bounding box when positions exist."""
        prev = make_state(
            timestep=0,
            velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            positions=np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 3.0]]),
        )
        curr = make_state(
            timestep=1,
            velocities=np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            positions=np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 3.0]]),
        )

        result = localise_spatial("Total Energy", prev, curr, top_n=1)
        assert result is not None
        assert result.bounding_box is not None

    def test_top_n_parameter(self):
        """top_n controls how many contributors are returned."""
        n_particles = 20
        prev = make_state(
            timestep=0,
            velocities=np.ones((n_particles, 3)),
            masses=np.ones(n_particles),
        )
        # Make particles 5, 10, 15 have big velocity changes
        v_curr = np.ones((n_particles, 3))
        v_curr[5] = [50.0, 0.0, 0.0]
        v_curr[10] = [40.0, 0.0, 0.0]
        v_curr[15] = [30.0, 0.0, 0.0]

        curr = make_state(
            timestep=1,
            velocities=v_curr,
            masses=np.ones(n_particles),
        )

        result = localise_spatial("Total Energy", prev, curr, top_n=3)
        assert result is not None
        assert len(result.indices) == 3
        # The top 3 should be particles 5, 10, 15
        top_set = set(result.indices.tolist())
        assert 5 in top_set
        assert 10 in top_set
        assert 15 in top_set
