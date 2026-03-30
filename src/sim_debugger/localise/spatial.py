"""Spatial localisation: identify where a violation is concentrated.

Given a conservation law violation, decompose the global invariant into
per-cell or per-particle contributions and identify the spatial region
that contributes most to the violation.

Key capabilities:
    - Per-particle kinetic energy contribution analysis
    - Per-particle momentum contribution analysis
    - Per-cell field energy contribution analysis
    - Top-N contributing particles/cells identification
    - Bounding box computation for anomalous regions

Usage::

    from sim_debugger.localise.spatial import localise_spatial

    result = localise_spatial(
        invariant_name="Total Energy",
        prev_state=state_t_minus_1,
        curr_state=state_t,
        top_n=10,
    )
    # result.indices = indices of top contributing particles
    # result.bounding_box = (xmin, xmax, ymin, ymax, ...)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import SpatialLocalisation

# ---------------------------------------------------------------------------
# Per-particle contribution analysis
# ---------------------------------------------------------------------------

@dataclass
class ContributionResult:
    """Result of a per-particle/per-cell contribution analysis.

    Attributes:
        contributions: Per-element contribution to the invariant change.
        indices: Indices sorted by decreasing |contribution|.
        top_contributions: The top-N contributions (element_index, value).
        bounding_box: Spatial bounding box of the top contributors,
                      or None if positions are not available.
    """

    contributions: np.ndarray
    indices: np.ndarray
    top_contributions: list[tuple[int, float]]
    bounding_box: tuple[float, ...] | None


def compute_kinetic_energy_contributions(
    state: SimulationState,
) -> np.ndarray:
    """Compute per-particle kinetic energy: 0.5 * m_i * |v_i|^2.

    Args:
        state: Simulation state with velocities and masses arrays.

    Returns:
        (N,) array of per-particle kinetic energies.

    Raises:
        KeyError: If velocities or masses are not in the state.
    """
    v = state.get_array("velocities")
    m = state.get_array("masses")

    if v.ndim == 1:
        v_sq = v * v
    else:
        v_sq = np.sum(v * v, axis=-1)

    if np.isscalar(m) or m.ndim == 0:
        return 0.5 * float(m) * v_sq  # type: ignore[no-any-return]
    return 0.5 * m * v_sq  # type: ignore[no-any-return]


def compute_momentum_contributions(
    state: SimulationState,
) -> np.ndarray:
    """Compute per-particle momentum vectors: m_i * v_i.

    Returns the full momentum vector per particle so that spatial
    localisation can detect directional changes (not just magnitude).
    Use np.linalg.norm(result, axis=-1) if you need magnitudes.

    Args:
        state: Simulation state with velocities and masses arrays.

    Returns:
        (N,) array for 1D or (N, D) array of per-particle momentum vectors.
    """
    v = state.get_array("velocities")
    m = state.get_array("masses")

    if np.isscalar(m) or m.ndim == 0:
        return float(m) * v  # type: ignore[no-any-return]
    else:
        if v.ndim == 1:
            return m * v  # type: ignore[no-any-return]
        else:
            return m[:, np.newaxis] * v  # type: ignore[no-any-return]


def compute_field_energy_contributions(
    state: SimulationState,
    field_name: str = "E_field",
) -> np.ndarray:
    """Compute per-cell field energy: 0.5 * eps_0 * |E|^2 * cell_volume.

    Returns actual energy per cell (not density), consistent with
    TotalEnergyInvariant.compute().

    Args:
        state: Simulation state with the named field array.
        field_name: Name of the field array (default "E_field").

    Returns:
        Array of per-cell field energies (same shape as grid,
        without the vector component dimension).
    """
    field = state.get_array(field_name)
    eps_0 = state.metadata.get("eps_0", 8.854187817e-12)
    dx = state.metadata.get("dx", 1.0)

    # Cell volume: dx^ndim_spatial (uniform grid assumed)
    ndim_spatial = field.ndim - 1
    if isinstance(dx, (int, float)):
        cell_volume = dx ** ndim_spatial
    else:
        cell_volume = float(np.prod(dx))

    # Per-cell field energy (not density)
    energy_per_cell = 0.5 * eps_0 * np.sum(field * field, axis=-1) * cell_volume
    return energy_per_cell  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Contribution change analysis
# ---------------------------------------------------------------------------

def compute_contribution_changes(
    prev_contributions: np.ndarray,
    curr_contributions: np.ndarray,
) -> np.ndarray:
    """Compute the change in per-element contributions between timesteps.

    For vector contributions (e.g., momentum), returns the norm of the
    per-element vector difference: ||p_new - p_old|| per particle.
    For scalar contributions (e.g., energy), returns curr - prev.

    Args:
        prev_contributions: Per-element contributions at previous timestep.
        curr_contributions: Per-element contributions at current timestep.

    Returns:
        (N,) array of scalar contribution changes.
    """
    if prev_contributions.shape != curr_contributions.shape:
        min_len = min(len(prev_contributions), len(curr_contributions))
        prev_trimmed = prev_contributions[:min_len]
        curr_trimmed = curr_contributions[:min_len]
        diff = curr_trimmed - prev_trimmed
    else:
        diff = curr_contributions - prev_contributions

    # For vector contributions, return norm of vector difference
    if diff.ndim > 1:
        return np.linalg.norm(diff, axis=-1)  # type: ignore[no-any-return]
    return diff  # type: ignore[no-any-return]


def find_top_contributors(
    changes: np.ndarray,
    top_n: int = 10,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Find the elements with the largest absolute contribution change.

    Args:
        changes: Per-element contribution changes.
        top_n: Number of top contributors to return.

    Returns:
        Tuple of (sorted_indices, top_contributions) where:
        - sorted_indices: all indices sorted by |change| descending
        - top_contributions: list of (index, change_value) for top N
    """
    abs_changes = np.abs(changes.ravel())
    sorted_indices = np.argsort(abs_changes)[::-1]

    top_n = min(top_n, len(sorted_indices))
    top_contributions = [
        (int(sorted_indices[i]), float(changes.ravel()[sorted_indices[i]]))
        for i in range(top_n)
    ]

    return sorted_indices, top_contributions


def compute_bounding_box(
    state: SimulationState,
    indices: np.ndarray,
) -> tuple[float, ...] | None:
    """Compute the spatial bounding box of the specified particles.

    Args:
        state: Simulation state with positions array.
        indices: Indices of the particles to include.

    Returns:
        Tuple of (xmin, xmax, ymin, ymax, ...) for each spatial dimension,
        or None if positions are not available.
    """
    if not state.has_array("positions"):
        return None

    positions = state.get_array("positions")
    if len(indices) == 0:
        return None

    # Clamp indices to valid range
    valid_indices = indices[indices < len(positions)]
    if len(valid_indices) == 0:
        return None

    selected = positions[valid_indices]

    if selected.ndim == 1:
        return (float(np.min(selected)), float(np.max(selected)))

    bounds: list[float] = []
    for d in range(selected.shape[1]):
        bounds.append(float(np.min(selected[:, d])))
        bounds.append(float(np.max(selected[:, d])))
    return tuple(bounds)


# ---------------------------------------------------------------------------
# Main spatial localisation function
# ---------------------------------------------------------------------------

def localise_spatial(
    invariant_name: str,
    prev_state: SimulationState,
    curr_state: SimulationState,
    top_n: int = 10,
) -> SpatialLocalisation | None:
    """Localise a violation to a spatial region.

    Decomposes the global invariant into per-particle or per-cell
    contributions, computes the change between timesteps, and identifies
    the spatial elements contributing most to the violation.

    Supported invariants:
        - "Total Energy", "Boris Energy": per-particle kinetic energy
        - "Linear Momentum": per-particle momentum
        - "Angular Momentum": per-particle angular momentum (via momentum proxy)
        - Field-based invariants: per-cell field energy

    Args:
        invariant_name: Name of the violated invariant.
        prev_state: State at the previous timestep.
        curr_state: State at the current timestep.
        top_n: Number of top contributors to identify.

    Returns:
        A SpatialLocalisation result, or None if localisation is not
        possible for this invariant type or state.
    """
    try:
        result = _localise_by_invariant(invariant_name, prev_state, curr_state, top_n)
        return result
    except (KeyError, ValueError):
        # Cannot localise: required arrays missing or incompatible
        return None


def _localise_by_invariant(
    invariant_name: str,
    prev_state: SimulationState,
    curr_state: SimulationState,
    top_n: int,
) -> SpatialLocalisation | None:
    """Dispatch spatial localisation based on invariant type."""

    # Energy-based invariants: per-particle kinetic energy
    if invariant_name in ("Total Energy", "Boris Energy"):
        return _localise_energy(prev_state, curr_state, top_n)

    # Momentum-based invariants
    if invariant_name in ("Linear Momentum", "Angular Momentum"):
        return _localise_momentum(prev_state, curr_state, top_n)

    # Charge-based invariants
    if invariant_name == "Charge Conservation":
        return _localise_charge(prev_state, curr_state, top_n)

    # Field-based invariants
    if invariant_name in ("Gauss's Law",):
        return _localise_field(prev_state, curr_state, top_n)

    return None


def _localise_energy(
    prev_state: SimulationState,
    curr_state: SimulationState,
    top_n: int,
) -> SpatialLocalisation | None:
    """Localise an energy violation via per-particle kinetic energy."""
    if not (curr_state.has_array("velocities") and curr_state.has_array("masses")):
        return None
    if not (prev_state.has_array("velocities") and prev_state.has_array("masses")):
        return None

    prev_ke = compute_kinetic_energy_contributions(prev_state)
    curr_ke = compute_kinetic_energy_contributions(curr_state)

    changes = compute_contribution_changes(prev_ke, curr_ke)
    sorted_indices, top_contribs = find_top_contributors(changes, top_n)

    bbox = compute_bounding_box(curr_state, sorted_indices[:top_n])

    return SpatialLocalisation(
        region_type="particles",
        indices=sorted_indices[:top_n].copy(),
        bounding_box=bbox,
    )


def _localise_momentum(
    prev_state: SimulationState,
    curr_state: SimulationState,
    top_n: int,
) -> SpatialLocalisation | None:
    """Localise a momentum violation via per-particle momentum."""
    if not (curr_state.has_array("velocities") and curr_state.has_array("masses")):
        return None
    if not (prev_state.has_array("velocities") and prev_state.has_array("masses")):
        return None

    prev_p = compute_momentum_contributions(prev_state)
    curr_p = compute_momentum_contributions(curr_state)

    changes = compute_contribution_changes(prev_p, curr_p)
    sorted_indices, top_contribs = find_top_contributors(changes, top_n)

    bbox = compute_bounding_box(curr_state, sorted_indices[:top_n])

    return SpatialLocalisation(
        region_type="particles",
        indices=sorted_indices[:top_n].copy(),
        bounding_box=bbox,
    )


def _localise_charge(
    prev_state: SimulationState,
    curr_state: SimulationState,
    top_n: int,
) -> SpatialLocalisation | None:
    """Localise a charge violation.

    If per-particle charges are available, find particles whose charge
    changed. If grid-based charge density is available, find cells with
    the largest residual.
    """
    if curr_state.has_array("charges") and prev_state.has_array("charges"):
        prev_q = prev_state.get_array("charges")
        curr_q = curr_state.get_array("charges")
        changes = compute_contribution_changes(prev_q, curr_q)
        sorted_indices, top_contribs = find_top_contributors(changes, top_n)
        bbox = compute_bounding_box(curr_state, sorted_indices[:top_n])
        return SpatialLocalisation(
            region_type="particles",
            indices=sorted_indices[:top_n].copy(),
            bounding_box=bbox,
        )

    if curr_state.has_array("charge_density") and prev_state.has_array("charge_density"):
        prev_rho = prev_state.get_array("charge_density")
        curr_rho = curr_state.get_array("charge_density")
        changes = compute_contribution_changes(prev_rho, curr_rho)
        sorted_indices, top_contribs = find_top_contributors(changes, top_n)
        return SpatialLocalisation(
            region_type="cells",
            indices=sorted_indices[:top_n].copy(),
            bounding_box=None,
        )

    return None


def _localise_field(
    prev_state: SimulationState,
    curr_state: SimulationState,
    top_n: int,
) -> SpatialLocalisation | None:
    """Localise a field-based violation via per-cell field energy."""
    if not curr_state.has_array("E_field"):
        return None
    if not prev_state.has_array("E_field"):
        return None

    prev_energy = compute_field_energy_contributions(prev_state)
    curr_energy = compute_field_energy_contributions(curr_state)

    changes = compute_contribution_changes(prev_energy, curr_energy)
    sorted_indices, top_contribs = find_top_contributors(changes, top_n)

    return SpatialLocalisation(
        region_type="cells",
        indices=sorted_indices[:top_n].copy(),
        bounding_box=None,
    )
