"""Heuristic-based invariant auto-detection.

Analyses a simulation's source code and/or initial state to determine
which invariant monitors are likely relevant. Uses three strategies:

1. Import analysis: inspect `import` statements for clues about the
   simulation type (numpy, scipy, jax, PIC-specific libraries).
2. Variable name heuristics: look for variable names that suggest
   physical quantities (E, B, v, x, q, rho, etc.).
3. State inspection: check which arrays are present in the initial
   SimulationState and match against invariant applicability.

Usage::

    from sim_debugger.core.auto_detect import auto_detect_invariants

    # From source code
    suggested = auto_detect_invariants(source_code=open("sim.py").read())

    # From a state snapshot
    suggested = auto_detect_invariants(state=initial_state)

    # Both
    suggested = auto_detect_invariants(
        source_code=open("sim.py").read(),
        state=initial_state,
    )
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from sim_debugger.core.state import SimulationState


@dataclass
class InvariantSuggestion:
    """A suggested invariant with the reason it was suggested.

    Attributes:
        name: Name of the invariant (matches registry names).
        confidence: Confidence level from 0.0 (guess) to 1.0 (certain).
        reason: Human-readable reason for the suggestion.
        source: How the suggestion was derived ("import", "variable", "state").
    """

    name: str
    confidence: float
    reason: str
    source: str


def auto_detect_invariants(
    source_code: str | None = None,
    state: SimulationState | None = None,
) -> list[InvariantSuggestion]:
    """Auto-detect which invariants are applicable.

    Combines source-code analysis and state inspection to suggest
    invariant monitors. Results are sorted by confidence (highest first).

    Args:
        source_code: Python source code of the simulation.
        state: An initial or representative simulation state.

    Returns:
        List of InvariantSuggestion sorted by confidence.
    """
    suggestions: list[InvariantSuggestion] = []

    if source_code is not None:
        suggestions.extend(_analyse_imports(source_code))
        suggestions.extend(_analyse_variables(source_code))
        suggestions.extend(_analyse_patterns(source_code))

    if state is not None:
        suggestions.extend(_analyse_state(state))

    # Deduplicate: keep the highest-confidence suggestion for each invariant
    best: dict[str, InvariantSuggestion] = {}
    for s in suggestions:
        if s.name not in best or s.confidence > best[s.name].confidence:
            best[s.name] = s

    # Sort by confidence descending
    return sorted(best.values(), key=lambda s: s.confidence, reverse=True)


def _analyse_imports(source: str) -> list[InvariantSuggestion]:
    """Analyse import statements for clues about simulation type."""
    suggestions: list[InvariantSuggestion] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return suggestions

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    # NumPy/SciPy -> likely numerical simulation
    if any(m.startswith("numpy") or m.startswith("np") for m in imported_modules):
        suggestions.append(InvariantSuggestion(
            name="Total Energy",
            confidence=0.3,
            reason="NumPy imported; numerical simulation likely involves energy",
            source="import",
        ))

    # SciPy ODE solvers -> Hamiltonian system
    if any("scipy.integrate" in m or "solve_ivp" in m or "odeint" in m
           for m in imported_modules):
        suggestions.append(InvariantSuggestion(
            name="Total Energy",
            confidence=0.6,
            reason="SciPy ODE solver detected; likely Hamiltonian system",
            source="import",
        ))

    # PIC-specific libraries
    pic_modules = {"pybpic", "wpbody", "picsar", "warpx", "fbpic", "picongpu"}
    if imported_modules & pic_modules:
        suggestions.append(InvariantSuggestion(
            name="Boris Energy",
            confidence=0.8,
            reason="PIC library imported; Boris pusher energy monitoring recommended",
            source="import",
        ))
        suggestions.append(InvariantSuggestion(
            name="Gauss's Law",
            confidence=0.8,
            reason="PIC library imported; Gauss's law monitoring recommended",
            source="import",
        ))
        suggestions.append(InvariantSuggestion(
            name="Charge Conservation",
            confidence=0.7,
            reason="PIC library imported; charge conservation monitoring recommended",
            source="import",
        ))

    return suggestions


def _analyse_variables(source: str) -> list[InvariantSuggestion]:
    """Analyse variable names for physics quantity clues."""
    suggestions: list[InvariantSuggestion] = []

    # Variable name patterns and their associated invariants
    _PATTERNS: list[tuple[str, str, float, str]] = [
        # (regex_pattern, invariant_name, confidence, reason)
        (r"\bvelocit(?:y|ies)\b", "Total Energy", 0.7,
         "velocity arrays detected; energy conservation applicable"),
        (r"\bvelocit(?:y|ies)\b", "Linear Momentum", 0.6,
         "velocity arrays detected; momentum conservation applicable"),
        (r"\bposition(?:s)?\b", "Angular Momentum", 0.4,
         "position arrays detected; angular momentum may be relevant"),
        (r"\b(?:charge|charges)\b", "Charge Conservation", 0.7,
         "charge variables detected"),
        (r"\bparticle(?:s|_count|_num)\b", "Particle Count", 0.5,
         "particle references detected"),
        (r"\bboris\b", "Boris Energy", 0.9,
         "Boris pusher explicitly referenced"),
        (r"\b(?:E_field|electric_field)\b", "Total Energy", 0.6,
         "electric field detected; electromagnetic energy applicable"),
        (r"\b(?:B_field|magnetic_field)\b", "Boris Energy", 0.7,
         "magnetic field detected; Boris energy monitoring recommended"),
        (r"\b(?:div|divergence|gauss)\b", "Gauss's Law", 0.5,
         "divergence or Gauss's law references detected"),
        (r"\blorentz\b", "Lorentz Force", 0.8,
         "Lorentz force explicitly referenced"),
        (r"\bracetrack\b", "Racetrack Symmetry", 0.8,
         "racetrack coil geometry referenced"),
        (r"\b(?:leapfrog|verlet|symplectic)\b", "Total Energy", 0.6,
         "symplectic integrator detected; energy conservation applicable"),
        (r"\b(?:euler|rk4|runge.kutta)\b", "Total Energy", 0.7,
         "non-symplectic integrator detected; energy monitoring important"),
    ]

    source_lower = source.lower()
    for pattern, inv_name, confidence, reason in _PATTERNS:
        if re.search(pattern, source_lower):
            suggestions.append(InvariantSuggestion(
                name=inv_name,
                confidence=confidence,
                reason=reason,
                source="variable",
            ))

    return suggestions


def _analyse_patterns(source: str) -> list[InvariantSuggestion]:
    """Analyse code patterns for specific simulation structures."""
    suggestions: list[InvariantSuggestion] = []

    # Boris pusher three-step pattern
    has_half_push = bool(re.search(
        r"(?:q|charge)\s*\*\s*(?:dt|delta_t)\s*/\s*\(?\s*2",
        source, re.IGNORECASE,
    ))
    has_cross = bool(re.search(r"cross\s*\(", source))
    if has_half_push and has_cross:
        suggestions.append(InvariantSuggestion(
            name="Boris Energy",
            confidence=0.9,
            reason="Boris pusher half-step and cross product pattern detected",
            source="pattern",
        ))

    # N-body force computation
    has_pairwise = bool(re.search(
        r"for\s+\w+\s+in\s+range.*?for\s+\w+\s+in\s+range",
        source, re.DOTALL,
    ))
    if has_pairwise and re.search(r"\bforce\b|\bF\b", source):
        suggestions.append(InvariantSuggestion(
            name="Linear Momentum",
            confidence=0.7,
            reason="pairwise force computation pattern detected (N-body)",
            source="pattern",
        ))

    # PDE/grid patterns
    has_grid = bool(re.search(
        r"\b(?:grid|mesh|cell|nx|ny|nz)\b", source, re.IGNORECASE
    ))
    if has_grid and re.search(r"\b(?:E_field|E|rho)\b", source):
        suggestions.append(InvariantSuggestion(
            name="Gauss's Law",
            confidence=0.6,
            reason="grid-based field computation detected",
            source="pattern",
        ))

    return suggestions


def _analyse_state(state: SimulationState) -> list[InvariantSuggestion]:
    """Analyse a simulation state snapshot for applicable invariants."""
    suggestions: list[InvariantSuggestion] = []

    has_vel = state.has_array("velocities")
    has_mass = state.has_array("masses")
    has_pos = state.has_array("positions")
    has_charge = state.has_array("charges")
    has_E = state.has_array("E_field")
    _has_B = state.has_array("B_field")  # noqa: F841, N806 -- reserved for future use
    has_rho = state.has_array("charge_density")
    has_force = state.has_array("applied_force")

    if has_vel and has_mass:
        suggestions.append(InvariantSuggestion(
            name="Total Energy",
            confidence=0.9,
            reason="velocities and masses arrays present in state",
            source="state",
        ))
        suggestions.append(InvariantSuggestion(
            name="Linear Momentum",
            confidence=0.8,
            reason="velocities and masses arrays present in state",
            source="state",
        ))

    if has_pos and has_vel and has_mass:
        pos = state.get_array("positions")
        if pos.ndim >= 2 and pos.shape[1] >= 2:
            suggestions.append(InvariantSuggestion(
                name="Angular Momentum",
                confidence=0.7,
                reason="positions, velocities, masses present with >= 2D positions",
                source="state",
            ))

    if has_charge:
        suggestions.append(InvariantSuggestion(
            name="Charge Conservation",
            confidence=0.9,
            reason="charges array present in state",
            source="state",
        ))

    if has_vel and has_mass and "dt" in state.metadata:
        suggestions.append(InvariantSuggestion(
            name="Boris Energy",
            confidence=0.7,
            reason="velocities, masses, and dt present; Boris monitoring applicable",
            source="state",
        ))

    if has_E and has_rho:
        suggestions.append(InvariantSuggestion(
            name="Gauss's Law",
            confidence=0.9,
            reason="E_field and charge_density arrays present",
            source="state",
        ))

    if has_force and has_vel and has_charge:
        if state.has_array("E_at_particles") and state.has_array("B_at_particles"):
            suggestions.append(InvariantSuggestion(
                name="Lorentz Force",
                confidence=0.95,
                reason="applied_force, velocities, charges, E_at_particles, "
                       "B_at_particles all present",
                source="state",
            ))

    # Particle count: applicable if we have positions or velocities
    if has_pos or has_vel:
        suggestions.append(InvariantSuggestion(
            name="Particle Count",
            confidence=0.5,
            reason="particle arrays present in state",
            source="state",
        ))

    return suggestions
