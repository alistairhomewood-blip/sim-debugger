"""Explanation templates for physics-language violation reports.

Each template is a dataclass with slots for the four parts of an explanation:
1. What happened (numeric description)
2. Where it happened (localisation)
3. Why it likely happened (diagnosis)
4. Suggested fix

Templates are organised by (invariant_type, violation_pattern) pairs.
The explanation generator selects the appropriate template and fills
in the slots from the violation data.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim_debugger.core.violations import ViolationPattern


@dataclass(frozen=True)
class ExplanationTemplate:
    """A template for generating a physics-language explanation.

    Attributes:
        invariant_type: The invariant this template applies to.
        pattern: The violation pattern this template describes.
        what: Template for what happened (uses Python format strings).
        where: Template for where it happened.
        why: Template for the diagnosis.
        fix: Template for the suggested fix.
    """

    invariant_type: str
    pattern: ViolationPattern
    what: str
    where: str
    why: str
    fix: str


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

# Key: (invariant_name, ViolationPattern) -> ExplanationTemplate
TEMPLATES: dict[tuple[str, ViolationPattern], ExplanationTemplate] = {}


def _register(t: ExplanationTemplate) -> None:
    TEMPLATES[(t.invariant_type, t.pattern)] = t


# ==========================================================================
# Total Energy templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Total Energy",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Total energy changed by {relative_error:.2%} at timestep {timestep} "
        "(from {prev_value:.6e} to {curr_value:.6e})."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why=(
        "A sudden single-timestep energy change is consistent with "
        "{diagnosis}."
    ),
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Total Energy",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Total energy drifted by {relative_error:.2%} over {duration} timesteps "
        "(from {prev_value:.6e} at timestep {first_timestep} to "
        "{curr_value:.6e} at timestep {timestep})."
    ),
    where=(
        "First deviation detected at timestep {first_timestep}. "
        "Monotonic {direction} since then{location_suffix}."
    ),
    why=(
        "A gradual energy drift is consistent with {diagnosis}."
    ),
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Total Energy",
    pattern=ViolationPattern.OSCILLATORY,
    what=(
        "Total energy oscillating with growing amplitude. "
        "Current deviation: {relative_error:.2%} at timestep {timestep}."
    ),
    where="Oscillation first exceeded threshold at timestep {first_timestep}{location_suffix}.",
    why=(
        "Growing energy oscillation is consistent with {diagnosis}."
    ),
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Total Energy",
    pattern=ViolationPattern.DIVERGENT,
    what=(
        "Total energy diverging exponentially. Current value {curr_value:.6e} "
        "(expected ~{prev_value:.6e}), deviation {relative_error:.2%} at "
        "timestep {timestep}."
    ),
    where="Exponential growth started at timestep {first_timestep}{location_suffix}.",
    why=(
        "Exponential energy growth is consistent with {diagnosis}."
    ),
    fix="{suggestion}",
))

# ==========================================================================
# Linear Momentum templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Linear Momentum",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Total linear momentum changed by {relative_error:.2%} at timestep "
        "{timestep} (magnitude: {prev_value:.6e} -> {curr_value:.6e})."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why="A sudden momentum change is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Linear Momentum",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Total linear momentum drifting. Change of {relative_error:.2%} over "
        "{duration} timesteps."
    ),
    where="First deviation at timestep {first_timestep}{location_suffix}.",
    why="Gradual momentum drift is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Linear Momentum",
    pattern=ViolationPattern.OSCILLATORY,
    what=(
        "Total linear momentum oscillating with growing amplitude. "
        "Deviation: {relative_error:.2%} at timestep {timestep}."
    ),
    where="Oscillation began at timestep {first_timestep}{location_suffix}.",
    why="Growing momentum oscillation is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Linear Momentum",
    pattern=ViolationPattern.DIVERGENT,
    what=(
        "Total linear momentum diverging exponentially. Deviation: "
        "{relative_error:.2%} at timestep {timestep}."
    ),
    where="Divergence started at timestep {first_timestep}{location_suffix}.",
    why="Exponential momentum growth is consistent with {diagnosis}.",
    fix="{suggestion}",
))

# ==========================================================================
# Angular Momentum templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Angular Momentum",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Total angular momentum changed by {relative_error:.2%} at timestep "
        "{timestep}."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why="A sudden angular momentum change is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Angular Momentum",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Total angular momentum drifting by {relative_error:.2%} over "
        "{duration} timesteps."
    ),
    where="First deviation at timestep {first_timestep}{location_suffix}.",
    why="Gradual angular momentum drift is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Angular Momentum",
    pattern=ViolationPattern.OSCILLATORY,
    what=(
        "Angular momentum oscillating with growing amplitude. "
        "Deviation: {relative_error:.2%}."
    ),
    where="Oscillation began at timestep {first_timestep}{location_suffix}.",
    why="Growing angular momentum oscillation is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Angular Momentum",
    pattern=ViolationPattern.DIVERGENT,
    what=(
        "Angular momentum diverging. Deviation: {relative_error:.2%} at "
        "timestep {timestep}."
    ),
    where="Divergence started at timestep {first_timestep}{location_suffix}.",
    why="Exponential angular momentum growth is consistent with {diagnosis}.",
    fix="{suggestion}",
))

# ==========================================================================
# Charge Conservation templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Charge Conservation",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Total charge changed by {absolute_error:.6e} at timestep {timestep} "
        "(from {prev_value:.6e} to {curr_value:.6e})."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why="A sudden charge change is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Charge Conservation",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Total charge drifting. Change of {absolute_error:.6e} over {duration} "
        "timesteps."
    ),
    where="First deviation at timestep {first_timestep}{location_suffix}.",
    why="Gradual charge drift is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Charge Conservation",
    pattern=ViolationPattern.DIVERGENT,
    what=(
        "Total charge diverging. Absolute change: {absolute_error:.6e} at "
        "timestep {timestep}."
    ),
    where="Divergence started at timestep {first_timestep}{location_suffix}.",
    why="Charge divergence is consistent with {diagnosis}.",
    fix="{suggestion}",
))

# ==========================================================================
# Particle Count templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Particle Count",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Particle count changed from {prev_value:.0f} to {curr_value:.0f} at "
        "timestep {timestep} ({absolute_error:.0f} particles "
        "{count_direction})."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why="A sudden particle count change is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Particle Count",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Particle count changing over time. Current: {curr_value:.0f} "
        "(expected: {prev_value:.0f}), {absolute_error:.0f} particles "
        "{count_direction}."
    ),
    where="First change at timestep {first_timestep}{location_suffix}.",
    why="Gradual particle count change is consistent with {diagnosis}.",
    fix="{suggestion}",
))

# ==========================================================================
# Boris Energy templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Boris Energy",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Boris pusher kinetic energy changed by {relative_error:.2%} at "
        "timestep {timestep} (from {prev_value:.6e} to {curr_value:.6e})."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why=(
        "A sudden energy change in the Boris pusher is consistent with "
        "{diagnosis}."
    ),
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Boris Energy",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Boris pusher kinetic energy drifting by {relative_error:.2%} over "
        "{duration} timesteps."
    ),
    where="First deviation at timestep {first_timestep}{location_suffix}.",
    why="Gradual Boris energy drift is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Boris Energy",
    pattern=ViolationPattern.OSCILLATORY,
    what=(
        "Boris pusher kinetic energy oscillating with growing amplitude. "
        "Deviation: {relative_error:.2%} at timestep {timestep}."
    ),
    where="Oscillation began at timestep {first_timestep}{location_suffix}.",
    why="Growing Boris energy oscillation is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Boris Energy",
    pattern=ViolationPattern.DIVERGENT,
    what=(
        "Boris pusher kinetic energy diverging exponentially. "
        "Deviation: {relative_error:.2%} at timestep {timestep}."
    ),
    where="Divergence started at timestep {first_timestep}{location_suffix}.",
    why="Exponential Boris energy growth is consistent with {diagnosis}.",
    fix="{suggestion}",
))

# ==========================================================================
# Gauss's Law templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Gauss's Law",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Gauss's law residual (RMS of div(E) - rho/eps_0) jumped to "
        "{curr_value:.6e} at timestep {timestep}."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why=(
        "A sudden Gauss's law violation is consistent with {diagnosis}."
    ),
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Gauss's Law",
    pattern=ViolationPattern.GRADUAL,
    what=(
        "Gauss's law residual growing. RMS: {curr_value:.6e} at timestep "
        "{timestep} (was {prev_value:.6e} at timestep {first_timestep})."
    ),
    where="Residual started growing at timestep {first_timestep}{location_suffix}.",
    why="Growing Gauss's law residual is consistent with {diagnosis}.",
    fix="{suggestion}",
))

_register(ExplanationTemplate(
    invariant_type="Gauss's Law",
    pattern=ViolationPattern.DIVERGENT,
    what=(
        "Gauss's law residual diverging exponentially. RMS: {curr_value:.6e}."
    ),
    where="Divergence started at timestep {first_timestep}{location_suffix}.",
    why="Diverging Gauss's law residual is consistent with {diagnosis}.",
    fix="{suggestion}",
))

# ==========================================================================
# Lorentz Force templates
# ==========================================================================

_register(ExplanationTemplate(
    invariant_type="Lorentz Force",
    pattern=ViolationPattern.SUDDEN,
    what=(
        "Lorentz force residual (normalised) is {curr_value:.6e} at timestep "
        "{timestep}. The applied force does not match q(E + v x B)."
    ),
    where="Violation is localised to timestep {timestep}{location_suffix}.",
    why="Lorentz force inconsistency is consistent with {diagnosis}.",
    fix="{suggestion}",
))


def get_template(
    invariant_name: str,
    pattern: ViolationPattern,
) -> ExplanationTemplate | None:
    """Look up the explanation template for a given invariant and pattern.

    Returns None if no template exists for the combination.
    """
    return TEMPLATES.get((invariant_name, pattern))
