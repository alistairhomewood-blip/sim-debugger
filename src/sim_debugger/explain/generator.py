"""Physics-language explanation generator.

Takes a Violation object (with optional localisation data) and produces
a human-readable explanation using the template system. Deterministic,
template-based -- no LLM dependency.
"""

from __future__ import annotations

from sim_debugger.core.violations import (
    Violation,
    ViolationPattern,
)
from sim_debugger.explain.templates import get_template

# ---------------------------------------------------------------------------
# Diagnosis lookup: (invariant_name, pattern, sign) -> (diagnosis, suggestion)
# ---------------------------------------------------------------------------

_DIAGNOSES: dict[tuple[str, ViolationPattern, str], tuple[str, str]] = {
    # Total Energy
    ("Total Energy", ViolationPattern.SUDDEN, "positive"): (
        "a force computation sign error or a timestep exceeding the "
        "CFL/Boris stability limit",
        "Reduce the timestep by a factor of 2 and check if the violation "
        "disappears. Verify the sign of all force terms in the particle pusher.",
    ),
    ("Total Energy", ViolationPattern.SUDDEN, "negative"): (
        "artificial dissipation from a non-conservative numerical scheme",
        "Switch to a symplectic integrator if currently using a "
        "non-symplectic one (e.g., replace Euler with leapfrog/Verlet).",
    ),
    ("Total Energy", ViolationPattern.GRADUAL, "positive"): (
        "a non-symplectic integrator causing secular energy drift, or "
        "incorrect operator splitting",
        "Use a symplectic integrator (leapfrog, Verlet, Boris). If already "
        "using one, verify the half-step structure is correct.",
    ),
    ("Total Energy", ViolationPattern.GRADUAL, "negative"): (
        "numerical damping or missing energy terms in the Hamiltonian",
        "Check that all energy contributions (kinetic, potential, field) "
        "are accounted for. Verify there is no unintended friction term.",
    ),
    ("Total Energy", ViolationPattern.OSCILLATORY, "positive"): (
        "resonance between the integration timestep and a natural frequency "
        "of the system",
        "Try a different timestep that avoids resonance. Consider adaptive "
        "timestepping.",
    ),
    ("Total Energy", ViolationPattern.DIVERGENT, "positive"): (
        "numerical instability; the system may be ill-conditioned or the "
        "timestep exceeds the stability limit",
        "Reduce the timestep. Check the condition number of the system. "
        "Consider an implicit integrator for stiff systems.",
    ),
    # Linear Momentum
    ("Linear Momentum", ViolationPattern.SUDDEN, "positive"): (
        "a non-symmetric force computation (F_ij != -F_ji) or an external "
        "force being applied incorrectly",
        "Verify Newton's third law: ensure F_ij = -F_ji for all particle "
        "pairs. Check boundary conditions for momentum conservation.",
    ),
    ("Linear Momentum", ViolationPattern.GRADUAL, "positive"): (
        "a systematic asymmetry in the force computation or boundary "
        "conditions that are not momentum-conserving",
        "Check boundary conditions. Verify that all internal forces "
        "satisfy Newton's third law.",
    ),
    ("Linear Momentum", ViolationPattern.OSCILLATORY, "positive"): (
        "oscillating boundary forces or periodic force asymmetry",
        "Review boundary condition implementation. Check for time-dependent "
        "external forces that should not be present.",
    ),
    ("Linear Momentum", ViolationPattern.DIVERGENT, "positive"): (
        "runaway force computation error or numerical instability in "
        "force evaluation",
        "Check for overflow in force computation. Reduce timestep. "
        "Verify particle separations are not becoming too small.",
    ),
    # Angular Momentum
    ("Angular Momentum", ViolationPattern.SUDDEN, "positive"): (
        "a symmetry-breaking bug or incorrect force direction",
        "Verify that forces are central (directed along the line between "
        "particles). Check for coordinate system errors.",
    ),
    ("Angular Momentum", ViolationPattern.GRADUAL, "positive"): (
        "a slow symmetry-breaking drift, possibly from non-conservative "
        "torques",
        "Check for unintended torques. Verify rotational symmetry of the "
        "force computation.",
    ),
    ("Angular Momentum", ViolationPattern.OSCILLATORY, "positive"): (
        "oscillating torque from boundary interactions or asymmetric forces",
        "Review boundary conditions for angular momentum conservation.",
    ),
    ("Angular Momentum", ViolationPattern.DIVERGENT, "positive"): (
        "runaway angular momentum growth from force computation error",
        "Check force directions and magnitudes. Reduce timestep.",
    ),
    # Charge Conservation
    ("Charge Conservation", ViolationPattern.SUDDEN, "positive"): (
        "a particle boundary condition error (particles not properly "
        "removed/reflected) or a charge deposition bug",
        "Check particle boundary conditions. Verify that absorbed particles "
        "have their charge properly removed from the system.",
    ),
    ("Charge Conservation", ViolationPattern.GRADUAL, "positive"): (
        "a non-charge-conserving current deposition scheme or slow charge "
        "leakage at boundaries",
        "Switch to a charge-conserving current deposition scheme "
        "(e.g., Esirkepov method). Check boundary charge handling.",
    ),
    ("Charge Conservation", ViolationPattern.DIVERGENT, "positive"): (
        "catastrophic charge non-conservation, possibly from a fundamental "
        "error in the deposition algorithm",
        "Review the current deposition implementation in detail. Verify "
        "that div(J) + d(rho)/dt = 0 is satisfied at the discrete level.",
    ),
    # Particle Count
    ("Particle Count", ViolationPattern.SUDDEN, "positive"): (
        "spurious particle creation at a boundary or in the injection "
        "routine",
        "Check boundary conditions and particle injection code. Verify "
        "that no particles are duplicated.",
    ),
    ("Particle Count", ViolationPattern.SUDDEN, "negative"): (
        "particles being destroyed at boundaries when they should be "
        "reflected or periodic",
        "Check boundary condition type (absorbing vs reflecting vs periodic). "
        "Verify particles are not incorrectly removed.",
    ),
    ("Particle Count", ViolationPattern.GRADUAL, "positive"): (
        "slow particle leakage from boundary handling or injection",
        "Check for off-by-one errors in boundary detection. Verify particle "
        "injection rate.",
    ),
    ("Particle Count", ViolationPattern.GRADUAL, "negative"): (
        "slow particle loss at boundaries",
        "Check boundary conditions. Verify absorbing boundaries are "
        "intentional.",
    ),
    # Boris Energy
    ("Boris Energy", ViolationPattern.SUDDEN, "positive"): (
        "the magnetic field rotation angle (omega_c * dt) exceeding the "
        "Boris stability limit, or a sign error in the E-field push",
        "Reduce dt so that omega_c * dt < 2. Check the sign of all "
        "terms in the Boris pusher half-step.",
    ),
    ("Boris Energy", ViolationPattern.GRADUAL, "positive"): (
        "an incorrect half-step structure in the Boris pusher (e.g., "
        "full E-push instead of two half-pushes), causing O(dt) energy "
        "drift instead of O(dt^2)",
        "Verify the Boris pusher implements the correct three-step "
        "structure: half E-push, B-rotation, half E-push. Check that "
        "E-field is evaluated at the correct particle position.",
    ),
    ("Boris Energy", ViolationPattern.GRADUAL, "negative"): (
        "artificial energy dissipation in the Boris rotation step or "
        "missing energy contributions",
        "Check the Boris rotation formula. Verify that |v| is preserved "
        "exactly by the rotation step (B-only case).",
    ),
    ("Boris Energy", ViolationPattern.OSCILLATORY, "positive"): (
        "the Boris rotation operating near the stability boundary "
        "(omega_c * dt close to 2), causing oscillating energy errors",
        "Reduce dt to bring omega_c * dt well below 2. Consider the "
        "implicit Boris algorithm for large omega_c * dt.",
    ),
    ("Boris Energy", ViolationPattern.DIVERGENT, "positive"): (
        "omega_c * dt > 2, causing exponential energy growth in the "
        "Boris pusher. The magnetic field rotation has become unstable",
        "Reduce dt so that omega_c * dt < 2 for all particles. "
        "Alternatively, switch to the implicit Boris algorithm which "
        "is unconditionally stable for large omega_c * dt.",
    ),
    # Gauss's Law
    ("Gauss's Law", ViolationPattern.SUDDEN, "positive"): (
        "a sudden error in the field solver or charge deposition, "
        "possibly from incorrect boundary conditions",
        "Check the Poisson solver boundary conditions. Verify the "
        "charge deposition stencil is correct.",
    ),
    ("Gauss's Law", ViolationPattern.GRADUAL, "positive"): (
        "accumulating error from a non-charge-conserving current "
        "deposition scheme",
        "Switch to a charge-conserving scheme (Esirkepov, Villasenor-Buneman). "
        "Alternatively, apply Boris correction to clean div(E).",
    ),
    ("Gauss's Law", ViolationPattern.DIVERGENT, "positive"): (
        "catastrophic failure in the field solver or charge deposition",
        "Review the field solver and deposition code. Check for "
        "numerical overflow.",
    ),
    # Lorentz Force
    ("Lorentz Force", ViolationPattern.SUDDEN, "positive"): (
        "a sign error in the v x B cross product, a wrong field "
        "interpolation, or a missing charge/mass factor",
        "Check the cross product implementation (right-hand rule). "
        "Verify field interpolation uses the correct particle position. "
        "Check units and factors of q and m.",
    ),
}


def generate_explanation(
    violation: Violation,
    pattern: ViolationPattern | None = None,
    first_timestep: int | None = None,
    duration: int | None = None,
) -> str:
    """Generate a physics-language explanation for a violation.

    Args:
        violation: The detected violation.
        pattern: The violation pattern (from temporal localisation).
                If None, defaults to SUDDEN.
        first_timestep: Timestep where violation first appeared.
                       Defaults to violation.timestep.
        duration: Number of timesteps from first deviation to detection.
                 Defaults to 0.

    Returns:
        A multi-line string with the full explanation.
    """
    if pattern is None:
        if violation.localisation and violation.localisation.temporal:
            pattern = violation.localisation.temporal.pattern
        else:
            pattern = ViolationPattern.SUDDEN

    if first_timestep is None:
        if violation.localisation and violation.localisation.temporal:
            first_timestep = violation.localisation.temporal.first_violation_timestep
        else:
            first_timestep = violation.timestep

    if duration is None:
        if violation.localisation and violation.localisation.temporal:
            duration = violation.localisation.temporal.duration
        else:
            duration = 0

    # Determine sign for diagnosis lookup
    sign = "positive" if violation.actual_value >= violation.expected_value else "negative"

    # Build template variables
    location_suffix = ""
    if violation.localisation and violation.localisation.source:
        src = violation.localisation.source
        location_suffix = f" (in {src.function_name}, {src.file}:{src.line_start})"

    count_direction = "gained" if violation.actual_value > violation.expected_value else "lost"
    direction = "increase" if violation.actual_value > violation.expected_value else "decrease"

    # Look up diagnosis
    diag_key = (violation.invariant_name, pattern, sign)
    diagnosis, suggestion = _DIAGNOSES.get(
        diag_key,
        # Fallback: try with "positive" sign
        _DIAGNOSES.get(
            (violation.invariant_name, pattern, "positive"),
            ("an unclassified numerical error", "Review the simulation code for correctness."),
        ),
    )

    template_vars = {
        "relative_error": violation.relative_error,
        "absolute_error": violation.absolute_error,
        "timestep": violation.timestep,
        "first_timestep": first_timestep,
        "duration": duration,
        "prev_value": violation.expected_value,
        "curr_value": violation.actual_value,
        "location_suffix": location_suffix,
        "diagnosis": diagnosis,
        "suggestion": suggestion,
        "count_direction": count_direction,
        "direction": direction,
    }

    # Try to use the template system
    template = get_template(violation.invariant_name, pattern)
    if template is not None:
        try:
            what = template.what.format(**template_vars)
            where = template.where.format(**template_vars)
            why = template.why.format(**template_vars)
            fix = template.fix.format(**template_vars)
            return f"{what}\n{where}\n{why}\n\nSuggested fix:\n  {fix}"
        except (KeyError, ValueError):
            pass

    # Fallback: generate a basic explanation without templates
    return _generate_fallback(violation, pattern, diagnosis, suggestion, template_vars)


def _generate_fallback(
    violation: Violation,
    pattern: ViolationPattern,
    diagnosis: str,
    suggestion: str,
    template_vars: dict,
) -> str:
    """Generate a basic explanation when no template matches."""
    lines = [
        f"{violation.invariant_name} violation detected at timestep "
        f"{violation.timestep}.",
        f"  Value changed from {violation.expected_value:.6e} to "
        f"{violation.actual_value:.6e} "
        f"(relative error: {violation.relative_error:.2%}).",
        f"  Pattern: {pattern.value}.",
        f"  This is consistent with {diagnosis}.",
        f"\nSuggested fix:\n  {suggestion}",
    ]
    return "\n".join(lines)
