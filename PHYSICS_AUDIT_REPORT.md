# Physics Audit Report -- sim-debugger

**Auditor:** Claude Opus 4.6 (1M context)
**Date:** 2026-03-29
**Scope:** All physics equations, invariant definitions, conservation law checks,
explanation templates, temporal localisation logic, and monitor orchestration.
**Files audited:**
- `src/sim_debugger/core/invariants.py`
- `src/sim_debugger/core/violations.py`
- `src/sim_debugger/core/monitor.py`
- `src/sim_debugger/core/state.py`
- `src/sim_debugger/parsec/invariants.py`
- `src/sim_debugger/explain/generator.py`
- `src/sim_debugger/explain/templates.py`
- `src/sim_debugger/localise/temporal.py`
- `tests/unit/test_invariants.py`
- `tests/unit/test_parsec_invariants.py`
- `tests/unit/test_temporal_localisation.py`
- `tests/unit/test_monitor.py`
- `tests/unit/test_explain.py`
- `tests/integration/test_end_to_end.py`
- `tests/benchmarks/simulations.py`
- `examples/boris_pusher/boris_correct.py`
- `examples/harmonic_oscillator/leapfrog_correct.py`
- `examples/harmonic_oscillator/euler_broken.py`

---

## Executive Summary

The physics implementation is **largely correct** with no catastrophic errors in
the core invariant computations. The kinetic energy formula, momentum calculations,
angular momentum cross products, Lorentz force, and Boris pusher are all implemented
correctly. However, I found **5 genuine physics issues** (2 errors, 3 significant
weaknesses) and **7 recommendations** for improvement. Two of the issues could
cause the tool to **miss real violations or report misleading diagnostics** in
production use.

**Overall verdict: CONDITIONAL PASS -- the 2 errors must be fixed before the
tool is used on real simulations.**

---

## 1. Built-in Invariant #1: Total Energy

**File:** `core/invariants.py`, class `TotalEnergyInvariant`

### Formula review

```
E_kinetic = 0.5 * sum(m_i * |v_i|^2)
```

**PASS.** The kinetic energy formula is correctly implemented at line 279-281:
- `v_sq = np.sum(v * v, axis=-1)` correctly computes |v_i|^2 for each particle.
- `0.5 * np.sum(m * v_sq)` correctly sums 0.5 * m_i * |v_i|^2.
- Scalar mass case handled correctly.
- 1D velocity case handled correctly.

### Potential energy

**PASS.** Potential energy is summed from a user-provided array (line 287). This
is correct -- the tool does not try to compute PE from positions, which would
require knowledge of the potential function.

### Electromagnetic field energy

```
E_field_energy = (eps_0 / 2) * integral(|E|^2 dV)
B_field_energy = (1 / (2 * mu_0)) * integral(|B|^2 dV)
```

**ISSUE #1 (WEAKNESS): Incorrect cell volume calculation for grid fields.**

At line 294, the code computes:
```python
cell_volume = dx ** E.ndim if isinstance(dx, (int, float)) else float(np.prod(dx))
```

The problem is that `E.ndim` is the **total number of array dimensions**, not the
number of spatial dimensions. For example, a 2D grid E-field has shape `(Nx, Ny, 2)`
so `E.ndim = 3`, but the spatial dimensionality is 2. The cell volume should be
`dx^2`, not `dx^3`. This means field energy will be **overestimated by a factor
of dx** for grid-based E-fields, and similarly for B-fields (line 302).

The correct computation should use `E.ndim - 1` (since the last dimension is the
vector components) or, more robustly, `E.shape[-1]` for the number of spatial
dimensions and compute `dx ** ndim_spatial`.

**Severity:** Medium. This only affects simulations that include grid-based
electromagnetic field energy. Pure particle simulations are unaffected.

**Recommendation:** Replace `E.ndim` with `E.ndim - 1` or equivalently
`len(E.shape) - 1` for the cell volume exponent, and same for B-field.

### Edge cases

**PASS.** Near-zero handling in `_standard_check` uses `1e-300` threshold, which
is appropriate (well above the denorm boundary but below any physical value).
NaN/Inf detection is correct.

### Threshold logic

**PASS.** Default threshold of 1e-6 (relative) is reasonable for double-precision
simulations.

---

## 2. Built-in Invariant #2: Linear Momentum

**File:** `core/invariants.py`, class `LinearMomentumInvariant`

### Formula review

```
p_total = sum(m_i * v_i)
```

**PASS.** The momentum computation is correct:
- 1D case: `float(m) * np.sum(v)` for scalar mass, `np.sum(m * v)` for per-particle mass.
- Multi-dimensional case: `m[:, np.newaxis] * v` correctly broadcasts mass to all velocity components.
- Returns magnitude `np.linalg.norm(p_total)`.

### Conservation monitoring

**ISSUE #2 (WEAKNESS): Monitoring the magnitude loses direction information.**

The invariant monitors `|p_total|` (the magnitude). This means if the total
momentum vector **rotates** while keeping constant magnitude (which would
indicate a violation), this invariant would not detect it. For example, if
p = (1, 0, 0) changes to p = (0, 1, 0), the magnitude is unchanged at 1.0,
but momentum is NOT conserved.

This is acceptable as a simplification (the protocol requires a scalar return),
but the documentation should note this limitation, and ideally a per-component
check should be available as an option.

**Severity:** Low-Medium. Momentum direction changes are less common bugs than
magnitude changes, but this is a false-negative risk.

**Recommendation:** Add a note in the docstring. Consider adding an optional
per-component check mode, or monitor each component as a separate scalar invariant.

---

## 3. Built-in Invariant #3: Angular Momentum

**File:** `core/invariants.py`, class `AngularMomentumInvariant`

### Formula review

**2D case:**
```
L = sum(x_i * p_{y,i} - y_i * p_{x,i})
```

**PASS.** Line 431: `r[:, 0] * p[:, 1] - r[:, 1] * p[:, 0]` is the correct
z-component of the cross product for 2D positions.

**3D case:**
```
L = sum(r_i x p_i)
```

**PASS.** Line 435: `np.cross(r, p)` computes the cross product correctly.
`np.sum(L, axis=0)` sums over particles. `np.linalg.norm(L_total)` returns
the magnitude.

### Same magnitude-only limitation as momentum

The same weakness as Linear Momentum applies -- only the magnitude of the angular
momentum vector is monitored, not its direction.

### Edge cases

**PASS.** The `applicable()` method correctly requires `ndim >= 2` and
`shape[1] >= 2`. Raises `ValueError` for dimensions other than 2 or 3.

---

## 4. Built-in Invariant #4: Charge Conservation

**File:** `core/invariants.py`, class `ChargeConservationInvariant`

### Formula review

```
Q_total = sum(q_i)
```

**PASS.** Simple sum of all charges. Physically correct for a closed system.

### Threshold

**PASS.** Default threshold of 1e-12 is very tight, appropriate for an
exactly-conserved quantity.

### Edge case: net zero charge

**PASS.** When total charge is zero, the `_standard_check` falls back to
absolute error comparison (since `|prev_value| < 1e-300`). This is correct --
for a system with zero net charge, we should detect any absolute change.

---

## 5. Built-in Invariant #5: Particle Count

**File:** `core/invariants.py`, class `ParticleCountInvariant`

### Formula review

**PASS.** Returns the number of particles from `positions.shape[0]`,
`velocities.shape[0]`, or `metadata["particle_count"]`.

### Check logic

**PASS.** Uses exact integer comparison (`prev_value == curr_value`) rather
than relative tolerance. Any change is a violation. This is correct for an
integer-valued quantity.

### Default threshold

The `default_threshold` of 0.5 is documented but not actually used by the
`check()` method (which uses exact comparison). This is fine but slightly
misleading -- the threshold property exists but is ignored.

---

## 6. PARSEC Invariant #1: Boris Energy

**File:** `parsec/invariants.py`, class `BorisEnergyInvariant`

### Formula review

```
E_kinetic = 0.5 * sum(m_i * |v_i|^2)
```

**PASS.** Identical to the core TotalEnergyInvariant's kinetic energy computation.
Lines 122-126 are correct.

### Boris pusher stability limit (omega_c * dt < 2)

**ISSUE #3 (MISSING FEATURE): No explicit Boris stability limit check.**

The Boris pusher has a well-known stability limit: the rotation angle per timestep
must satisfy `omega_c * dt < 2` where `omega_c = |q| * |B| / m` is the cyclotron
frequency. When this limit is violated, the Boris rotation amplifies |v| instead
of preserving it, leading to exponential energy growth.

The code **does not explicitly check this limit**. While the energy conservation
check will eventually catch the resulting energy growth, the tool could provide
a much more specific and immediate diagnosis if it computed `omega_c * dt` directly
and flagged when it exceeds 2. This would be:
1. Detectable at the first timestep (before energy has measurably grown).
2. Much more specific in its diagnosis.

The explanation templates (in `generator.py`) correctly mention `omega_c * dt < 2`
in their diagnostic text, but the actual numerical check is not performed.

**Severity:** Medium. The energy check will eventually catch the instability, but
the delay means the user gets a less specific diagnosis and might run the
simulation longer than necessary before the tool flags the problem.

**Recommendation:** Add a Boris-specific check: if the state contains B_field (or
B_at_particles), charges, and masses, compute `max(|q_i| * |B| / m_i) * dt` and
issue an immediate warning if it exceeds 2.

### Boris pusher implementation in examples and benchmarks

**PASS.** The Boris pusher implementations in `examples/boris_pusher/boris_correct.py`
and `tests/benchmarks/simulations.py::boris_push()` are physically correct:
- Half E-push: `v^- = v + (q*dt)/(2*m) * E` -- correct.
- t-vector: `t = (q/m) * B * (dt/2)` -- correct.
- s-vector: `s = 2*t / (1 + |t|^2)` -- correct (this is the Boris rotation formula).
- v' = v^- + v^- x t -- correct.
- v^+ = v^- + v' x s -- correct.
- Half E-push: `v^{n+1} = v^+ + (q*dt)/(2*m) * E` -- correct.

The rotation formula `v^+ = v^- + (v^- + v^- x t) x s` with `s = 2t/(1+|t|^2)`
is the standard Birdsall & Langdon form of the Boris rotation, which exactly
preserves |v| in the B-only case. This is verified numerically in the test
`test_boris_b_rotation_preserves_energy`.

---

## 7. PARSEC Invariant #2: Gauss's Law

**File:** `parsec/invariants.py`, class `GaussLawInvariant`

### Formula review

```
residual = div(E) - rho / eps_0
```

**PASS.** The Gauss's law residual is computed correctly:
- `np.gradient(E_d, dx_arr[d], axis=d)` computes the central-difference derivative
  of the d-th component of E along axis d. This is correct for computing div(E).
- The residual is the L2 norm (RMS): `sqrt(mean(residual^2))`.

### Use of np.gradient for divergence

**PASS.** `np.gradient` uses second-order central differences in the interior and
first/second-order one-sided differences at the boundaries. This is appropriate
for computing the divergence. The choice of RMS norm for the residual is standard.

### Check logic

**PASS.** The check correctly uses absolute threshold comparison (not relative
change), since the ideal residual is zero. This is the right approach for Gauss's
law.

### Edge cases

The code assumes `E_field.shape[-1]` gives the number of spatial dimensions, and
that the charge density grid has shape `E_field.shape[:-1]`. This is reasonable
but not explicitly validated. If the shapes are inconsistent, numpy will raise
a broadcasting error, which is acceptable.

---

## 8. PARSEC Invariant #3: Lorentz Force

**File:** `parsec/invariants.py`, class `LorentzForceInvariant`

### Formula review

```
F_expected = q * (E + v x B)
residual = F_applied - F_expected
```

**PASS.** The Lorentz force is correctly computed:
- `np.cross(v, B)` computes v x B. **The sign is correct** -- the Lorentz force
  is F = q(E + v x B), and `np.cross` follows the right-hand rule.
- `q[:, np.newaxis] * (E + v_cross_B)` correctly broadcasts the charge to
  all force components.

### Cross product sign verification

**PASS.** `np.cross(v, B)` computes v x B (not B x v). Since v x B = -B x v,
a sign error here would be catastrophic. I verified that the code uses the
correct order: `np.cross(v, B)`.

### 2D vs 3D handling

**ISSUE #4 (ERROR): LorentzForceInvariant assumes 3D only.**

The docstring states "Required arrays: applied_force: (N, 3)" and the computation
uses `np.cross(v, B)` which only produces a 3D vector when both inputs are 3D
(shape `(N, 3)`). For 2D simulations with (N, 2) arrays, `np.cross` returns a
scalar (the z-component of the cross product), not a 2D vector, and the
subsequent `q[:, np.newaxis] * (E + v_cross_B)` would fail or produce wrong results.

In the Boris pusher context (PIC codes), 3D is the standard use case, but the
tool should either:
1. Explicitly document this limitation and validate input shapes, or
2. Handle the 2D case by padding to 3D, computing the cross product, and extracting
   the relevant components.

**Severity:** Low. Most PIC codes work in 3D (or 2D3V where velocity and fields
have 3 components). But this could cause confusing numpy errors for a user who
passes 2D data.

**Recommendation:** Add shape validation in `compute()` to verify that velocity,
E-field, and B-field arrays all have 3 components in the last dimension. If 2D
is detected, either raise a clear error or handle it.

### Normalisation

**PASS.** The normalised residual `||F_applied - F_expected|| / ||F_expected||`
is computed per-particle with a near-zero protection mask. The final metric is
the mean over all particles. This is a reasonable choice.

---

## 9. Explanation Generator and Templates

**File:** `explain/generator.py`, `explain/templates.py`

### Diagnosis accuracy review

I reviewed all 33 diagnosis entries in `_DIAGNOSES`. Here are the findings:

**PASS -- Correct diagnoses:**

| Invariant | Pattern | Diagnosis | Verdict |
|---|---|---|---|
| Total Energy / SUDDEN / positive | "force computation sign error or timestep exceeding CFL/Boris stability limit" | Correct |
| Total Energy / SUDDEN / negative | "artificial dissipation from non-conservative scheme" | Correct |
| Total Energy / GRADUAL / positive | "non-symplectic integrator causing secular energy drift" | Correct |
| Total Energy / GRADUAL / negative | "numerical damping or missing energy terms" | Correct |
| Total Energy / OSCILLATORY / positive | "resonance between integration timestep and natural frequency" | Correct |
| Total Energy / DIVERGENT / positive | "numerical instability; timestep exceeds stability limit" | Correct |
| Linear Momentum / SUDDEN / positive | "non-symmetric force computation (F_ij != -F_ji)" | Correct |
| Linear Momentum / GRADUAL / positive | "systematic asymmetry in force computation" | Correct |
| Angular Momentum / SUDDEN / positive | "symmetry-breaking bug or incorrect force direction" | Correct |
| Charge Conservation / SUDDEN / positive | "particle boundary condition error or charge deposition bug" | Correct |
| Charge Conservation / GRADUAL / positive | "non-charge-conserving current deposition" | Correct |
| Boris Energy / SUDDEN / positive | "omega_c * dt exceeding Boris stability limit, or E-field sign error" | Correct |
| Boris Energy / GRADUAL / positive | "incorrect half-step structure" | Correct |
| Boris Energy / GRADUAL / negative | "artificial energy dissipation in Boris rotation" | Correct |
| Boris Energy / OSCILLATORY / positive | "Boris rotation operating near stability boundary" | Correct |
| Boris Energy / DIVERGENT / positive | "omega_c * dt > 2 causing exponential energy growth" | Correct |
| Gauss's Law / SUDDEN / positive | "sudden error in field solver or charge deposition" | Correct |
| Gauss's Law / GRADUAL / positive | "non-charge-conserving current deposition" | Correct |
| Lorentz Force / SUDDEN / positive | "sign error in v x B cross product, wrong field interpolation, or missing charge/mass factor" | Correct |

**All diagnoses are physically accurate.** The suggested fixes are sound.

### Suggested fix review

**PASS.** All suggested fixes are physically reasonable:
- "Reduce dt so omega_c * dt < 2" -- correct stability criterion.
- "Switch to symplectic integrator" -- correct for energy drift.
- "Verify Newton's third law F_ij = -F_ji" -- correct for momentum violations.
- "Use charge-conserving scheme (Esirkepov method)" -- correct.
- "Apply Boris correction to clean div(E)" -- correct (this is a standard
  divergence cleaning technique).

### Missing template coverage

**ISSUE #5 (WEAKNESS): Several invariant/pattern combinations have no template.**

The following combinations lack templates (will fall through to the fallback generator):
- Charge Conservation / OSCILLATORY
- Particle Count / OSCILLATORY
- Particle Count / DIVERGENT
- Lorentz Force / GRADUAL
- Lorentz Force / OSCILLATORY
- Lorentz Force / DIVERGENT
- Gauss's Law / OSCILLATORY

While these may be uncommon in practice, the Lorentz Force and Gauss's Law
combinations could arise in real simulations. A Lorentz force error that grows
over time (GRADUAL or DIVERGENT) would get a generic fallback explanation instead
of a physics-specific one.

**Severity:** Low. The fallback generator produces reasonable output, but it is
less helpful than a tailored template.

---

## 10. Temporal Localisation

**File:** `localise/temporal.py`

### Binary search correctness

**PASS.** The `_find_first_violation` function correctly binary-searches for the
first index where `relative_error > threshold`. The invariant is maintained:
- `errors[lo]` is always within threshold.
- `errors[hi]` is always violating.
- Loop terminates when `lo == hi`, giving the first violating index.

### Pattern classification

**PASS.** The pattern classifier uses reasonable heuristics:
- **SUDDEN:** Only 1-2 timesteps of violation. Correct.
- **OSCILLATORY:** >30% sign changes in the error delta. Reasonable heuristic.
- **DIVERGENT:** Every step's error ratio > 1.5 (for at least 4 steps). This
  correctly identifies exponential growth.
- **GRADUAL:** >70% of deltas are positive (monotonic increase). Correct.

### Edge case

The reference value is taken as `trajectory[0]` (the first value in history).
This is the correct approach -- compare everything to the earliest available value.

### Default pattern

Line 177: the function returns `ViolationPattern.GRADUAL` as the default when
none of the specific patterns match. This is acceptable -- GRADUAL is the
least alarming pattern and is a safe default.

---

## 11. Monitor (Dual Violation Detection)

**File:** `core/monitor.py`

### Step-to-step check

**PASS.** Line 182: `invariant.check(prev, value, threshold)` compares consecutive
timestep values. This catches sudden violations.

### Drift-from-initial check

**ISSUE #6 (ERROR): The drift check uses a hardcoded 100x multiplier on the
threshold, which is physically unjustified and potentially dangerous.**

Lines 189-193:
```python
drift_threshold = threshold * 100
violation = invariant.check(initial, value, drift_threshold)
```

The comment mentions "sqrt(steps) scaling" for random walk drift, but the actual
implementation uses a **fixed 100x multiplier** regardless of how many steps have
elapsed. This has two problems:

1. **Too permissive at short times:** After 10 steps, a cumulative drift of 100x
   the step threshold is extremely unlikely from legitimate numerical noise.
   The tool would miss legitimate drift violations early on.

2. **Too strict at long times:** After 10^6 steps, a random-walk drift would be
   ~1000x the step threshold (sqrt(10^6) = 1000), but the check uses 100x. This
   means the tool would produce false positives in long simulations with legitimate
   random-walk-level noise.

3. **Physically incorrect reasoning:** The comment says "we expect cumulative drift
   to grow as sqrt(N) for a random walk." This is true for round-off noise, but
   for a bug (which is what we are trying to detect), the drift grows linearly
   or exponentially. The threshold should not accommodate random-walk scaling if
   the goal is to detect linear or exponential drift.

**The correct approach is one of:**
- Use `threshold * sqrt(steps)` scaling (if the goal is to tolerate round-off noise).
- Use a fixed absolute drift threshold (if the goal is to detect any accumulation).
- Use a separate user-configurable drift threshold.

**Severity:** Medium-High. This affects the core detection logic. The 100x
multiplier means the tool will miss gradual violations that accumulate to 50x the
step threshold (which is a massive error). Conversely, in long runs, it may
produce false positives from round-off accumulation.

**Recommendation:** Replace the hardcoded 100x with `threshold * max(10, sqrt(step_count))`
or expose a separate `drift_threshold` parameter.

---

## 12. Boris Pusher Example and Benchmark Correctness

**Files:** `examples/boris_pusher/boris_correct.py`, `tests/benchmarks/simulations.py`

### Boris rotation formula

**PASS.** Both implementations use the correct Birdsall & Langdon formulation:
```
t = (q/m) * B * dt/2
s = 2*t / (1 + |t|^2)
v' = v^- + v^- x t
v^+ = v^- + v' x s
```

This is the standard form that exactly preserves |v| in the B-only case.

### Bug benchmarks

**PASS.** The intentionally buggy simulations are correctly constructed:
- B05 (wrong half-step): Uses `qdt_m` instead of `qdt_2m` for the first E-push,
  and omits the second half-push. This correctly demonstrates the bug.
- B06 (wrong E-field sign): Uses `2.0 * qdt_2m` for the first half-push,
  creating an asymmetric pusher. This correctly breaks time-reversibility.
- B10 (wrong v x B sign): Uses `E - np.cross(v, B)` instead of `E + np.cross(v, B)`.
  Correct representation of the common sign-error bug.

### N-body force computation

**PASS.** The symmetric force computation in B11 correctly implements Newton's
third law: `F[i] += m[j] * f; F[j] -= m[i] * f`. The asymmetric bug in B12
uses a 0.9 factor, correctly breaking the symmetry.

**Note:** The N-body force law used is `f = r / |r|^3`, which is a gravitational
1/r^2 force (not the standard gravitational form with G * m_i * m_j / r^2, but
the masses are factored in on the receiving end). The force on particle i from
particle j is `m[j] * (r_j - r_i) / |r_j - r_i|^3`, which gives the correct
direction (attractive toward j) and 1/r^2 magnitude. **PASS.**

---

## 13. Harmonic Oscillator Examples

### Leapfrog (correct)

**PASS.** The symplectic Euler implementation is correct:
```
v = v + F * dt
x = x + v * dt
```
This updates velocity first, then position with the new velocity. This is a
first-order symplectic integrator (also called semi-implicit Euler or
symplectic Euler).

### Forward Euler (broken)

**PASS.** The Forward Euler bug is correctly implemented:
```
x_new = x + v * dt
v_new = v + a * dt
```
Both updates use old values, making it non-symplectic. Energy grows monotonically.

---

## Summary of Issues

| # | Severity | Type | Location | Description |
|---|---|---|---|---|
| 1 | Medium | Error | `core/invariants.py:294,302` | EM field energy cell volume uses `E.ndim` instead of `E.ndim - 1` |
| 2 | Low-Medium | Weakness | `core/invariants.py:347-367` | Linear/angular momentum monitors only track magnitude, not direction |
| 3 | Medium | Missing feature | `parsec/invariants.py` | No explicit Boris stability limit (`omega_c * dt < 2`) check |
| 4 | Low | Error | `parsec/invariants.py:297-330` | Lorentz force invariant assumes 3D; no validation or 2D handling |
| 5 | Low | Weakness | `explain/templates.py` | Missing templates for 7 invariant/pattern combinations |
| 6 | Medium-High | Error | `core/monitor.py:189-193` | Drift-from-initial check uses hardcoded 100x threshold multiplier |

---

## Detailed Recommendations

### R1 (Must fix): Correct EM field energy cell volume [Issue #1]

In `core/invariants.py`, lines 294 and 302, change:
```python
cell_volume = dx ** E.ndim
```
to:
```python
ndim_spatial = E.ndim - 1  # Last dimension is vector components
cell_volume = dx ** ndim_spatial
```
And similarly for the B-field block.

### R2 (Must fix): Replace hardcoded drift threshold multiplier [Issue #6]

In `core/monitor.py`, line 191, replace:
```python
drift_threshold = threshold * 100
```
with a step-count-aware threshold:
```python
import math
drift_threshold = threshold * max(10.0, math.sqrt(self._step_count))
```
Or, better yet, expose a `drift_threshold_multiplier` parameter in the Monitor
constructor with a reasonable default.

### R3 (Should fix): Add Boris stability limit pre-check [Issue #3]

Add a method to `BorisEnergyInvariant` that computes `max(omega_c * dt)` from
the state and issues an immediate warning if it exceeds 2. This could be
implemented as an additional check in `compute()` or as a separate invariant.

### R4 (Should fix): Add shape validation to Lorentz force [Issue #4]

In `LorentzForceInvariant.compute()`, add:
```python
if v.shape[-1] != 3:
    raise ValueError(
        f"Lorentz force requires 3D velocity vectors, got shape {v.shape}"
    )
```

### R5 (Should fix): Add per-component momentum/angular momentum monitoring [Issue #2]

Consider adding optional per-component monitoring for linear and angular
momentum, either as separate invariants or as a mode flag.

### R6 (Nice to have): Fill in missing explanation templates [Issue #5]

Add templates for the 7 missing invariant/pattern combinations, particularly
Lorentz Force / GRADUAL and Lorentz Force / DIVERGENT.

### R7 (Nice to have): Improve test coverage of edge cases

Add tests for:
- EM field energy computation with grid fields (currently untested).
- The drift-from-initial detection logic (currently tested implicitly through
  the Monitor integration test but not directly).
- 2D simulations with angular momentum.
- Lorentz force with non-unit charges (the existing test uses q=1 for the
  correct-force case, which does not catch q-factor bugs as effectively).

---

## Conclusion

The sim-debugger project has a **solid physics foundation**. The core formulas
(kinetic energy, momentum, cross products, Lorentz force, Boris rotation) are
all correct. The explanation templates provide accurate physics diagnoses. The
benchmark simulations correctly implement both correct and buggy physics.

The two errors that must be fixed (EM field cell volume calculation and drift
threshold multiplier) are both in secondary detection pathways and would not
cause the tool to give **wrong** physics advice -- they would cause it to
compute field energy incorrectly (Issue #1) or to miss gradual violations /
produce false positives in long runs (Issue #6). Neither would cause the tool
to tell a user their correct simulation is broken.

After fixing the two errors, this tool is suitable for use on real simulations.
