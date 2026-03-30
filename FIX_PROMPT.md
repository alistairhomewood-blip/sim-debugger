You are implementing fixes for sim-debugger based on two independent external audits (FULL_AUDIT_REPORT.md and CODEX_AUDIT_REPORT.md). Findings confirmed by BOTH audits are marked CRITICAL. Fix everything listed below. Run the full test suite after each logical group of fixes to ensure zero regressions. Do not proceed to the next group if tests are failing.

## Merged Audit Sources

- **FULL_AUDIT_REPORT.md** (2026-03-29): comprehensive manual audit
- **CODEX_AUDIT_REPORT.md** (2026-03-29): automated tool-assisted audit

Findings present in BOTH reports are promoted to CRITICAL regardless of individual severity.

---

## CRITICAL Fixes

### Group CRIT-A: Boris Energy Invariant (parsec/invariants.py)

**CRIT-01: BorisEnergyInvariant.compute() computes total KE, not the work-energy residual**
- **Confirmed by:** FULL_AUDIT (CRITICAL), CODEX_AUDIT (CRITICAL C-1)
- **Severity:** CRITICAL (both audits agree)
- File: `src/sim_debugger/parsec/invariants.py:114-126`
- Problem: `compute()` returns `0.5 * sum(m * |v|^2)` -- identical to `TotalEnergyInvariant`. The docstring correctly says "kinetic energy change must equal work done by electric field" but the implementation never reads E_field or particle positions. The `check()` method then flags any KE change, which is not the Boris-specific invariant.
- Expected fix: Rewrite `compute()` to return the accumulated work-energy residual. The invariant value should be `KE_current - KE_initial - W_accumulated` where `W_accumulated = sum over all steps of sum_i(q_i * E_i . dr_i)`. This requires the state to provide `E_field` (per-particle or interpolated to particle positions), `charges`, and `positions`. Update `applicable()` to require these arrays. Write a test using a known Boris pusher trajectory (constant B field, no E field) and verify the residual stays below 1e-10, then add a test with wrong E field that confirms the residual grows.

### Group CRIT-B: Gauss's Law Divergence Stencil (parsec/invariants.py)

**CRIT-02: GaussLawInvariant uses central differences instead of staggered-grid forward differences**
- **Confirmed by:** FULL_AUDIT (CRITICAL)
- File: `src/sim_debugger/parsec/invariants.py:202-208`
- Problem: `np.gradient(E_d, dx, axis=d)` uses central differences `(E[i+1] - E[i-1]) / (2*dx)`. PIC codes using the Yee grid store E on cell faces; the correct discrete divergence uses forward differences: `(E[i+1] - E[i]) / dx`.
- Expected fix: Replace the `np.gradient` call with explicit forward-difference slicing. Add a `staggered_grid` parameter (default `True`) to allow both staggered and collocated grids. When `staggered_grid=False`, use central differences. Write tests verifying correct divergence on a Yee grid.

### Group CRIT-C: Momentum Invariant Timestep Reporting (core/invariants.py)

**CRIT-03: LinearMomentumInvariant and AngularMomentumInvariant hardcode timestep=0 in all Violation objects**
- **Confirmed by:** FULL_AUDIT (CRITICAL)
- Files: `src/sim_debugger/core/invariants.py:396-410` (Linear), `src/sim_debugger/core/invariants.py:509-522` (Angular)
- Problem: All per-component `Violation` objects use `timestep=0, time=0.0` hardcoded. A violation at timestep 50,000 is reported as "timestep 0", making it impossible to locate the violation in time.
- Expected fix: Add `timestep: int = 0` and `time: float = 0.0` parameters to both `check()` methods. Pass these through to all `Violation` constructors. Update all callers to pass the current timestep and time.

### Group CRIT-D: Near-Zero Threshold Fallback (core/invariants.py)

**CRIT-04: _standard_check fallback compares absolute error against relative threshold**
- **Confirmed by:** FULL_AUDIT (CRITICAL)
- File: `src/sim_debugger/core/invariants.py:210-215`
- Problem: When `abs(prev_value) <= 1e-300`, `relative_error = absolute_error` (a dimensioned quantity). This is then compared to `threshold` (a dimensionless relative tolerance). For invariants that are legitimately near zero (e.g., total momentum in an equilibrium plasma), any small absolute fluctuation exceeding the threshold (in native units) triggers a false positive.
- Expected fix: Add a separate `absolute_threshold` parameter to `_standard_check`. When `prev_value` is near zero, compare `absolute_error > absolute_threshold` instead of `absolute_error > threshold`. The `absolute_threshold` should default to `None` (fall back to existing behaviour if not specified) so no existing callers break.

### Group CRIT-E: Field Energy for Non-Grid Arrays (core/invariants.py)

**CRIT-05: TotalEnergyInvariant field energy formula wrong for per-particle E arrays**
- **Confirmed by:** FULL_AUDIT (CRITICAL)
- File: `src/sim_debugger/core/invariants.py:289-305`
- Problem: `ndim_spatial = E.ndim - 1` is used as a proxy for grid dimensionality. For per-particle E with shape `(N, 3)`, `ndim_spatial = 1` and `cell_volume = dx**1` -- wrong.
- Expected fix: Add explicit detection of per-particle vs grid arrays. If `E.ndim == 2` and `E.shape[-1] <= 3`, treat it as per-particle and skip the field energy contribution. If `E.ndim > 2`, treat it as a grid field and apply the volumetric formula.

---

## HIGH Fixes

- **H-1: exec() with no sandboxing at 3 locations.** (FULL_AUDIT: HIGH, CODEX_AUDIT: M-3 confirmed)
  - `src/sim_debugger/cli/main.py:235` -- exec(code, namespace)
  - `src/sim_debugger/cli/main.py:263` -- exec(code, namespace)
  - `src/sim_debugger/instrument/import_hook.py:112` -- exec(code, module.__dict__)
  - Add security documentation, `# noqa: S102` comments at each call site.

- **H-2: `_parse_config_file` complexity E** (FULL_AUDIT: HIGH)
  - `src/sim_debugger/core/config.py:217`
  - Refactor using section-specific parsing functions and a dispatch dict.

- **H-3: Unused variables in localise/ suggest incomplete implementation.** (FULL_AUDIT: HIGH)
  - `src/sim_debugger/localise/source.py:203` -- `violation_timestep` unused
  - `src/sim_debugger/localise/temporal.py:28` -- `current_timestep` unused
  - Investigate and fix or remove with TODO comments.

---

## MEDIUM Fixes

- **M-1: 71 unused imports (F401) across `src/` and `tests/`.** (FULL_AUDIT: MEDIUM)
  - Run `ruff check --select F401 --fix .` to auto-fix.

- **M-2: mypy reports 29 errors.** (CODEX_AUDIT: M-1, new finding)
  - Representative failures in `ast_rewriter.py:74`, `config.py:209`, JAX/Scipy backends.
  - Fix all type errors.

- **M-3: ruff check reports 196 total issues.** (CODEX_AUDIT: M-2, new finding)
  - Categories: 74 N806 (physics variables, suppress), 71 F401 (unused imports), 15 I001 (unsorted imports), 13 F841 (unused variables), 6 UP045, 4 E501, 3 F541, 3 UP035, 2 E741, 2 N803, 2 UP015, 1 UP036.

- **M-4: 7 B904 raise-without-from-inside-except.** (FULL_AUDIT: MEDIUM)
  - Change each `raise X(...)` inside `except` to `raise X(...) from err`.

- **M-5: 2 E741 ambiguous variable names.** (FULL_AUDIT: MEDIUM)
  - Rename `l` to `length`/`label`, `O` to `output`.

- **M-6: 3 F541 f-string-missing-placeholders.** (FULL_AUDIT: MEDIUM)
  - Add placeholders or convert to regular strings.

---

## LOW Fixes (Suppress / Defer)

- **L-1: 74 N806 physics variable naming violations.** (FULL_AUDIT: LOW, false positives)
  - Variables like `E`, `B`, `q` are standard physics notation. Suppress via ruff per-file-ignores.

- **L-2: 7 B904 raise-without-from.** (FULL_AUDIT: LOW)
  - Fix with `raise X from Y`.

---

## Recommended Fix Priority

1. **Immediately:** CRIT-01 through CRIT-05 (scientific correctness)
2. **This week:** H-1 through H-3 (exec docs, config refactor, localise vars)
3. **This sprint:** M-1 through M-6 (imports, mypy, ruff, B904, E741, F541)
4. **Next sprint:** L-1, L-2 (suppress N806, fix B904)

After all fixes, run the complete test suite, confirm zero regressions, and write FIXES_APPLIED.md summarising every change.
