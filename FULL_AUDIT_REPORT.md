# Full Audit Report: sim-debugger

**Date:** 2026-03-29
**Stack:** Python CLI, NumPy, Textual TUI
**Codebase:** 37 Python files, 20 test files

---

## Executive Summary

The sim-debugger has **5 CRITICAL scientific correctness bugs** in its core invariant monitoring library. The Boris energy invariant computes total kinetic energy (identical to TotalEnergyInvariant) instead of the work-energy residual, making it useless for detecting Boris pusher errors. Gauss's law uses central differences on what is expected to be a Yee staggered grid, producing wrong divergence values. Both LinearMomentumInvariant and AngularMomentumInvariant hardcode `timestep=0` in all violation reports, making violations untraceable. The near-zero threshold fallback in `_standard_check` compares an absolute error against a relative threshold, which is dimensionally inconsistent.

The tool also executes user-supplied Python scripts via `exec()` by design, with a critically complex configuration parser (`_parse_config_file` rated E by Radon), and unused variables in the localization module that likely indicate incomplete logic.

---

## Findings by Severity

### CRITICAL (Scientific Correctness)

#### 1. BorisEnergyInvariant.compute() Returns KE, Not Work-Energy Residual

- **File:** `src/sim_debugger/parsec/invariants.py:114-126`
- **Finding:** `compute()` returns `0.5 * sum(m * |v|^2)` -- total kinetic energy. The description says "kinetic energy change must equal work done by electric field", but the implementation never uses the electric field or particle positions. The `check()` method then calls `_standard_check(prev_KE, curr_KE, thr)`, which flags any KE change. This is identical to `TotalEnergyInvariant` and does not test the Boris-specific invariant.
- **Correct behaviour:** The Boris energy invariant should compute the work-energy residual: `residual = ΔKE - W_E` where `W_E = sum_i(q_i * E_i · Δr_i)`. A correctly running Boris pusher has `residual ≈ 0` at each half-step. The current implementation cannot detect Boris pusher bugs that change KE in a way that happens to conserve total energy.
- **Impact:** This invariant is the primary detector for Boris pusher numerical errors. As implemented, it is effectively disabled.

#### 2. GaussLawInvariant Uses Wrong Divergence Stencil for Yee Grid

- **File:** `src/sim_debugger/parsec/invariants.py:202-208`
- **Finding:** `np.gradient(E_d, dx, axis=d)` computes central differences: `(E[i+1] - E[i-1]) / (2*dx)`. PIC codes using the Yee staggered grid store E-components on cell faces, not cell centres. The correct discrete divergence for a Yee grid is a forward difference: `(E[i+1] - E[i]) / dx`. Using central differences on a staggered grid gives the wrong divergence at every point and may mask real Gauss's law violations or produce spurious ones.
- **Impact:** Gauss's law monitoring produces incorrect results for any simulation using a standard Yee grid.

#### 3. TotalEnergyInvariant Field Energy Formula Wrong for Per-Particle Arrays

- **File:** `src/sim_debugger/core/invariants.py:289-305`
- **Finding:** `ndim_spatial = E.ndim - 1` is used to determine whether to treat E as a grid field. For a 3D grid field `(Nx, Ny, Nz, 3)`, `ndim_spatial = 3` and `cell_volume = dx**3` is correct. But if E is stored as a per-particle field `(N, 3)`, `ndim_spatial = 1` incorrectly treats it as a 1D grid and multiplies by `dx**1`. The formula `0.5 * eps_0 * sum(E*E) * cell_volume` is also only valid for uniform grids; non-uniform grids require summing `0.5 * eps_0 * |E_i|^2 * V_i` per cell.
- **Impact:** Field energy is computed incorrectly for PIC codes that store per-particle fields or use non-uniform grids.

#### 4. LinearMomentumInvariant and AngularMomentumInvariant Hardcode timestep=0

- **Files:** `src/sim_debugger/core/invariants.py:396-410` (Linear), `src/sim_debugger/core/invariants.py:509-522` (Angular)
- **Finding:** All `Violation` objects created in the per-component `check()` paths hardcode `timestep=0, time=0.0`. A violation at timestep 47,000 is reported as "occurred at timestep 0". Additionally, `_prev_components` and `_curr_components` are instance variables mutated by `compute()`. If `check()` is called multiple times per step (or skipped), the component arrays desync from the scalar `prev_value`/`curr_value` passed to `check()`.
- **Impact:** Violation reports are useless for locating when a momentum violation occurred. The mutable state creates potential for silent misreporting if the calling pattern deviates from strict alternating compute/check.

#### 5. _standard_check Near-Zero Fallback is Dimensionally Inconsistent

- **File:** `src/sim_debugger/core/invariants.py:210-215`
- **Finding:** When `abs(prev_value) <= 1e-300`, the code sets `relative_error = absolute_error` and then compares to `threshold` (a dimensionless relative tolerance like `1e-6`). This assigns a dimensioned absolute error (in the units of the invariant) to a dimensionless variable, then compares it against a dimensionless threshold. For a neutral plasma where total momentum is legitimately near zero, any fluctuation of `|Δp| > threshold` in native units (e.g., kg·m/s) triggers a false positive regardless of physical significance.
- **Impact:** False-positive violations for any invariant that is legitimately near zero, particularly total momentum in equilibrium simulations.

---

### HIGH

#### 1. exec() Without Sandboxing (3 Locations -- By Design)

- **Locations:**
  - `cli/main.py:234-235` -- Compiles and exec's a user-supplied Python script file after AST transformation (run command)
  - `cli/main.py:262-263` -- Same pattern for the `check` command
  - `instrument/import_hook.py:111-112` -- exec's AST-transformed module code during import hooking
- **Finding:** The tool uses `exec()` to execute user-supplied Python code. This is the tool's core functionality: it instruments simulation scripts by transforming their AST and executing them with monitoring hooks injected.
- **Risk:** Any script run through sim-debugger has full, unsandboxed access to the host system. This is expected for a local developer tool (the user explicitly chooses to run their own script), but the risk should be documented. Specific concerns:
  - If sim-debugger is ever used in a multi-tenant context (shared server, CI pipeline with untrusted scripts), the lack of sandboxing becomes a critical vulnerability.
  - If the AST transformation introduces unintended code paths, the exec'd code may behave differently than the user expects.
- **Recommendation:** This does not need to be "fixed" but must be:
  1. Documented prominently in the README and CLI help text.
  2. Annotated with `# noqa: S102` and explanatory comments at each call site.
  3. Consider adding an optional `--sandbox` flag for future multi-tenant use (e.g., using `subprocess` with resource limits or `RestrictedPython`).

#### 2. _parse_config_file Rated E (Very High Complexity)

- **Location:** `config.py:_parse_config_file`
- **Finding:** Rated E by Radon, indicating extremely high cyclomatic complexity. This function likely handles many configuration options with deeply nested conditionals.
- **Risk:** High complexity makes the function difficult to test, maintain, and debug. Configuration parsing errors are hard to isolate. If the config is parsed incorrectly, the instrumentation may behave unpredictably -- monitoring the wrong invariants, applying incorrect thresholds, or silently ignoring configuration sections.
- **Recommendation:** Refactor using a data-driven schema approach or Pydantic validation. Extract each config section into its own parsing function. Target: reduce to C or better.

#### 3. D-Rated Functions: run Command and _analyse_state

- **Location:** CLI `run` command handler and `_analyse_state` function
- **Finding:** Both rated D complexity by Radon. The `run` command likely handles argument parsing, script loading, AST transformation, and execution in a single function. `_analyse_state` likely inspects multiple variable types with branching logic.
- **Recommendation:** Extract discrete stages into separate functions. For `_analyse_state`, use a dispatch pattern mapping variable types to analysis functions.

#### 4. Unused Variables in localise/ Module -- Potential Bugs

- **Location:** `localise/` directory -- `violation_timestep` and `current_timestep` are computed but unused
- **Finding:** These variables are calculated in the localization logic but never used in return values, comparisons, or storage.
- **Risk:** This strongly suggests incomplete implementation or a bug. If `violation_timestep` is computed but not returned or stored, the localization output is missing critical timing information. If `current_timestep` is not used in a comparison, the localization logic may fail to correctly identify when a violation occurs. For a physics tool where temporal localization of conservation law violations is a core feature, this is a significant correctness concern.
- **Recommendation:** Investigate immediately. Check if these variables were meant to be included in return values, used in conditionals, or passed to downstream functions. Write targeted tests that verify localization output includes correct timestep data.

---

### MEDIUM

#### 5. 35 Unused Imports

- **Location:** Various files, with notable dead imports in `dashboard/app.py` (`on`, `work`, `Message`, `DataTable`, `Sparkline`, `Static` from Textual)
- **Finding:** The unused Textual widget imports in the dashboard suggest planned but unimplemented features (data tables, sparkline charts). Other unused imports across the codebase add clutter.
- **Recommendation:** For `dashboard/app.py`, determine if the widgets are planned for the next iteration and add TODO comments or remove. For all others, run `ruff check --select F401 --fix .`.

#### 6. 8 Unused Variables (Beyond localise/)

- **Location:** Various files
- **Recommendation:** Review each. Prefix intentionally unused variables with `_`. Investigate any that may indicate bugs.

#### 7. Ambiguous Variable Names (E741 -- 2 Instances)

- **Location:** 2 instances
- **Finding:** Variable names like `l`, `O`, or `I` that could be confused with digits.
- **Recommendation:** Rename to descriptive names unless they are physics variables with clear surrounding context.

---

### LOW

#### 8. Physics Variable Naming (N806 -- 22 Violations -- FALSE POSITIVES)

- **Location:** Throughout the codebase
- **Finding:** 22 N806 (non-lowercase variable name) violations for variables like `E` (electric field), `B` (magnetic field), `q` (charge). These are standard physics notation universally used in computational physics.
- **Risk:** None. These are intentional, conventional, and correct for the domain.
- **Recommendation:** Suppress globally for physics directories via Ruff per-file-ignores configuration:
  ```toml
  [tool.ruff.lint.per-file-ignores]
  "sim_debugger/physics/**" = ["N806"]
  "sim_debugger/analysis/**" = ["N806"]
  "sim_debugger/parsec/**" = ["N806"]
  ```

#### 9. Exception Chaining (B904 -- 7 Violations)

- **Location:** 7 instances of `raise` without `from` in exception handlers.
- **Recommendation:** Fix with `raise X from Y` for proper exception chain context.

---

## Metrics Summary

| Metric | Value |
|--------|-------|
| CRITICAL Scientific Correctness Bugs | 5 |
| Total Ruff Errors | 107 |
| exec() Usage (S102) | 3 (by design) |
| Unused Imports (F401) | 35 |
| Unused Variables (F841) | 8 + 2 potential bugs in localise/ |
| N806 Physics Variables | 22 (false positives) |
| E741 Ambiguous Names | 2 |
| B904 Raise-Without-From | 7 |
| E-rated Functions (Radon) | 1 (_parse_config_file) |
| D-rated Functions (Radon) | 2 (run command, _analyse_state) |
| Average Complexity | C |
| Bandit Findings | 3x B102 exec-used (by design) |
| Dead Code (Vulture) | 10+ items (unused Textual widgets, localise vars) |

---

## Recommended Fix Priority

1. **Immediately:** Fix BorisEnergyInvariant to compute the actual work-energy residual (parsec/invariants.py:114-126). Fix timestep=0 hardcoding in LinearMomentumInvariant and AngularMomentumInvariant (core/invariants.py:396-410, 509-522). These bugs make the invariant reports actively misleading.
2. **This week:** Fix GaussLawInvariant to use forward differences for Yee grid (parsec/invariants.py:202-208). Fix field energy formula ambiguity (core/invariants.py:289-305). Fix _standard_check near-zero fallback (core/invariants.py:210-215). Investigate unused variables in `localise/` -- likely indicate bugs in localization logic. Add security documentation for `exec()` usage.
3. **This sprint:** Refactor `_parse_config_file` from E to C or better. Refactor D-rated `run` command and `_analyse_state`. Clean up unused imports.
4. **Next sprint:** Suppress N806 false positives via Ruff config. Fix B904 exception chaining. Evaluate and clean up dead Textual widget imports in the dashboard module.
5. **Ongoing:** Consider optional sandboxing mechanism for future multi-tenant use cases.
