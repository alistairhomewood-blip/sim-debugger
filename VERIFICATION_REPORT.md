# VERIFICATION_REPORT

Scope: verified only the fixes claimed in `FIXES_APPLIED.md`. I did not audit for new issues.

Counts: `CONFIRMED 17` | `FAILED 0` | `UNCERTAIN 0`

## CONFIRMED
- `CRIT-01 Boris work-energy residual` is present at `src/sim_debugger/parsec/invariants.py:69-220`. The invariant now computes `(KE_current - KE_prev) - W_E` using particle charges, particle-position displacement, and the electric field at particles. From first principles and the Boris scheme described by Birdsall and Langdon, magnetic rotation does no work, so the correct discrete residual is exactly a kinetic-energy change minus electric work term. The direct tests in `tests/unit/test_parsec_invariants.py` passed, including the B-only zero-residual and wrong-work cases.
- `CRIT-02 Gauss-law Yee forward differences` is present at `src/sim_debugger/parsec/invariants.py:227-323`. For staggered grids the divergence now uses forward differences via `np.diff()`, which matches the Yee placement of field samples on cell faces; collocated grids still fall back to central differences. The dedicated parsec invariant tests passed.
- `CRIT-03 real timestep/time propagation in momentum invariants` is present at `src/sim_debugger/core/invariants.py:409-447` and `src/sim_debugger/core/invariants.py:524-560`. I exercised `LinearMomentumInvariant.check(..., timestep=7, time=0.7)` and the returned violation preserved those values.
- `CRIT-04 absolute-threshold handling for near-zero invariants` is present at `src/sim_debugger/core/invariants.py:170-247`. I exercised `_standard_check(...)` with a tiny previous value and a larger `absolute_threshold`; it correctly suppressed the false positive that the old relative comparison would have raised.
- `CRIT-05 per-particle versus grid field energy` is present at `src/sim_debugger/core/invariants.py:306-336`. I exercised the invariant with per-particle `E_field` shaped `(N, 3)` and it correctly skipped volumetric field energy instead of treating the particle axis as a grid axis.
- `H-1 exec() security documentation` is present at `src/sim_debugger/cli/main.py:236,266` and `src/sim_debugger/instrument/import_hook.py:116` via explicit `# noqa: S102` annotations and rationale comments.
- `H-2 config parser refactor` is present at `src/sim_debugger/core/config.py:210-325` via `_parse_monitor_section`, `_parse_thresholds_section`, `_parse_output_section`, `_parse_performance_section`, `_parse_plugins_section`, and `_SECTION_PARSERS`.
- `H-3 documented unused localisation parameters` is present at `src/sim_debugger/localise/source.py:224` and `src/sim_debugger/localise/temporal.py:45`.
- `M-1 unused-import cleanup` is confirmed by `ruff check src tests` succeeding cleanly.
- `M-2 import sorting` is confirmed by the same clean `ruff` run.
- `M-3 mypy cleanup` is confirmed by `mypy src` succeeding with `0` errors.
- `M-4 unused-variable cleanup` is reflected in the cleaned identifiers and underscore placeholders in `src/sim_debugger/cli/main.py`, `src/sim_debugger/core/auto_detect.py`, `src/sim_debugger/dashboard/app.py`, and `src/sim_debugger/instrument/decorators.py`; the clean `ruff` run confirms no remaining F841 issues.
- `M-5 ambiguous-name cleanup` is present in `src/sim_debugger/localise/source.py`, where the old `l` comprehension variable is now `ln`.
- `M-6 B904 raise-from fixes` are confirmed by the clean `ruff` run and by the explicit `from None` / `from exc` sites in `src/sim_debugger/cli/main.py` and `src/sim_debugger/core/history.py`.
- `M-7 line-length cleanup` is confirmed by the clean `ruff` run.
- `M-8 deprecated-pattern cleanup` is confirmed by the clean `ruff` run and the typed/mypy-clean source tree.
- `M-9 ruff per-file ignores for physics notation` is present at `pyproject.toml:118-128`, including the `B904` select list and the N806/N803 per-file ignores for the physics-heavy modules.

## FAILED
- None.

## UNCERTAIN
- None.
