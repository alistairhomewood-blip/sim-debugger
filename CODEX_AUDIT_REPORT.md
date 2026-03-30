# CODEX Audit Report: sim-debugger
## Date: 2026-03-29
## Stack: Python instrumentation + invariant monitoring

---

### 1. CRITICAL Issues

- **C-1: `BorisEnergyInvariant` does not implement the work-energy relation it claims to monitor.**
  The class docstring says it checks `delta_E_kinetic = q * E_dot_displacement`, but `compute()` only returns total kinetic energy and `check()` only compares successive scalars. The optional `E_at_particles`, `charges`, and `positions` data are never used in the implementation (`src/sim_debugger/parsec/invariants.py:83-131`).

### 2. HIGH Issues

- No additional high-severity issue exceeded the scientific defect above in this pass.

### 3. MEDIUM Issues

- **M-1: `mypy src` fails with 29 errors.**
  Representative failures are in `src/sim_debugger/instrument/ast_rewriter.py:74`, `src/sim_debugger/core/config.py:209`, and the JAX/Scipy backends.

- **M-2: `ruff check src tests` fails with 196 issues.**

- **M-3: Bandit reports medium-severity `exec()` usage in runtime instrumentation paths.**
  Confirmed in `src/sim_debugger/cli/main.py:235`, `:263`, and `src/sim_debugger/instrument/import_hook.py:112`.

### 4. LOW Issues

- **L-1: I could not obtain a stable final `pytest -q` exit status from this shell environment.**
  Multiple runs collected `310` tests and began executing, but the unified exec session terminated without returning a final summary. I am treating the suite result as inconclusive rather than fabricating pass/fail status.

### 5. VERIFIED / CONFIRMED

- The Boris-energy mismatch above is visible directly in source and does not depend on test behavior.

### 6. Tool Output Summary

| Tool | Result |
|------|--------|
| `pytest -q` | inconclusive in this shell session; collection reached `310` tests |
| `mypy src` | `29 errors` |
| `ruff check src tests` | `196 errors` |
| `bandit -q -r src` | 3 medium `exec` findings + 1 low plugin exception finding |

### 7. Notes

- The previous audit already identified major scientific correctness problems. The strongest independently confirmed one from this pass is the Boris-energy invariant mismatch above.
