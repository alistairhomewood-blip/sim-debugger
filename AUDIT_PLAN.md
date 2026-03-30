# AUDIT_PLAN.md -- sim-debugger

## Audit Philosophy

The primary audit axis for sim-debugger is **physics correctness**. A missed conservation law violation (false negative) in a production simulation can lead to days of wasted compute time and incorrect scientific conclusions. This is categorically worse than a false alarm (false positive), which merely interrupts the user.

Therefore: **the false negative rate must be zero for all known violation classes.** Every invariant monitor must be verified against a known simulation with a known, analytically computable violation before it is shipped.

Secondary audit axes, in order of priority:
1. Non-intrusiveness (instrumented code behaves identically to original code, minus monitoring overhead)
2. Performance (overhead within budget)
3. Explanation quality (physics-language descriptions are accurate and actionable)
4. Usability (CLI and dashboard are intuitive for computational physicists)

---

## 1. Physics Correctness Audit

### 1.1 Invariant Validation Protocol

Every invariant monitor must pass this validation before shipping:

1. **Analytical test:** Run a simulation with an analytically known invariant value. Verify that `invariant.compute(state)` returns the correct value to machine precision.

2. **Correct simulation test:** Run a correctly implemented simulation for 10,000+ timesteps. Verify that the invariant monitor reports zero violations.

3. **Known-bug test:** Introduce a specific, known bug into the simulation (e.g., wrong sign, missing factor, oversized timestep). Verify that the invariant monitor detects the violation within 10 timesteps of its onset.

4. **Threshold sensitivity test:** Vary the violation threshold across 5 orders of magnitude. Verify that:
   - At tight thresholds (1e-12): known bugs are detected, numerical noise may trigger warnings (acceptable)
   - At default thresholds: known bugs are detected, no false alarms on correct simulations
   - At loose thresholds (1e-2): only severe bugs are detected

5. **Edge case test:** Test with:
   - State values near zero (division-by-zero protection)
   - State values near machine epsilon
   - Very large state values (overflow protection)
   - NaN and Inf in state arrays

### 1.2 Known-Bug Benchmark Suite

The benchmark suite (`tests/benchmarks/`) must include at minimum these simulations:

| ID | Simulation | Bug Introduced | Expected Violation | Invariant Tested |
|---|---|---|---|---|
| B01 | Harmonic oscillator (Euler forward) | Non-symplectic integrator | Monotonic energy growth | Total energy |
| B02 | Harmonic oscillator (leapfrog, large dt) | dt > stability limit | Exponential energy growth | Total energy |
| B03 | Boris pusher (correct) | None | No violation | Boris energy |
| B04 | Boris pusher (dt too large) | omega_c * dt > 2 | Energy instability | Boris energy |
| B05 | Boris pusher (full E-push instead of half) | Wrong half-step structure | O(dt) energy drift | Boris energy |
| B06 | Boris pusher (wrong E-field sign) | Sign error in E | Immediate energy jump | Boris energy |
| B07 | PIC charge deposition (Esirkepov, correct) | None | No violation | Charge conservation |
| B08 | PIC charge deposition (naive, non-conserving) | Non-charge-conserving scheme | Gauss's law violation | Charge conservation |
| B09 | Lorentz force (correct) | None | No violation | Lorentz force |
| B10 | Lorentz force (wrong sign on v x B) | Sign error | Force magnitude error | Lorentz force |
| B11 | N-body (leapfrog, correct) | None | No violation | Momentum, energy |
| B12 | N-body (non-symmetric force) | F_ij != -F_ji | Momentum violation | Linear momentum |
| B13 | Particle boundary (reflecting, correct) | None | No violation | Particle count |
| B14 | Particle boundary (absorbing, leak) | Particles not removed | Particle count growth | Particle count |

**Pass criterion for each benchmark:**
- Correct simulations (B03, B07, B09, B11, B13): zero violations detected at default threshold
- Buggy simulations (all others): violation detected within 10 timesteps of bug onset

**Benchmark suite must achieve 100% pass rate before any release.**

### 1.3 Physics Expert Review Checkpoint

**Reviewer:** Alistair (project owner, PARSEC domain expert)

**Before the invariant library is finalised (before Checkpoint 2), Alistair must review:**

1. The mathematical definition of each invariant (is the formula correct?)
2. The threshold defaults (are they physically reasonable for typical simulations?)
3. The PARSEC-specific invariants (do they match the actual PARSEC code structure?)
4. The Boris pusher sub-step instrumentation (does it correctly identify the three sub-steps?)
5. The charge conservation check (is div(E) computed with the correct finite difference stencil?)
6. The explanation templates (are the physics descriptions accurate? Would a computational physicist find them helpful?)

**This review is a hard gate.** No invariant ships without Alistair's sign-off.

---

## 2. Non-Intrusiveness Audit

### 2.1 Behavioural Equivalence Tests

For each benchmark simulation, verify that the instrumented version produces bit-identical results to the non-instrumented version:

1. Run the simulation without sim-debugger; save final state to file
2. Run the simulation with sim-debugger instrumentation; save final state to file
3. Compare the two final states: they must be identical (within floating-point reproducibility)

**Exception:** If the instrumentation captures state via copies (not views), the memory layout may differ, but numerical values must be identical.

### 2.2 Import Compatibility Tests

Test that the import hook does not break:
- Standard library imports
- NumPy, SciPy, JAX imports
- User modules with unusual AST patterns (decorators, metaclasses, async, generators)
- Modules that use `exec()` or `eval()`
- Cython extensions (should be passed through unmodified)

### 2.3 Error Passthrough Tests

Verify that if the user's simulation raises an exception, it propagates correctly:
- Exception type is preserved
- Traceback points to the correct line in the original source (not the rewritten AST)
- sim-debugger does not swallow or modify exceptions

---

## 3. Performance Audit

### 3.1 Overhead Budget

| Mode | Max Overhead | Measurement Method |
|---|---|---|
| Lightweight (check every 100 timesteps) | 1% | Wall-clock time ratio over 10,000 timesteps |
| Default (check every timestep, views) | 5% | Wall-clock time ratio over 10,000 timesteps |
| Full diagnostic (check every timestep, copies + spatial localisation) | 20% | Wall-clock time ratio over 10,000 timesteps |

### 3.2 Performance Test Suite

Run each benchmark simulation at three scales:
- Small: 1,000 timesteps, 1,000 particles / 64^3 grid
- Medium: 10,000 timesteps, 100,000 particles / 256^3 grid
- Large: 100,000 timesteps, 1,000,000 particles / 512^3 grid

Measure and report:
- Wall-clock time with and without instrumentation
- Peak memory with and without instrumentation
- Overhead percentage at each scale

**Pass criterion:** overhead within budget at all three scales.

### 3.3 Memory Audit

Verify that the state history ring buffer:
- Does not grow unboundedly
- Releases old states correctly
- Total memory overhead is bounded by `ring_buffer_size * state_size`

---

## 4. Explanation Quality Audit

### 4.1 Template Accuracy Review

For each explanation template:
1. Generate the explanation for a specific violation instance
2. Have a computational physicist (Alistair) review:
   - Is the physics description accurate?
   - Is the suggested diagnosis plausible?
   - Is the suggested fix actionable?
   - Would this explanation help you find the bug faster than a stack trace?

### 4.2 Explanation Coverage Matrix

Verify that every combination of (invariant type, violation pattern) has a corresponding explanation template:

| Invariant | Sudden | Gradual | Oscillatory | Divergent |
|---|---|---|---|---|
| Total energy | Template exists | Template exists | Template exists | Template exists |
| Linear momentum | Template exists | Template exists | Template exists | Template exists |
| Angular momentum | Template exists | Template exists | Template exists | Template exists |
| Charge conservation | Template exists | Template exists | N/A | Template exists |
| Particle count | Template exists | Template exists | N/A | N/A |
| Boris energy | Template exists | Template exists | Template exists | Template exists |
| Lorentz force | Template exists | N/A | N/A | N/A |
| Racetrack symmetry | Template exists | Template exists | N/A | N/A |

Cells marked "Template exists" must have a template before the corresponding invariant ships. "N/A" means this combination is physically implausible.

---

## 5. Usability Audit

### 5.1 First-Run Experience Test

Give the tool to a computational physicist who has never seen it before. They should be able to:
1. Install sim-debugger in under 2 minutes (`pip install sim-debugger`)
2. Run it on their own simulation in under 5 minutes (`sim-debugger run my_sim.py`)
3. Understand the first violation report without reading documentation

### 5.2 CLI Audit

- All commands have `--help` text
- Error messages are clear and suggest corrective action
- Output is readable in both dark and light terminal themes
- JSON output is well-structured and documented

---

## 6. Audit Checkpoint Schedule

### Checkpoint 0: Planning Complete
- All planning documents exist and are complete
- No implementation code exists
- Tech stack choices are justified
- GitHub research has been performed
- PARSEC integration is explicitly planned

### Checkpoint 1: Core Invariant Library
- At least 5 built-in invariants + 3 PARSEC invariants implemented
- Each invariant passes the validation protocol (Section 1.1)
- Benchmark simulations B01-B14 exist (may be simple/placeholder)
- Physics expert review (Alistair) has been conducted for all invariant definitions
- **Hard gate:** No Checkpoint 2 work begins until Alistair signs off on invariant definitions

### Checkpoint 2: Instrumentation Works
- AST rewriter can instrument a harmonic oscillator and Boris pusher
- Import hook works for standard simulation modules
- Decorator hooks work as documented
- Behavioural equivalence tests pass (Section 2.1)
- Performance overhead measured and within budget for small-scale tests

### Checkpoint 3: End-to-End MVP
- CLI `sim-debugger run` works end-to-end
- Full pipeline: instrument -> detect -> localise -> explain
- All 14 benchmarks pass (100% detection rate)
- Explanation templates reviewed by Alistair
- Performance within budget at small and medium scales
- Documentation exists for basic usage

### Checkpoint 4: Dashboard and Extended Backends
- Textual TUI dashboard functional
- JAX backend works (at least decorator-only)
- Spatial localisation works for field-based simulations
- Performance within budget at large scale

### Checkpoint 5: PARSEC Integration
- sim-debugger runs on actual PARSEC simulation code
- All PARSEC-specific invariants validated against real PARSEC outputs
- Alistair confirms that the tool would have caught a real bug he has encountered

### Checkpoint 6: Release Candidate
- Full benchmark suite passes
- Documentation complete
- CI/CD pipeline configured
- Packaging works (`pip install` from wheel and source)
- Performance audit at all three scales passes
- No known false negatives

---

## 7. Regression Testing Protocol

After each code change:
1. Run the full benchmark suite (B01-B14)
2. Run behavioural equivalence tests
3. Run performance measurement (small scale only for quick feedback)
4. Any new violation type or invariant must add a corresponding benchmark

**CI gate:** PRs cannot be merged if benchmark pass rate drops below 100%.

---

## 8. Security and Safety Considerations

Since sim-debugger executes user-provided simulation code via AST rewriting and import hooks:

1. **Arbitrary code execution:** The tool is designed to run arbitrary Python code (the user's simulation). This is expected and acceptable -- it is the core use case.
2. **AST rewriting safety:** The rewriter must not introduce security vulnerabilities into the user's code. It must only inject calls to sim-debugger's monitoring functions.
3. **No network access:** The core tool must not make any network requests. Network access is reserved for the cloud version (Phase 5) and must be opt-in.
4. **No data exfiltration:** Simulation state captured by the monitoring engine must not be written to disk unless the user explicitly requests it (via `--output` or `--log`).
