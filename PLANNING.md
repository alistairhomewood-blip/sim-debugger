# PLANNING.md -- sim-debugger

## Project Summary

**sim-debugger** instruments numerical simulations, monitors physically meaningful invariants in real time, and explains failures in physics language -- not stack traces. It is a Python-first CLI tool targeting computational physicists and simulation engineers who work with NumPy, SciPy, and JAX.

This project is directly connected to Alistair's PARSEC particle-in-cell simulation work. Boris pusher energy conservation, charge conservation in current deposition, Lorentz force correctness, and racetrack coil field symmetries are first-class targets.

---

## 1. MVP Definition

### What the MVP Does

The MVP is a CLI tool that a computational physicist can point at their existing Python simulation script and get real-time conservation law monitoring with physics-language violation reports.

**MVP user story:** "I run `sim-debugger run my_boris_pusher.py` and see a live terminal display of total energy, total momentum, and charge conservation. When the Boris pusher violates energy conservation at timestep 4400, sim-debugger tells me 'Total energy increased by 3.2% at timestep 4400; violation localised to velocity rotation step; consistent with dt exceeding Boris stability limit.'"

### MVP Scope (In)

1. CLI command `sim-debugger run <script.py>` that instruments and monitors a simulation
2. AST-based instrumentation of timestep loops (no user code modification required)
3. At least 5 built-in invariant monitors: total energy, linear momentum, angular momentum, charge conservation, particle count
4. 3 PARSEC-specific invariant monitors: Boris pusher energy, charge conservation (div E = rho/eps0), Lorentz force correctness
5. Temporal localisation: identify the first violating timestep
6. Physics-language explanation for each violation type
7. Rich terminal output (formatted violation reports)
8. NumPy backend support (the dominant case)
9. Decorator-based opt-in instrumentation (`@monitor`, `@timestep`)

### MVP Scope (Out)

- Textual TUI dashboard (Phase 2)
- JAX backend (Phase 2)
- SciPy ODE solver integration (Phase 2)
- Spatial localisation (Phase 2)
- Source-code localisation (Phase 3)
- IDE plugin (Phase 4)
- Cloud-hosted version (Phase 5)
- Auto-detection of which invariants to monitor (Phase 3)

### MVP Success Criteria

1. Can instrument and monitor a Boris pusher simulation without modifying its source code
2. Detects a known energy conservation violation in a deliberately buggy Boris pusher
3. Produces a physics-language explanation that a computational physicist finds useful
4. Performance overhead < 10% for a 10,000-timestep simulation (target < 5% for release)
5. Zero false negatives on the benchmark suite of known-buggy simulations

---

## 2. Phase Breakdown

### Phase 1: Core Engine + CLI (MVP) -- Target: 4 weeks

**Week 1: Core data models and invariant library**
- Define `SimulationState`, `Violation`, `Invariant` protocols/dataclasses
- Implement 5 built-in invariants (energy, linear momentum, angular momentum, charge, particle count)
- Implement 3 PARSEC invariants (Boris energy, charge conservation, Lorentz force)
- Write tests against known simulations for each invariant

**Week 2: AST instrumentation**
- Implement `SimDebugTransformer(ast.NodeTransformer)` for timestep loop detection
- Implement import hook for zero-modification instrumentation
- Implement decorator hooks (`@monitor`, `@timestep`, `@track_state`)
- NumPy backend adapter
- Test against a simple harmonic oscillator and Boris pusher

**Week 3: Temporal localisation + explanation**
- Implement state history ring buffer
- Implement temporal localisation (binary search for first violation)
- Implement template-based explanation generator
- Write explanation templates for all 8 invariant types
- Test end-to-end: instrument -> detect -> localise -> explain

**Week 4: CLI + integration + polish**
- Implement Typer CLI (`run`, `check`, `list-invariants`, `report`)
- Rich terminal output formatting
- Integration tests
- Benchmark suite (known-buggy simulations)
- Performance profiling and optimisation
- Documentation (basic usage guide)

### Phase 2: Dashboard + Extended Backends -- Target: 3 weeks

- Textual TUI dashboard with live invariant display
- JAX backend adapter (investigate tracing compatibility)
- SciPy ODE solver integration (solver callback hooks)
- Spatial localisation (per-cell/per-particle contribution analysis)
- `sim-debugger watch` command for attaching to running simulations
- Racetrack coil symmetry invariant

### Phase 3: Advanced Features -- Target: 3 weeks

- Source-code localisation (AST source mapping + bisection)
- Invariant auto-detection (heuristic-based)
- Multi-physics coupling support
- Configuration file support (`.sim-debugger.toml`)
- Violation history and trend analysis
- Export to JSON/HDF5 for post-analysis

### Phase 4: IDE Plugin -- Target: 4 weeks

- VS Code extension: inline violation annotations
- Violation gutter markers at source locations
- Hover explanations in physics language
- Integration with VS Code's debugging protocol

### Phase 5: Cloud Platform -- Target: 6+ weeks

- Persistent monitoring history
- Team dashboards
- Alerting (email, Slack, webhook)
- Comparison across simulation runs
- Authentication and billing

---

## 3. Invariant Library Plan

### 3.1 What Physical Invariants Are Monitored

Each invariant is a scalar quantity that should be conserved (constant) over time in a correctly implemented simulation. The invariant library defines how to compute each quantity from the simulation state and how to detect violations.

#### Built-in Invariants

| Invariant | Formula | Applicable When | Violation Indicates |
|---|---|---|---|
| Total energy | E_kinetic + E_potential + E_field | Any Hamiltonian system | Numerical instability, wrong integrator, timestep too large |
| Linear momentum | sum(m_i * v_i) + field momentum | No external forces | Force computation error, boundary condition bug |
| Angular momentum | sum(r_i x (m_i * v_i)) | Rotationally symmetric system | Symmetry-breaking bug, incorrect force direction |
| Charge conservation | div(E) - rho/eps_0 = 0 | Electromagnetic PIC | Current deposition error, particle boundary bug |
| Particle count | N_particles | Closed system | Spurious particle creation/destruction at boundaries |

#### PARSEC-Specific Invariants

| Invariant | Formula | Context | Violation Indicates |
|---|---|---|---|
| Boris energy | E_k(t+dt) - E_k(t) = q*E_applied * displacement | Boris pusher | Bug in half-step structure, dt exceeds stability limit |
| Lorentz force | F_applied == q(E + v x B) | Any EM particle code | Wrong field interpolation, sign error, missing factor |
| Racetrack symmetry | B(x,y,z) symmetries under reflection/rotation | Racetrack coil geometry | Mesh error, boundary condition asymmetry |
| Gauss's law | div(E) = rho/eps_0 at every grid point | PIC with Poisson solve | Field solver error, charge deposition inconsistency |
| Phase-space volume | Liouville's theorem: df/dt = 0 along trajectories | Collisionless plasma | Non-symplectic integrator, artificial dissipation |

### 3.2 How Invariants Are Computed

Each invariant is a Python class implementing the `Invariant` protocol:

```
Protocol Invariant:
    name: str
    description: str
    applicable_heuristic(state: SimulationState) -> bool
        # Returns True if this invariant is meaningful for the given simulation
    compute(state: SimulationState) -> float
        # Compute the invariant value from the current state
    check(prev_value: float, curr_value: float, threshold: float) -> Violation | None
        # Compare consecutive values and return a Violation if the change exceeds threshold
    default_threshold: float
        # Default relative tolerance (e.g., 1e-6 for energy, 1e-12 for charge)
```

Computation uses the appropriate backend adapter (NumPy/SciPy/JAX) so that array operations are dispatched correctly regardless of the simulation's array library.

### 3.3 How Violations Are Detected

**Detection algorithm per timestep:**
1. Capture the simulation state (array views where possible, copies where mutation is detected)
2. For each active invariant, call `invariant.compute(state)` to get the current value
3. Call `invariant.check(prev_value, curr_value, threshold)` to test for violation
4. If a violation is returned, enqueue it for localisation and explanation
5. Update the state history ring buffer

**Threshold model:**
- Relative threshold: `|curr - prev| / |prev| > threshold` (default per invariant)
- Absolute threshold: `|curr - prev| > abs_threshold` (for quantities near zero)
- Trend threshold: moving average of `|delta|` over last N timesteps exceeds limit (catches gradual drift)
- User-configurable per invariant via CLI flags or config file

**Severity classification:**
- WARNING: relative error in [threshold, 10 * threshold)
- ERROR: relative error in [10 * threshold, 100 * threshold)
- CRITICAL: relative error >= 100 * threshold, or invariant becomes NaN/Inf

---

## 4. Instrumentation System Plan

### 4.1 How Existing Simulation Code Is Wrapped Without Modification

The instrumentation system provides three mechanisms, in order of intrusiveness:

#### Mechanism 1: Import Hook (zero modification)

When the user runs `sim-debugger run my_simulation.py`, the tool:
1. Installs a custom `sys.meta_path` finder/loader
2. When `my_simulation.py` (and its imports) are loaded, the loader:
   a. Reads the source file
   b. Parses it with `ast.parse()`
   c. Applies `SimDebugTransformer` to the AST
   d. Compiles and executes the transformed AST
3. The transformer identifies timestep loops and injects monitoring calls

#### Mechanism 2: Decorator Hooks (opt-in, explicit)

For users who want precise control:
```
@sim_debugger.monitor(invariants=["energy", "momentum"])
def timestep_update(state, dt):
    ...
```
The decorator wraps the function to capture state before and after each call.

#### Mechanism 3: Programmatic API (library mode)

For integration into existing workflows:
```
monitor = sim_debugger.Monitor(invariants=["energy", "charge"])
for t in range(num_timesteps):
    state = solver.step(state, dt)
    monitor.check(state, t)
```

### 4.2 AST Rewriter Design

The `SimDebugTransformer` must identify timestep loops. Heuristics for loop detection:

1. **Variable name heuristics:** Loops over variables named `t`, `timestep`, `step`, `iter`, `n` that are incremented
2. **Range pattern:** `for t in range(num_timesteps)` or `for t in range(0, T, dt)`
3. **While pattern:** `while t < T_max:` with `t += dt` in the body
4. **Call pattern:** Loops containing calls to known solver functions (`solve_ivp`, `odeint`, `.step()`)
5. **Nested loop avoidance:** Only instrument the outermost matching loop (spatial loops inside the timestep loop should not be instrumented)

The transformer injects:
- Before the loop: initialisation of the monitoring engine
- At the start of each iteration: state capture (before update)
- At the end of each iteration: state capture (after update), invariant check
- After the loop: final report generation

**Source mapping:** The transformer records a mapping from injected AST nodes to original source line numbers. This mapping is stored and used by the source-code localiser.

### 4.3 Backend Adapters

The backend adapter abstracts array operations so that invariant computations work regardless of whether the simulation uses NumPy, SciPy, or JAX.

**Backend detection:** At instrumentation time, inspect the simulation's imports to determine which array library is used. If `import jax.numpy` is found, use the JAX backend. If `import numpy` is found, use the NumPy backend.

**JAX-specific challenges:**
- JAX uses a tracing/compilation model. Our AST rewriting injects Python-level calls that may break JIT compilation.
- Strategy A: Instrument only outside JIT boundaries (before `jax.jit` is called). This limits granularity but preserves performance.
- Strategy B: Use JAX's `jax.experimental.io_callback` to inject monitoring inside JIT-compiled functions. This preserves granularity but may have overhead.
- Strategy C: Register custom JAX primitives for monitoring. Most complex but cleanest.
- Decision: Start with Strategy A for MVP; investigate B and C in Phase 2.

---

## 5. Localisation Algorithm Plan

### 5.1 Temporal Localisation

**Goal:** Given a violation detected at timestep T, find the earliest timestep T_0 where the invariant began to deviate.

**Algorithm:**
1. Maintain a ring buffer of the last N state snapshots (default N=100)
2. When a violation is detected at timestep T:
   a. Binary search the ring buffer for the last timestep where the invariant was within tolerance
   b. Report the transition timestep T_0 and the violation trajectory from T_0 to T
3. Classify the violation pattern:
   - **Sudden:** violation appears in a single timestep (T_0 = T - 1)
   - **Gradual:** violation accumulates over many timesteps (T_0 << T)
   - **Oscillatory:** violation oscillates with growing amplitude
   - **Divergent:** invariant grows exponentially (numerical instability)

**Memory management:** State snapshots can be large. The ring buffer stores full copies only for the most recent M states (default M=10) and downsampled/compressed snapshots for older states.

### 5.2 Spatial Localisation (Phase 2)

**Goal:** For field-based simulations, identify the spatial region where the violation is concentrated.

**Algorithm:**
1. Decompose the global invariant into per-cell contributions:
   - Total energy = sum over cells of (cell kinetic energy + cell field energy)
   - Charge conservation = per-cell divergence check
2. Compute the per-cell change in invariant contribution between T-1 and T
3. Identify cells where the change exceeds the expected value
4. Report the bounding box or cell indices of the anomalous region

### 5.3 Source-Code Localisation (Phase 3)

**Goal:** Given a violation at timestep T, identify which function or code block within the timestep update is responsible.

**Algorithm:**
1. The AST rewriter injects monitoring at multiple points within the timestep body (not just before/after the whole step)
2. Sub-step monitoring: instrument before and after each major operation (field solve, particle push, current deposition, boundary conditions)
3. When a violation is detected at the whole-timestep level, check the sub-step monitors to narrow down
4. If sub-step data is insufficient, offer to re-run with finer instrumentation (bisection approach)

**Boris pusher specific localisation:**
The Boris pusher has three distinct sub-steps:
1. Half E-field push: v_minus = v_n + (q*dt)/(2*m) * E
2. B-field rotation: v_plus = rotation(v_minus, B, dt)
3. Half E-field push: v_{n+1} = v_plus + (q*dt)/(2*m) * E

sim-debugger will instrument each sub-step individually when a Boris pusher is detected (by AST heuristics or user annotation), allowing the violation to be attributed to the specific sub-step.

---

## 6. Explanation Generator Plan

### 6.1 How Numeric Violations Are Translated Into Physics-Language Descriptions

The explanation generator is template-based (not LLM-based) for determinism and reproducibility. Each explanation has four parts:

1. **What happened:** "Total energy increased by {magnitude}% between timesteps {T_start} and {T_end}."
2. **Where it happened:** "Violation localised to {location_description}." (timestep, spatial region, source function)
3. **Why it likely happened:** "This pattern is consistent with {diagnosis}." (matched against known violation patterns)
4. **Suggested fix:** "Consider {suggestion}." (specific to the diagnosis)

### 6.2 Violation Pattern Matching

Each invariant type has a set of known violation patterns with associated diagnoses:

**Energy violations:**
- Sudden increase at single timestep -> "timestep exceeds stability limit" or "sign error in force computation"
- Gradual monotonic increase -> "non-symplectic integrator causing energy drift" or "artificial dissipation missing"
- Exponential growth -> "numerical instability; system may be ill-conditioned"
- Periodic oscillation with growing envelope -> "resonance with grid or timestep frequency"

**Charge conservation violations:**
- Localised to boundary cells -> "particle boundary condition not conserving charge"
- Uniform across domain -> "current deposition scheme not charge-conserving"
- At specific particles -> "interpolation weight error in charge deposition"

**Boris pusher energy violations:**
- Violation in rotation sub-step -> "B-field rotation angle too large; dt * omega_c > stability limit"
- Violation in E-field half-push -> "E-field interpolation error or wrong field at particle position"
- Violation scales with |B| -> "magnetic field magnitude exceeding Boris stability threshold"

### 6.3 Explanation Template Format

Templates are plain-text with slots:

```
Template: ENERGY_SUDDEN_INCREASE
What: "Total energy changed by {relative_error:.2%} at timestep {timestep}
       (from {prev_value:.6e} to {curr_value:.6e})."
Where: "Violation is localised to timestep {timestep}{location_suffix}."
Why: "A sudden single-timestep energy change is consistent with {diagnosis}."
Fix: "{suggestion}"
Diagnosis options:
  - If relative_error > 0: "a force computation sign error or a timestep
    exceeding the CFL/Boris stability limit"
  - If relative_error < 0: "artificial dissipation from a non-conservative
    numerical scheme"
Suggestion options:
  - "Reduce the timestep by a factor of 2 and check if the violation disappears."
  - "Verify the sign of all force terms in the particle pusher."
  - "Switch to a symplectic integrator if currently using a non-symplectic one."
```

---

## 7. User Interface Plan

### 7.1 What a Developer Sees When a Violation Is Detected

**CLI output (MVP):**

```
$ sim-debugger run my_boris_pusher.py --invariants energy,charge

sim-debugger v0.1.0 -- monitoring 2 invariants
Instrumented: my_boris_pusher.py (1 timestep loop detected, lines 42-87)

[timestep 1/10000] energy: 1.000000e+00  charge: 0.000000e+00  OK
[timestep 100/10000] energy: 1.000002e+00  charge: 1.2e-15  OK
...
[timestep 4400/10000] energy: 1.032147e+00  charge: 3.1e-04

!! VIOLATION DETECTED !!

Invariant:  Total Energy
Severity:   ERROR
Timestep:   4400
Value:      1.032147e+00 (expected: 1.000000e+00)
Change:     +3.21% (threshold: 0.01%)

Localisation:
  First deviation: timestep 4395 (gradual increase over 5 steps)
  Pattern: monotonic increase, doubling each timestep

Explanation:
  Total energy increased by 3.21% over 5 timesteps starting at t=4395.
  The violation is localised to the Boris pusher velocity rotation step.
  This exponential growth pattern is consistent with the magnetic field
  rotation angle (omega_c * dt) exceeding the Boris stability limit.

Suggested fix:
  Reduce dt so that omega_c * dt < 2 (currently omega_c * dt ~ 2.3).
  Alternatively, switch to the implicit Boris algorithm which is
  unconditionally stable for large omega_c * dt.

Simulation paused. Press Enter to continue or Ctrl+C to abort.
```

**TUI Dashboard (Phase 2):**

The Textual dashboard will have four panels:
1. **Invariant Monitor:** live sparklines showing each invariant's value over time, with colour-coded status (green = OK, yellow = warning, red = violation)
2. **Violation Log:** scrolling list of all detected violations with timestamp and severity
3. **Detail Panel:** full explanation of the currently selected violation
4. **Simulation Status:** current timestep, wall-clock time, estimated time remaining, overhead percentage

### 7.2 Output Formats

- **Terminal:** Rich-formatted text with colour coding (default)
- **JSON:** machine-readable violation reports for post-processing (`--output json`)
- **Log file:** append violations to a log file for long-running simulations (`--log violations.log`)
- **HDF5:** save full state history for offline analysis (Phase 3)

---

## 8. Data Models

### Core Data Classes

```
SimulationState:
    timestep: int
    time: float
    arrays: dict[str, ndarray]      # Named state arrays (positions, velocities, fields, etc.)
    metadata: dict[str, Any]        # Simulation parameters (dt, grid size, etc.)
    source_file: str                # Path to the source file being monitored
    source_line: int                # Current line number in the source

Invariant (Protocol):
    name: str
    description: str
    default_threshold: float
    compute(state: SimulationState) -> float
    check(prev: float, curr: float, threshold: float) -> Violation | None
    applicable_heuristic(state: SimulationState) -> bool

Violation:
    invariant_name: str
    timestep: int
    time: float
    expected_value: float
    actual_value: float
    relative_error: float
    absolute_error: float
    severity: ViolationSeverity     # WARNING | ERROR | CRITICAL
    localisation: LocalisationResult | None
    explanation: str | None

ViolationSeverity: Enum
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

LocalisationResult:
    temporal: TemporalLocalisation | None
    spatial: SpatialLocalisation | None
    source: SourceLocalisation | None

TemporalLocalisation:
    first_violation_timestep: int
    pattern: str                    # "sudden" | "gradual" | "oscillatory" | "divergent"
    violation_trajectory: list[tuple[int, float]]  # (timestep, invariant_value) pairs

SpatialLocalisation:
    region_type: str                # "cells" | "particles" | "boundary"
    indices: ndarray                # Cell or particle indices
    bounding_box: tuple | None      # (xmin, xmax, ymin, ymax, ...)

SourceLocalisation:
    file: str
    line_start: int
    line_end: int
    function_name: str
    sub_step: str | None            # e.g., "boris_rotation", "e_field_push"
```

---

## 9. API Surface

### Public Python API

```
# Core
sim_debugger.Monitor(invariants: list[str], thresholds: dict[str, float] = {})
sim_debugger.Monitor.check(state: SimulationState, timestep: int) -> list[Violation]
sim_debugger.Monitor.report() -> str

# Invariant registry
sim_debugger.list_invariants() -> list[str]
sim_debugger.get_invariant(name: str) -> Invariant
sim_debugger.register_invariant(invariant: Invariant) -> None

# Decorators
sim_debugger.monitor(invariants: list[str], threshold: float = None)
sim_debugger.timestep(func)
sim_debugger.track_state(variables: list[str])
sim_debugger.ignore(func)

# Instrumentation
sim_debugger.instrument(source_path: str) -> str  # Returns instrumented source
sim_debugger.run(script_path: str, invariants: list[str]) -> Report
```

### CLI Commands

```
sim-debugger run <script.py> [--invariants LIST] [--threshold FLOAT]
    [--mode lightweight|full] [--output json|text] [--log FILE]
sim-debugger check <script.py>
    # Static analysis: suggest applicable invariants
sim-debugger list-invariants
    # Show all registered invariants with descriptions
sim-debugger report <violations.json>
    # Re-render a saved violation report
sim-debugger dashboard <script.py> [--invariants LIST]
    # Launch TUI dashboard (Phase 2)
```

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| AST rewriting breaks user code | Medium | High | Extensive test suite of real simulation code; `@ignore` decorator as escape hatch; fallback to decorator-only mode |
| Performance overhead too high | Medium | High | "Lightweight" mode that checks invariants every N timesteps instead of every timestep; use array views not copies; profile and optimise hot paths |
| JAX tracing incompatibility | High | Medium | Start with NumPy-only MVP; investigate JAX strategies in Phase 2; worst case, JAX backend is decorator-only (no AST rewriting) |
| False negatives in invariant checking | Low | Critical | Benchmark suite of known-buggy simulations; physics expert review (Alistair) of every invariant before shipping; threshold defaults are conservative |
| Timestep loop detection fails | Medium | Medium | Multiple heuristics (variable names, range patterns, call patterns); manual annotation via `@timestep` decorator as fallback |
| Explanation templates are unhelpful | Medium | Medium | Review with target users (computational physicists); iterate on template language; provide raw numeric data alongside explanations |
| Scope creep into cloud/IDE before core is solid | Low | High | Strict phase gates; no Phase 2 work until MVP audit passes; PLANNING.md must be re-approved before each phase |

---

## 11. Parallelisation Map (Agent Work Division)

This map defines which work items can be developed in parallel by multiple agents and which have dependencies.

### Parallel Track A: Core Engine (Agent A)
- Invariant protocol and base classes
- Built-in invariants (energy, momentum, angular momentum, charge, particle count)
- PARSEC invariants (Boris energy, charge conservation, Lorentz force, racetrack symmetry)
- Violation data model
- State capture mechanism
- State history ring buffer

**Dependencies:** None (foundational layer)

### Parallel Track B: Instrumentation (Agent B)
- AST rewriter (SimDebugTransformer)
- Import hook
- Decorator hooks
- NumPy backend adapter
- Source mapping

**Dependencies:** Needs Invariant protocol from Track A (interface only, not implementation)

### Parallel Track C: Localisation + Explanation (Agent C)
- Temporal localisation algorithm
- Violation pattern classification
- Explanation templates
- Explanation generator

**Dependencies:** Needs Violation data model from Track A (interface only)

### Parallel Track D: Interface (Agent D)
- Typer CLI implementation
- Rich output formatting
- Integration wiring (connecting instrumentation -> detection -> localisation -> explanation)

**Dependencies:** Needs interfaces from Tracks A, B, C. Should be last to start, first to integrate.

### Dependency Graph

```
Track A (Core) -----> Track B (Instrument)
      |                     |
      +-----> Track C (Explain) ---> Track D (Interface)
      |                                    ^
      +------------------------------------+
```

Tracks A, B, C can start simultaneously. Track D starts once interfaces from A, B, C are defined (does not need implementation to be complete).

---

## 12. PARSEC Integration Notes

This section documents the specific connection between sim-debugger and Alistair's PARSEC work. sim-debugger is a standalone tool, but PARSEC is the primary test case and the motivation for the project.

### Boris Pusher Specifics

The Boris pusher is a second-order accurate, volume-preserving (symplectic in the magnetic-only case) particle integrator. Its three-step structure:

1. **Half E-push:** v^- = v^n + (q*dt)/(2*m) * E(x^n)
2. **B-rotation:** v^+ = rotate(v^-, B(x^n), dt) using the Boris rotation formula
3. **Half E-push:** v^{n+1} = v^+ + (q*dt)/(2*m) * E(x^n)

**Energy conservation property:** In the absence of electric fields (E=0), the Boris pusher exactly conserves |v|^2 (kinetic energy) because the rotation step preserves the velocity magnitude. With electric fields, energy is conserved to O(dt^2).

**Known failure modes:**
- If omega_c * dt > 2 (where omega_c = qB/m is the cyclotron frequency), the rotation becomes unstable
- If the electric field interpolation uses the wrong particle position, energy conservation degrades
- If the half-step structure is implemented incorrectly (e.g., full E-push instead of two half-pushes), the second-order accuracy is lost

sim-debugger must detect all three of these failure modes.

### Charge Conservation in PIC

In a PIC code, charge conservation means that the continuity equation d(rho)/dt + div(J) = 0 is satisfied discretely. This is equivalent to Gauss's law div(E) = rho/eps_0 being maintained at every timestep if it holds at t=0.

The current deposition step (depositing particle currents J onto the grid) must be charge-conserving. The standard Esirkepov scheme guarantees this, but bugs in the implementation (wrong interpolation weights, incorrect stencil) will break it.

sim-debugger checks: compute div(E) - rho/eps_0 at every grid point at every timestep. Any non-zero value (above machine precision) indicates a charge conservation violation.

### Racetrack Coil Simulations

Racetrack coils have specific geometric symmetries (mirror symmetry about the midplane, rotational symmetry for circular coils). sim-debugger can verify these symmetries in the computed magnetic field by comparing field values at symmetry-related points.

### First PARSEC Benchmarks

The benchmark suite (tests/benchmarks/) must include:
1. **Boris pusher with correct dt:** should show zero energy violation
2. **Boris pusher with dt too large:** should detect energy instability
3. **Boris pusher with wrong half-step:** should detect O(dt) energy drift instead of O(dt^2)
4. **PIC code with charge-conserving deposition:** should show zero charge violation
5. **PIC code with non-conserving deposition:** should detect charge violation
6. **Lorentz force with sign error:** should detect force inconsistency

---

## 13. Open Design Decisions

These decisions are deferred and will be resolved during implementation of the relevant phase:

1. **State capture strategy:** Copy-on-capture vs view-with-mutation-detection. Views are faster but may be invalidated by in-place operations. Need to profile both approaches.

2. **Ring buffer size:** Default N=100 states. May need to be configurable for large simulations where states are multi-GB. Consider a tiered approach: full copies for last 10, checksums for last 1000.

3. **Invariant auto-detection heuristics:** Analyse imports (numpy -> likely numerical simulation), function signatures (arguments named `E`, `B`, `v`, `x` suggest electromagnetic PIC), and array shapes (3D arrays suggest field data). Exact heuristics TBD in Phase 3.

4. **Configuration file format:** `.sim-debugger.toml` in the project root or user home. Schema TBD.

5. **JAX instrumentation strategy:** Strategy A (outside JIT) vs B (io_callback) vs C (custom primitives). Requires experimentation in Phase 2.

6. **GPU array handling:** When simulation state lives on GPU (JAX, CuPy), invariant computation should also run on GPU to avoid device transfer overhead. Need backend adapters to handle device placement.
