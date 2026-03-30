# CLAUDE_AI_PROJECT_BRIEF.md -- sim-debugger

## Suggested Claude.ai Project Configuration

**Project Name:** sim-debugger
**Project Description:** Physics simulation debugger -- instruments numerical simulations to monitor physical invariants in real time and explains violations in physics language, not stack traces.

---

## Full Context Document

Paste the entire content below into the Claude.ai Project's custom instructions / knowledge base.

---

### What is sim-debugger?

sim-debugger is a Python tool for computational physicists and simulation engineers. It instruments numerical simulation code -- without requiring modifications to that code -- to monitor physical conservation laws (energy, momentum, charge, symmetry) in real time. When a conservation law is violated, sim-debugger localises the violation to a specific timestep, spatial region, and source code location, then generates a physics-language explanation of what went wrong and suggests fixes.

### Why does this project exist?

Numerical simulations of physical systems can silently produce wrong results. A bug in a particle pusher might cause energy to grow by 0.1% per timestep -- the simulation runs to completion, produces output files, and the physicist only discovers the problem days later when analysing results. Traditional debuggers and profilers are useless here: the code has no crashes, no exceptions, no errors. The bug is purely physical -- a conservation law is violated.

sim-debugger bridges the gap between software debugging (which catches code errors) and physics validation (which catches numerical errors). It treats conservation laws as runtime assertions.

### Who is the user?

- Computational physicists running PIC (particle-in-cell) simulations
- Simulation engineers in plasma physics, fluid dynamics, and electromagnetics
- Anyone using numerical solvers from NumPy, SciPy, or JAX
- Specifically: Alistair's own PARSEC particle-in-cell simulation work (Boris pusher, racetrack coil simulations, particle trajectory codes)

### Technical Architecture

**Instrumentation layer:** Uses Python's `ast` module to rewrite simulation code at import time. A custom import hook (`sys.meta_path`) applies an `ast.NodeTransformer` that injects state-capture calls before and after each timestep iteration. Alternatively, users can opt in with decorators (`@monitor`, `@track_state`, `@timestep`).

**Invariant library:** A registry of physical invariants, each defined as a computation over the simulation state. Built-in invariants include total energy, linear momentum, angular momentum, charge density (div E = rho/epsilon_0), and discrete symmetries. PARSEC-specific invariants include Boris pusher energy conservation, charge conservation in current deposition, and Lorentz force correctness.

**Violation detection:** At each instrumented timestep, the engine computes all active invariants and compares to the previous timestep. Violations are classified by severity (warning, error, critical) based on configurable thresholds.

**Localisation:** When a violation is detected, three localisation strategies are applied:
1. Temporal: binary search over the state history to find the first violating timestep
2. Spatial: per-cell or per-particle contribution analysis to find the region of violation
3. Source: using the AST rewriter's source map to identify the responsible code block

**Explanation generator:** Template-based system that translates numerical violation data into physics-language explanations. Each invariant type has explanation templates for different violation patterns (sudden jump, gradual drift, oscillatory growth, etc.).

**User interface:** CLI tool (Typer + Rich) for running instrumented simulations and generating reports. Real-time TUI dashboard (Textual) for monitoring invariants during simulation runs.

### Tech Stack

- Python 3.11+ (core language)
- ast (stdlib) + astor (AST read/write)
- NumPy, SciPy, JAX (target simulation backends)
- Typer + Rich (CLI framework)
- Textual (TUI dashboard)
- pytest + hypothesis (testing)
- hatchling (packaging)

### Key Directories

```
src/sim_debugger/
  core/        -- Invariant definitions, violation detection, data models
  instrument/  -- AST rewriter, decorator hooks, import hooks
  localise/    -- Temporal, spatial, and source-code localisation
  explain/     -- Physics-language explanation generator and templates
  cli/         -- Typer CLI commands
  dashboard/   -- Textual TUI dashboard
  parsec/      -- PARSEC-specific invariants and integrations
  backends/    -- NumPy/SciPy/JAX backend adapters
```

### PARSEC Connection

This project is directly motivated by and connected to Alistair's PARSEC work:
- Boris pusher: the standard algorithm for advancing charged particles in electromagnetic fields. The three-step structure (half E-push, B-rotation, half E-push) has specific energy conservation properties that sim-debugger must monitor.
- Racetrack coil: a specific magnet geometry with known symmetries that sim-debugger can verify.
- Current deposition: the step in PIC codes where particle currents are deposited onto the grid. Must conserve charge exactly (Gauss's law).
- Particle boundaries: particles leaving and entering the simulation domain must not create or destroy charge.

### Monetisation Plan

- Open-source core: free, builds credibility in the physics community
- Paid cloud-hosted version: persistent monitoring history, team dashboards, alerts
- Paid IDE plugin: integrated violation display in VS Code / PyCharm

### Development Rules

1. No code is written until PLANNING.md is reviewed and approved.
2. Every invariant monitor must be validated against a known simulation with a known violation.
3. False negatives (missed violations) are unacceptable and are the primary audit axis.
4. Physics correctness is more important than software elegance.
5. PARSEC-specific invariants are first-class citizens, not afterthoughts.

### What Claude Should Do With This Context

When working on sim-debugger in Claude.ai conversations:
- Always check CLAUDE.md and PLANNING.md before making design decisions.
- When discussing invariants, be specific about the physics (e.g., "total energy" means kinetic + potential + field energy, not just kinetic).
- When discussing the Boris pusher, understand the three-step structure and why it conserves energy to second order.
- When discussing charge conservation, understand that this means div(E) = rho/epsilon_0 must hold at every timestep, and that violations typically come from the current deposition step.
- Prefer template-based explanations over LLM-generated ones for the core tool (determinism matters in debugging).
- Always think about performance overhead -- computational physicists will reject a tool that slows their simulations by more than 5%.

---

## Instructions for the Human (Alistair)

### Setting up the Claude.ai Project

1. Go to claude.ai and create a new Project.
2. **Name:** sim-debugger
3. **Description:** Physics simulation debugger -- instruments numerical simulations to monitor physical invariants in real time and explains violations in physics language, not stack traces.
4. **Custom Instructions:** Copy the entire "Full Context Document" section above (from "What is sim-debugger?" through "Always think about performance overhead") into the project's custom instructions.
5. Optionally, upload these files as project knowledge:
   - `CLAUDE.md` (from this repo)
   - `PLANNING.md` (from this repo, once reviewed)
   - Any PARSEC-related code or documentation you want Claude to reference

### Using the Project

- Start conversations in this project when working on sim-debugger design, architecture, or implementation questions.
- Claude will have persistent context about the project's goals, tech stack, architecture, and your PARSEC work.
- For implementation work, continue using Claude Code with the CLAUDE.md file in the repo root.
- The Claude.ai Project is best for: design discussions, physics invariant brainstorming, architecture reviews, explanation template drafting.
- Claude Code is best for: writing and editing code, running tests, managing files.

### Keeping Context Fresh

- When PLANNING.md is updated, re-upload it to the project knowledge.
- After each audit checkpoint, update the project description with the current phase.
- If you add new invariant types or change the architecture, update the custom instructions.
