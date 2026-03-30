# CLAUDE.md -- sim-debugger

## Project Overview

**Name:** sim-debugger
**One-line:** Instrument a simulation, monitor physically meaningful invariants in real time, explain failures in physics language not stack traces.
**Core user:** Computational physicists, simulation engineers, anyone running numerical solvers.

## Goals

1. Automatically identify which physical invariants to monitor for a given simulation (energy, momentum, charge, symmetry).
2. Non-intrusively instrument existing simulation code via AST rewriting and decorator hooks.
3. Localise conservation-law violations to specific spatial regions and timesteps.
4. Generate physics-language explanations of failures -- not stack traces, but statements like "Total energy increased by 3.2% at timestep 4400; dominant contribution from Boris pusher half-step rotation."
5. Provide a real-time terminal dashboard and CLI for monitoring running simulations.
6. First-class support for PARSEC-related simulations (Boris pusher, racetrack coil, particle trajectory codes).

## Tech Stack

| Layer | Choice | Justification |
|---|---|---|
| Language | Python 3.11+ | Dominant language in computational physics; AST module in stdlib; target user base writes Python |
| AST manipulation | `ast` (stdlib) + `astor` | Rewrite simulation functions at import time to inject monitoring hooks without modifying user code |
| Decorator hooks | Custom decorators | Lightweight opt-in instrumentation for users who prefer explicit annotation |
| Numerics target | NumPy, SciPy, JAX | The three dominant numerical backends in the target community |
| CLI framework | Typer + Rich | Modern CLI with type hints; Rich provides formatted terminal output and live displays |
| TUI dashboard | Textual | Real-time terminal dashboard for monitoring invariants during simulation runs |
| Testing | pytest + hypothesis | Property-based testing is natural for verifying invariant monitors against known physics |
| Packaging | hatchling / hatch | Modern Python packaging; PEP 621 compliant |
| Docs | MkDocs + mkdocstrings | Standard for Python tool documentation |

## Directory Structure Overview

```
sim-debugger/
  CLAUDE.md                  -- This file
  PLANNING.md                -- Detailed project plan (must be reviewed before coding)
  AUDIT_PLAN.md              -- Audit criteria and schedule
  CLAUDE_AI_PROJECT_BRIEF.md -- Context document for Claude.ai project
  pyproject.toml             -- Project metadata (placeholder only)
  src/
    sim_debugger/
      __init__.py
      core/                  -- Invariant definitions, violation detection engine
      instrument/            -- AST rewriter, decorator hooks, import hooks
      localise/              -- Violation localisation (spatial + temporal)
      explain/               -- Physics-language explanation generator
      cli/                   -- Typer CLI commands
      dashboard/             -- Textual TUI dashboard
      parsec/                -- PARSEC-specific invariants and integrations
      backends/              -- NumPy/SciPy/JAX backend adapters
  tests/
    unit/
    integration/
    benchmarks/              -- Known-buggy simulations for false-negative testing
    fixtures/                -- Simulation snapshots for reproducible tests
  docs/
  examples/
    boris_pusher/
    harmonic_oscillator/
    racetrack_coil/
```

## Development Philosophy and Constraints

1. **No code until PLANNING.md is reviewed and approved.** This document must be read, understood, and signed off by the project owner before any implementation begins.
2. **Physics correctness over software elegance.** A monitor that misses a violation (false negative) is worse than one that produces a false alarm. False negative rate must be zero for known violation classes.
3. **Non-intrusive by default.** The tool must work on unmodified simulation code. Decorators are opt-in for additional precision.
4. **Explain, don't just detect.** Every violation report must include a physics-language explanation, not just a numerical flag.
5. **Test against known physics.** Every invariant monitor must be validated against a simulation with a known, analytically-computable violation before it ships.
6. **PARSEC first.** The project is directly connected to Alistair's PARSEC work. Boris pusher energy conservation, charge conservation, and Lorentz force correctness are first-class targets, not afterthoughts.

## Agent Instructions for Sub-Agent Division of Work

When multiple Claude Code agents work on this project, divide along these boundaries:

- **Agent A (Core/Invariants):** Owns `src/sim_debugger/core/` and `src/sim_debugger/parsec/`. Responsible for invariant definitions, the violation detection engine, and PARSEC-specific monitors. Must have physics knowledge.
- **Agent B (Instrumentation):** Owns `src/sim_debugger/instrument/` and `src/sim_debugger/backends/`. Responsible for AST rewriting, decorator hooks, import hooks, and backend adapters. Must understand Python AST internals.
- **Agent C (Explanation + Localisation):** Owns `src/sim_debugger/localise/` and `src/sim_debugger/explain/`. Responsible for the violation localisation algorithm and the physics-language explanation generator.
- **Agent D (Interface):** Owns `src/sim_debugger/cli/` and `src/sim_debugger/dashboard/`. Responsible for the Typer CLI and Textual TUI dashboard.

**Coordination rules:**
- All agents must read PLANNING.md before starting work.
- Agents must not modify files outside their owned directories without coordination.
- Shared interfaces (data classes, protocols) live in `src/sim_debugger/core/` and are owned by Agent A, but changes require review from all agents.
- Every PR must include tests in `tests/` corresponding to the changed module.

## Audit Checkpoint Schedule

| Checkpoint | Gate | Criteria |
|---|---|---|
| 0 | Planning complete | All planning docs exist, no code written, tech stack justified |
| 1 | Core invariant library | At least 5 invariants implemented, each validated against known simulation |
| 2 | Instrumentation works | AST rewriter can instrument a simple NumPy simulation without modification |
| 3 | End-to-end MVP | CLI can instrument, monitor, detect, localise, and explain a violation in a Boris pusher simulation |
| 4 | Dashboard live | Textual TUI shows real-time invariant monitoring |
| 5 | PARSEC integration | All PARSEC-specific invariants working against real PARSEC simulation code |
| 6 | Release candidate | Docs, packaging, CI/CD, false-negative benchmark suite passes |

## Known Ambiguities and Open Questions

1. **JAX tracing compatibility:** JAX's own tracing mechanism may conflict with our AST rewriting approach. Need to investigate whether we must hook at the JAX primitive level instead. See JAX's tracer architecture: https://github.com/jax-ml/jax
2. **Granularity of localisation:** When a conservation law is violated, how fine-grained can we be? Timestep-level is straightforward; identifying the specific code line within a timestep update is harder and may require source-mapping through the AST rewriter.
3. **Performance overhead budget:** What is the acceptable slowdown? Computational physicists will not tolerate >5% overhead for production runs. Need a "lightweight" mode vs "full diagnostic" mode.
4. **Invariant auto-detection:** How do we automatically determine which invariants to monitor without user annotation? Heuristics based on imported libraries? Analysis of the simulation's mathematical structure?
5. **Multi-physics simulations:** Some simulations couple different physical systems (e.g., electromagnetic + fluid). How do we handle cross-domain invariants?
6. **GPU/accelerator support:** JAX and CuPy run on GPU. Can we instrument GPU kernels, or only the host-side dispatch?
7. **Relationship to PARSEC codebase:** Is sim-debugger a standalone tool that PARSEC uses, or does it become part of PARSEC? Current plan: standalone tool, PARSEC is the first and primary test case.

## External Dependencies and APIs Required

### Python Libraries (runtime)
- `numpy` >= 1.24
- `scipy` >= 1.11
- `jax` >= 0.4 (optional, for JAX backend)
- `typer` >= 0.12 (CLI framework) -- https://github.com/fastapi/typer
- `rich` >= 13.0 (terminal formatting) -- https://github.com/Textualize/rich
- `textual` >= 0.80 (TUI dashboard) -- https://github.com/Textualize/textual
- `astor` >= 0.8 (AST read/write) -- https://github.com/berkerpeksag/astor

### Python Libraries (development)
- `pytest` >= 8.0
- `hypothesis` >= 6.0
- `hatch` (build system)
- `ruff` (linter)
- `mypy` (type checking)
- `mkdocs` + `mkdocstrings` (documentation)

### No external APIs required for the core tool. The paid cloud version (future) will need:
- Cloud storage API (S3 or equivalent) for persistent monitoring history
- Authentication service for paid tier

## Relevant Skills and Resources from GitHub Search

### Claude Code Skills and Templates
- awesome-claude-code-toolkit (135 agents, 35 skills, 42 commands): https://github.com/rohitg00/awesome-claude-code-toolkit
- Claude Code Ultimate Guide (templates, workflows): https://github.com/FlorianBruniaux/claude-code-ultimate-guide
- awesome-claude-skills (ComposioHQ): https://github.com/ComposioHQ/awesome-claude-skills
- claude-code-templates: https://github.com/davila7/claude-code-templates
- awesome-claude-code: https://github.com/hesreallyhim/awesome-claude-code

### Python AST Manipulation
- Python ast module (stdlib): https://docs.python.org/3/library/ast.html
- astor (AST read/write): https://github.com/berkerpeksag/astor
- astmonkey (AST tools): https://github.com/mutpy/astmonkey
- awesome-python-ast (curated list): https://github.com/gyermolenko/awesome-python-ast
- profilehooks (decorator profiling): https://github.com/mgedmin/profilehooks

### Physics Simulation and PIC Codes
- PIConGPU (exascale PIC): https://github.com/ComputationalRadiationPhysics/picongpu
- PICSAR (modular PIC routines, Boris/Vay pushers): https://github.com/ECP-WarpX/picsar
- Boris method implementation: https://github.com/gbogopolsky/boris-method
- Boris algorithm (C/Matlab/Python): https://github.com/iwhoppock/boris-algorithm
- Plasma Simulations by Example: https://github.com/particleincell/plasma-simulations-by-example
- Boris pusher particle push explanation: https://www.particleincell.com/2011/vxb-rotation/

### Symplectic Integrators and Conservation Checking
- pyHamSys (Hamiltonian systems, energy conservation checks): https://github.com/cchandre/pyhamsys
- SIMPLE (symplectic orbit tracing, invariant conservation): https://github.com/itpplasma/SIMPLE
- Neural Symplectic Integrator: https://github.com/maxwelltsai/neural-symplectic-integrator
- SciPy symplectic solver discussion: https://github.com/scipy/scipy/issues/12690

### CLI and TUI Frameworks
- Typer (CLI framework): https://github.com/fastapi/typer
- Rich (terminal formatting): https://github.com/Textualize/rich
- Textual (TUI framework): https://github.com/Textualize/textual
- awesome-tuis (curated list): https://github.com/rothgar/awesome-tuis

### JAX Ecosystem
- JAX (composable transformations): https://github.com/jax-ml/jax
- JAX-MD (differentiable molecular dynamics): https://github.com/jax-md/jax-md
- JAX NumPy/SciPy scope: https://docs.jax.dev/en/latest/jep/18137-numpy-scipy-scope.html
