# CHECKPOINT_0_AUDIT.md -- sim-debugger

**Audit Date:** 2026-03-29
**Auditor:** Claude (automated planning audit)
**Checkpoint:** 0 -- Planning Complete

---

## Audit Checklist

### [PASS] CLAUDE.md covers all required sections

Verified sections present in `/Users/alistair/Documents/claude/projects/sim-debugger/CLAUDE.md`:
- Project name, one-line description, and goals
- Tech stack with justification (table with 10 technology choices, each justified)
- Directory structure overview (full tree)
- Development philosophy and constraints (6 principles)
- "Do not begin coding until PLANNING.md has been reviewed and approved" (Constraint #1)
- Agent instructions for sub-agent division of work (4 agents with owned directories and coordination rules)
- Audit checkpoint schedule (7 checkpoints, 0 through 6)
- Known ambiguities and open questions (7 open questions)
- External dependencies and APIs required (runtime and dev dependencies listed with versions)
- Skills identified from GitHub search (30+ resources with URLs across 6 categories)

**Result: PASS**

---

### [PASS] PLANNING.md has a realistic MVP that could be built by one developer

Verified in `/Users/alistair/Documents/claude/projects/sim-debugger/PLANNING.md`:
- MVP is clearly scoped: CLI tool, AST instrumentation, 8 invariants (5 built-in + 3 PARSEC), temporal localisation, physics-language explanations, NumPy backend only
- MVP explicitly excludes TUI dashboard, JAX/SciPy backends, spatial/source localisation, IDE plugin, cloud platform
- Timeline is 4 weeks for one developer, broken into week-by-week deliverables
- MVP success criteria are specific and measurable (5 criteria including performance target and false-negative requirement)
- Phase breakdown has 6 phases with clear scope boundaries and dependencies
- Data models are defined (7 core data classes with fields)
- API surface is defined (public Python API + CLI commands)
- Risks are identified with likelihood/impact/mitigation (7 risks)
- Parallelisation map defines 4 parallel tracks with a dependency graph
- PARSEC integration is explicitly planned as first-class (Section 12, with Boris pusher sub-steps, charge conservation, racetrack coil specifics, and 6 PARSEC-specific benchmarks)
- Invariant library plan covers what is monitored, how it is computed, and how violations are detected (Section 3)
- Instrumentation system plan covers all three mechanisms with AST rewriter design (Section 4)
- Localisation algorithm plan covers temporal, spatial, and source localisation (Section 5)
- Explanation generator plan covers template structure and violation pattern matching (Section 6)
- User interface plan includes mock CLI output and dashboard panel layout (Section 7)

**Result: PASS**

---

### [PASS] AUDIT_PLAN.md has project-specific (not generic) audit criteria

Verified in `/Users/alistair/Documents/claude/projects/sim-debugger/AUDIT_PLAN.md`:
- Primary audit axis is physics correctness, explicitly stated as more important than all other criteria
- False negative rate is called out as the critical metric ("must be zero for all known violation classes")
- Invariant validation protocol has 5 specific test types (analytical, correct simulation, known-bug, threshold sensitivity, edge case)
- Known-bug benchmark suite lists 14 specific simulations (B01-B14) with exact bugs and expected violations -- these are physics-specific, not generic software tests
- Physics expert review checkpoint explicitly names Alistair and lists 6 specific items he must review
- Behavioural equivalence testing is specific to AST rewriting (instrumented code must produce bit-identical results)
- Performance audit defines overhead budgets per mode (1%, 5%, 20%) with specific measurement methods
- Explanation quality audit requires physicist review of template accuracy
- Explanation coverage matrix maps invariant types to violation patterns
- Checkpoint schedule has 7 gates with physics-specific criteria at each
- PARSEC-specific items appear throughout: Boris pusher sub-steps, charge conservation stencil, racetrack symmetry

**Result: PASS**

---

### [PASS] Tech stack choices are justified and optimal

Verified in CLAUDE.md tech stack table:
- **Python 3.11+**: Correct choice. Target users write Python simulations. AST module in stdlib. No alternative language makes sense.
- **ast (stdlib) + astor**: Correct. ast is the standard for Python AST manipulation. astor provides the read-back-to-source capability that ast alone lacks. astmonkey was considered but astor is more mature.
- **NumPy/SciPy/JAX**: These are the three dominant numerical backends in the computational physics community. Supporting all three covers the vast majority of target users.
- **Typer + Rich**: Modern, type-hint-based CLI framework with excellent terminal formatting. Alternatives (Click, argparse) are either lower-level or less modern. Typer is built on Click so compatibility is assured.
- **Textual**: The leading Python TUI framework for real-time dashboards. Backed by Textualize (same team as Rich), ensuring compatibility.
- **pytest + hypothesis**: Standard testing stack. Hypothesis is particularly well-suited for property-based testing of invariant monitors.
- **hatchling**: Modern PEP 621 build backend. Preferable to setuptools for new projects.

No unnecessary dependencies. No over-engineering. Stack is minimal for the requirements.

**Result: PASS**

---

### [PASS] No implementation code exists anywhere

Verified by inspecting all 33 Python files in the project. Every `.py` file begins with a comment containing "Placeholder only. No implementation code." No file contains any Python statements, class definitions, function definitions, or import statements. The `pyproject.toml` file is also placeholder-only (comments only, no TOML configuration).

Files checked:
- `src/sim_debugger/__init__.py` and all 15 module files under `src/`
- All `__init__.py` files under `tests/`
- All `__init__.py` files under `examples/`
- `pyproject.toml`
- `docs/.gitkeep`

**Result: PASS**

---

### [PASS] All GitHub searches were actually performed

Six web searches were performed and results recorded in CLAUDE.md:

1. **Claude Code skills and CLAUDE.md templates**: Found 5 relevant repositories (awesome-claude-code-toolkit, Claude Code Ultimate Guide, awesome-claude-skills, claude-code-templates, awesome-claude-code)
2. **Python AST manipulation and instrumentation**: Found 5 relevant resources (ast stdlib, astor, astmonkey, awesome-python-ast, profilehooks)
3. **Physics simulation debugging and invariant checking**: Found GitHub topics for physics-simulation, numerical-simulations, and related projects. Also found pyHamSys and SIMPLE for conservation checking.
4. **PARSEC / PIC / Boris pusher**: Found 5 relevant repositories (PIConGPU, PICSAR, boris-method, boris-algorithm, plasma-simulations-by-example)
5. **CLI and TUI frameworks**: Found Typer, Rich, Textual with current version information and adoption data
6. **JAX/NumPy/SciPy instrumentation**: Found JAX tracing architecture, JAX-MD for simulation, and JAX NumPy/SciPy scope documentation
7. **Symplectic integrators and conservation**: Found pyHamSys, SIMPLE, Neural Symplectic Integrator, and SciPy symplectic solver discussion

All findings are recorded with URLs in the "Relevant Skills and Resources from GitHub Search" section of CLAUDE.md. A total of 30+ resources with URLs are documented.

**Result: PASS**

---

### [PASS] CLAUDE_AI_PROJECT_BRIEF.md is complete and self-contained

Verified in `/Users/alistair/Documents/claude/projects/sim-debugger/CLAUDE_AI_PROJECT_BRIEF.md`:
- Suggested Claude.ai Project name and description provided
- Full context document for the Claude.ai Project is self-contained (does not require reading other files to understand the project)
- Covers: what the project is, why it exists, who the user is, technical architecture (5 subsystems), tech stack, key directories, PARSEC connection, monetisation plan, development rules, and guidance for Claude on how to use the context
- Instructions for the human include: step-by-step project setup in Claude.ai, which files to upload, how to use the project for different tasks, and how to keep context fresh
- The document is readable and useful without any other file from the repository

**Result: PASS**

---

## Summary

| Audit Item | Result |
|---|---|
| CLAUDE.md covers all required sections | PASS |
| PLANNING.md has a realistic MVP | PASS |
| AUDIT_PLAN.md has project-specific audit criteria | PASS |
| Tech stack choices are justified and optimal | PASS |
| No implementation code exists anywhere | PASS |
| All GitHub searches were actually performed | PASS |
| CLAUDE_AI_PROJECT_BRIEF.md is complete and self-contained | PASS |

**Overall Checkpoint 0 Result: PASS (7/7)**

The project is ready to proceed to Phase 1 implementation after Alistair reviews and approves PLANNING.md.
