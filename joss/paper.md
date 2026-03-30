---
title: 'sim-debugger: Real-Time Conservation Law Monitoring with Physics-Language Explanations for Numerical Simulations'
tags:
  - Python
  - physics simulation
  - conservation laws
  - debugging
  - particle-in-cell
  - Boris pusher
  - numerical methods
authors:
  - name: Alistair Homewood
    orcid: 0009-0004-8005-3571
    affiliation: 1
affiliations:
  - name: University of British Columbia, Vancouver, Canada
    index: 1
date: 29 March 2026
bibliography: paper.bib
---

# Summary

`sim-debugger` is a Python command-line tool that instruments numerical simulations, monitors physically meaningful invariants in real time, and reports violations in physics language rather than stack traces. Given a simulation script, `sim-debugger` automatically identifies timestep loops via AST rewriting, injects monitoring hooks without modifying user code, and checks conservation of energy, momentum, angular momentum, charge, and particle count at every timestep. When a violation is detected, the tool reports what invariant was broken, by how much, at which timestep, and provides a physics-language diagnosis with a suggested fix.

The tool targets computational physicists, simulation engineers, and students writing numerical solvers in Python with NumPy. It provides first-class support for particle-in-cell (PIC) simulation codes, including Boris pusher energy conservation, Gauss's law verification, and Lorentz force correctness checks. A plugin architecture allows users to register custom invariants for domain-specific conservation laws. `sim-debugger` is available as a `pip`-installable package and operates via a Typer-based CLI with Rich-formatted terminal output.

# Statement of Need

Conservation laws are the fundamental correctness criteria for physics simulations. A simulation that violates energy conservation, momentum conservation, or Gauss's law is producing wrong results regardless of whether it runs without errors. Yet the standard debugging tools available to computational physicists---Python tracebacks, print statements, and general-purpose debuggers---operate at the software level, not the physics level. They can tell a user that an array index is out of bounds, but not that total energy increased by 3% at timestep 4400 because the Boris pusher rotation angle exceeded its stability limit.

Currently, physicists who want to monitor conservation laws must write ad hoc checking code for each simulation. This code is typically scattered throughout the simulation loop, is not reusable across projects, and produces raw numbers rather than actionable diagnostics. For particle-in-cell codes in particular, verifying properties such as Gauss's law (div **E** = rho/epsilon_0) or the energy-work theorem for the Boris pusher requires careful implementation that is easy to get wrong [@birdsall1991plasma; @boris1970relativistic].

Existing tools address adjacent problems but not this one directly. Structure-preserving integrators [@hairer2006geometric] reduce conservation errors by construction but do not detect them when they occur. Hamiltonian system libraries such as pyHamSys [@chandre_pyhamsys] provide symplectic solvers but not runtime monitoring of arbitrary invariants across general simulation codes. `sim-debugger` fills this gap by providing a non-intrusive, reusable conservation law monitor that works on existing simulation code and explains violations in the language of the physics being simulated.

# Implementation

## Architecture

`sim-debugger` is organized into four layers: instrumentation, invariant computation, localization, and explanation.

The **instrumentation layer** uses Python's `ast` module to parse and transform user simulation scripts at import time. A custom `ast.NodeTransformer` identifies timestep loops via heuristics (variable names, `range()` patterns, `while` loop conditions) and injects monitoring calls at the start and end of each iteration. Users who prefer explicit control can use decorator hooks (`@monitor`, `@timestep`) or a programmatic API. No modification of the original simulation source code is required for the default AST-based path.

The **invariant library** implements eight invariant monitors, each as a class following a common `Invariant` protocol. Five are general-purpose (total energy, linear momentum, angular momentum, charge conservation, particle count) and three are specific to PIC simulations (Boris pusher work-energy residual, Gauss's law, Lorentz force correctness). Each invariant computes a scalar quantity from the simulation state and checks it against the previous timestep using configurable relative and absolute thresholds with three severity levels (WARNING, ERROR, CRITICAL). The Boris energy invariant computes the work-energy residual---the difference between kinetic energy change and electric field work---rather than simply tracking total kinetic energy, enabling it to detect errors specific to the Boris pusher's half-step structure [@boris1970relativistic].

The **localization module** performs temporal localization using a ring buffer of state snapshots. When a violation is detected, binary search identifies the first deviating timestep and classifies the violation pattern as sudden, gradual, oscillatory, or divergent.

The **explanation generator** uses a template-based system (no large language model dependency) with over 40 (invariant, pattern, sign) combinations mapping to specific physics-language diagnoses. For example, a sudden energy increase in a Boris pusher is diagnosed as consistent with the cyclotron frequency exceeding the stability limit, and the suggested fix is to reduce the timestep so that omega_c * dt < 2.

## Design Decisions

`sim-debugger` uses NumPy array views rather than copies where possible to minimize overhead. The `SimulationState` interface accepts named NumPy arrays and a metadata dictionary, making it compatible with any Python simulation that stores state in arrays. Backend adapters for SciPy and JAX are provided for simulations using those libraries.

# Real-World Validation

`sim-debugger` was validated against a suite of 17 self-contained reference simulations spanning four physics domains: charged particle dynamics, N-body gravity, electrostatic particle-in-cell, and multi-invariant Hamiltonian systems. Each simulation has analytically known conservation properties providing ground truth.

Of the 17 tests, 10 are correct simulations (expected to produce no violations) and 6 contain specific, analytically understood bugs that break conservation laws. One additional test documents a known edge case. The tool achieved **100% detection sensitivity** (6/6 bugs detected) and **100% specificity** (10/10 correct simulations produced no false alarms). All six violation explanations were rated as acceptable or good by keyword matching against expected physics terminology. The full 17-test suite executes in under 0.5 seconds.

The bugs detected include: forward Euler energy growth in magnetic and electromagnetic fields, an incorrect force sign in N-body gravity, Newton's third law violation from asymmetric forces (5% asymmetry), corrupted charge deposition in a 1D electrostatic PIC code, and forward Euler energy growth in a harmonic oscillator. In each case, the explanation correctly identified the violated invariant, quantified the error, and suggested a concrete fix.

A known limitation is that the per-component momentum check produces false positives when a component of momentum is near machine epsilon. This occurs because relative error is undefined for quantities that are numerically zero. The workaround is to use absolute thresholds or check only the momentum magnitude. This limitation is documented and does not affect detection of genuine conservation law violations.

# Acknowledgements

This work was supported by the University of British Columbia. The author thanks the PARSEC collaboration for motivating the PIC-specific invariants.

# References
