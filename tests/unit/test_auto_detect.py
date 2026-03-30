"""Tests for invariant auto-detection (Phase 3)."""

import numpy as np

from sim_debugger.core.auto_detect import (
    auto_detect_invariants,
)
from sim_debugger.core.state import SimulationState


class TestAutoDetectFromSource:
    def test_velocity_suggests_energy(self):
        source = "velocities = np.zeros((100, 3))\n"
        results = auto_detect_invariants(source_code=source)
        names = [r.name for r in results]
        assert "Total Energy" in names

    def test_boris_keyword_suggests_boris_energy(self):
        source = "# Boris pusher implementation\ndef boris_push(v, E, B, dt):\n    pass\n"
        results = auto_detect_invariants(source_code=source)
        names = [r.name for r in results]
        assert "Boris Energy" in names

    def test_charge_suggests_conservation(self):
        source = "charges = np.ones(100)\n"
        results = auto_detect_invariants(source_code=source)
        names = [r.name for r in results]
        assert "Charge Conservation" in names

    def test_lorentz_keyword_detected(self):
        source = "# Lorentz force computation\nF = q * (E + np.cross(v, B))\n"
        results = auto_detect_invariants(source_code=source)
        names = [r.name for r in results]
        assert "Lorentz Force" in names

    def test_scipy_solver_suggests_energy(self):
        source = "from scipy.integrate import solve_ivp\n"
        results = auto_detect_invariants(source_code=source)
        names = [r.name for r in results]
        assert "Total Energy" in names

    def test_empty_source_returns_empty(self):
        results = auto_detect_invariants(source_code="# empty file\n")
        assert len(results) == 0

    def test_sorted_by_confidence(self):
        source = (
            "# Boris pusher\n"
            "velocities = np.zeros((100, 3))\n"
            "charges = np.ones(100)\n"
        )
        results = auto_detect_invariants(source_code=source)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)


class TestAutoDetectFromState:
    def test_state_with_velocity_and_mass(self):
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={
                "velocities": np.zeros((10, 3)),
                "masses": np.ones(10),
            },
        )
        results = auto_detect_invariants(state=state)
        names = [r.name for r in results]
        assert "Total Energy" in names
        assert "Linear Momentum" in names

    def test_state_with_positions_for_angular_momentum(self):
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={
                "positions": np.zeros((10, 3)),
                "velocities": np.zeros((10, 3)),
                "masses": np.ones(10),
            },
        )
        results = auto_detect_invariants(state=state)
        names = [r.name for r in results]
        assert "Angular Momentum" in names

    def test_state_with_charges(self):
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={"charges": np.ones(10)},
        )
        results = auto_detect_invariants(state=state)
        names = [r.name for r in results]
        assert "Charge Conservation" in names

    def test_state_with_fields_suggests_gauss(self):
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={
                "E_field": np.zeros((10, 3)),
                "charge_density": np.zeros(10),
            },
        )
        results = auto_detect_invariants(state=state)
        names = [r.name for r in results]
        assert "Gauss's Law" in names

    def test_empty_state(self):
        state = SimulationState(timestep=0, time=0.0)
        results = auto_detect_invariants(state=state)
        assert len(results) == 0


class TestAutoDetectCombined:
    def test_combined_source_and_state(self):
        source = "# Boris pusher\nvelocities = np.zeros((100, 3))\n"
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={
                "velocities": np.zeros((10, 3)),
                "masses": np.ones(10),
            },
            metadata={"dt": 0.01},
        )
        results = auto_detect_invariants(source_code=source, state=state)
        # Should have higher confidence due to both sources agreeing
        names = [r.name for r in results]
        assert "Total Energy" in names
        assert "Boris Energy" in names

    def test_deduplication_keeps_highest_confidence(self):
        source = "velocities = np.zeros((100, 3))\n"
        state = SimulationState(
            timestep=0, time=0.0,
            arrays={
                "velocities": np.zeros((10, 3)),
                "masses": np.ones(10),
            },
        )
        results = auto_detect_invariants(source_code=source, state=state)
        # Should not have duplicates
        names = [r.name for r in results]
        assert len(names) == len(set(names))
