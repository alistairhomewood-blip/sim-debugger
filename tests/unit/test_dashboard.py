"""Tests for the Textual TUI dashboard.

Tests the dashboard components in isolation (no full Textual app mount).
Focuses on data structures, state management, helper functions, and
widget logic that can be tested without a terminal.
"""


from sim_debugger.core.monitor import Monitor
from sim_debugger.core.violations import Violation, ViolationSeverity
from sim_debugger.dashboard.app import (
    DashboardState,
    SimDebuggerDashboard,
    _render_sparkline,
    _severity_rank,
    create_dashboard,
)

# ===========================================================================
# DashboardState data class
# ===========================================================================

class TestDashboardState:
    def test_default_construction(self):
        """DashboardState has sensible defaults."""
        ds = DashboardState()
        assert ds.timestep == 0
        assert ds.total_timesteps is None
        assert ds.sim_time == 0.0
        assert ds.invariant_values == {}
        assert ds.invariant_statuses == {}
        assert ds.violations == []
        assert ds.wall_clock_elapsed == 0.0

    def test_with_values(self):
        """DashboardState can be constructed with invariant data."""
        ds = DashboardState(
            timestep=100,
            total_timesteps=1000,
            sim_time=1.0,
            invariant_values={"Total Energy": 25.0, "Linear Momentum": 3.0},
            invariant_statuses={"Total Energy": "ok", "Linear Momentum": "warning"},
            wall_clock_elapsed=5.0,
        )
        assert ds.timestep == 100
        assert ds.total_timesteps == 1000
        assert ds.invariant_values["Total Energy"] == 25.0
        assert ds.invariant_statuses["Linear Momentum"] == "warning"

    def test_with_violations(self):
        """DashboardState can hold violation objects."""
        v = Violation(
            invariant_name="Total Energy",
            timestep=50,
            time=0.5,
            expected_value=1.0,
            actual_value=1.5,
            relative_error=0.5,
            absolute_error=0.5,
            severity=ViolationSeverity.ERROR,
        )
        ds = DashboardState(violations=[v])
        assert len(ds.violations) == 1
        assert ds.violations[0].invariant_name == "Total Energy"


# ===========================================================================
# Sparkline rendering
# ===========================================================================

class TestSparklineRendering:
    def test_empty_values(self):
        """Empty input returns empty string."""
        assert _render_sparkline([]) == ""

    def test_single_value(self):
        """Single value returns a mid-level block."""
        result = _render_sparkline([1.0])
        assert len(result) == 1

    def test_constant_values(self):
        """Constant values produce a flat line."""
        result = _render_sparkline([5.0] * 10)
        # All characters should be the same (flat line)
        assert len(set(result)) == 1
        assert len(result) == 10

    def test_increasing_values(self):
        """Increasing values produce ascending blocks."""
        result = _render_sparkline([float(i) for i in range(8)])
        assert len(result) == 8
        # First character should be lowest block, last should be highest
        assert result[0] < result[-1]  # Character comparison works for these

    def test_width_limit(self):
        """Sparkline respects width limit."""
        result = _render_sparkline(list(range(100)), width=20)
        assert len(result) <= 20

    def test_two_values(self):
        """Two different values produce two characters."""
        result = _render_sparkline([0.0, 1.0])
        assert len(result) == 2


# ===========================================================================
# Severity ranking
# ===========================================================================

class TestSeverityRank:
    def test_warning_rank(self):
        assert _severity_rank(ViolationSeverity.WARNING) == 1

    def test_error_rank(self):
        assert _severity_rank(ViolationSeverity.ERROR) == 2

    def test_critical_rank(self):
        assert _severity_rank(ViolationSeverity.CRITICAL) == 3

    def test_ordering(self):
        """Critical > Error > Warning."""
        assert _severity_rank(ViolationSeverity.CRITICAL) > _severity_rank(ViolationSeverity.ERROR)
        assert _severity_rank(ViolationSeverity.ERROR) > _severity_rank(ViolationSeverity.WARNING)


# ===========================================================================
# Dashboard factory
# ===========================================================================

class TestCreateDashboard:
    def test_create_without_monitor(self):
        """Dashboard can be created without a monitor."""
        dashboard = create_dashboard()
        assert isinstance(dashboard, SimDebuggerDashboard)

    def test_create_with_monitor(self):
        """Dashboard can be created with a monitor."""
        monitor = Monitor(invariants=["Total Energy"])
        dashboard = create_dashboard(monitor=monitor)
        assert isinstance(dashboard, SimDebuggerDashboard)
        assert dashboard._monitor is monitor

    def test_initial_state(self):
        """Dashboard starts in the correct initial state."""
        dashboard = create_dashboard()
        assert dashboard.paused is False
        assert dashboard.verbose is False
        assert dashboard._all_violations == []
