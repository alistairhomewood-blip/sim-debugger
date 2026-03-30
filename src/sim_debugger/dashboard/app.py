"""Textual TUI dashboard for real-time invariant monitoring.

Provides a terminal-based dashboard with four panels:
1. Invariant Monitor -- live sparklines showing invariant values over time
2. Violation Log -- scrolling list of detected violations
3. Detail Panel -- full explanation of the selected violation
4. Simulation Status -- current timestep, wall-clock time, overhead

Key bindings:
    q -- quit the dashboard
    p -- pause monitoring
    r -- resume monitoring
    v -- toggle verbose output
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Header,
    Label,
)

from sim_debugger.core.monitor import Monitor
from sim_debugger.core.state import SimulationState
from sim_debugger.core.violations import Violation, ViolationSeverity

# ---------------------------------------------------------------------------
# Severity colour mapping
# ---------------------------------------------------------------------------

SEVERITY_COLOURS: dict[ViolationSeverity, str] = {
    ViolationSeverity.WARNING: "yellow",
    ViolationSeverity.ERROR: "red",
    ViolationSeverity.CRITICAL: "bold bright_red",
}

STATUS_COLOURS: dict[str, str] = {
    "ok": "green",
    "warning": "yellow",
    "error": "red",
    "critical": "bright_red",
}


# ---------------------------------------------------------------------------
# Data feed: decouples simulation from dashboard
# ---------------------------------------------------------------------------

@dataclass
class DashboardState:
    """Snapshot of data the dashboard needs each tick.

    This is the data transport between a running simulation and the TUI.
    The simulation pushes DashboardState objects; the dashboard renders them.
    """

    timestep: int = 0
    total_timesteps: int | None = None
    sim_time: float = 0.0
    invariant_values: dict[str, float] = field(default_factory=dict)
    invariant_statuses: dict[str, str] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    wall_clock_elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------

class InvariantPanel(Widget):
    """Live-updating panel displaying current invariant values with sparklines.

    Shows one row per invariant: name, current value, status indicator, and
    a sparkline of the value history.
    """

    DEFAULT_CSS = """
    InvariantPanel {
        height: auto;
        min-height: 5;
        border: solid $accent;
        padding: 0 1;
    }
    InvariantPanel .invariant-row {
        height: 3;
        layout: horizontal;
    }
    InvariantPanel .inv-name {
        width: 20;
        content-align: left middle;
    }
    InvariantPanel .inv-value {
        width: 18;
        content-align: right middle;
    }
    InvariantPanel .inv-status {
        width: 10;
        content-align: center middle;
    }
    InvariantPanel .inv-sparkline {
        width: 1fr;
        height: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._histories: dict[str, deque[float]] = {}
        self._max_history = 60

    def compose(self) -> ComposeResult:
        yield Label("[b]Invariant Monitor[/b]", id="inv-title")

    def update_invariants(
        self,
        values: dict[str, float],
        statuses: dict[str, str],
    ) -> None:
        """Update the displayed invariant values and statuses."""
        for name, value in values.items():
            if name not in self._histories:
                self._histories[name] = deque(maxlen=self._max_history)
            self._histories[name].append(value)

        # Rebuild display
        self._refresh_display(values, statuses)

    def _refresh_display(
        self,
        values: dict[str, float],
        statuses: dict[str, str],
    ) -> None:
        """Rebuild the invariant display with current data."""
        lines: list[str] = ["[b]Invariant Monitor[/b]", ""]
        for name, value in values.items():
            status = statuses.get(name, "ok")
            colour = STATUS_COLOURS.get(status, "white")

            # Format the sparkline as a simple text bar using Unicode blocks
            history = self._histories.get(name, deque())
            sparkline_str = _render_sparkline(list(history))

            status_label = f"[{colour}]{status.upper():^8s}[/{colour}]"
            lines.append(
                f"  {name:<20s}  {value:>14.6e}  {status_label}  {sparkline_str}"
            )

        title_widget = self.query_one("#inv-title", Label)
        title_widget.update("\n".join(lines))


class ViolationLog(Widget):
    """Scrolling log of all detected violations with severity colour coding."""

    DEFAULT_CSS = """
    ViolationLog {
        height: 1fr;
        min-height: 5;
        border: solid $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._violations: list[Violation] = []
        self._selected_index: int | None = None

    def compose(self) -> ComposeResult:
        yield Label("[b]Violation Log[/b]", id="vlog-content")

    def add_violations(self, violations: list[Violation]) -> None:
        """Add new violations to the log."""
        self._violations.extend(violations)
        self._refresh()

    def _refresh(self) -> None:
        lines = ["[b]Violation Log[/b]", ""]
        if not self._violations:
            lines.append("  [dim]No violations detected.[/dim]")
        else:
            for i, v in enumerate(self._violations[-50:]):  # Show last 50
                colour = SEVERITY_COLOURS.get(v.severity, "white")
                marker = ">" if i == self._selected_index else " "
                lines.append(
                    f" {marker}[{colour}]{v.severity.value.upper():>8s}[/{colour}]"
                    f"  t={v.timestep:<6d}"
                    f"  {v.invariant_name:<20s}"
                    f"  err={v.relative_error:.2e}"
                )
        content = self.query_one("#vlog-content", Label)
        content.update("\n".join(lines))

    @property
    def selected_violation(self) -> Violation | None:
        if self._selected_index is not None and self._selected_index < len(self._violations):
            return self._violations[self._selected_index]
        return None


class DetailPanel(Widget):
    """Full explanation of the currently selected violation."""

    DEFAULT_CSS = """
    DetailPanel {
        height: auto;
        min-height: 5;
        border: solid $accent;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("[b]Violation Detail[/b]\n\n  [dim]Select a violation to see details.[/dim]",
                     id="detail-content")

    def show_violation(self, violation: Violation | None) -> None:
        """Display the full explanation for a violation."""
        if violation is None:
            text = "[b]Violation Detail[/b]\n\n  [dim]No violation selected.[/dim]"
        else:
            colour = SEVERITY_COLOURS.get(violation.severity, "white")
            text_lines = [
                "[b]Violation Detail[/b]",
                "",
                f"  Invariant:  [cyan]{violation.invariant_name}[/cyan]",
                f"  Severity:   [{colour}]{violation.severity.value.upper()}[/{colour}]",
                f"  Timestep:   {violation.timestep}",
                f"  Time:       {violation.time:.6e}",
                f"  Expected:   {violation.expected_value:.6e}",
                f"  Actual:     {violation.actual_value:.6e}",
                f"  Rel Error:  {violation.relative_error:.2e}",
                f"  Abs Error:  {violation.absolute_error:.2e}",
            ]
            if violation.explanation:
                text_lines.append("")
                text_lines.append("  [b]Explanation:[/b]")
                for line in violation.explanation.split("\n"):
                    text_lines.append(f"    {line}")
            text = "\n".join(text_lines)

        content = self.query_one("#detail-content", Label)
        content.update(text)


class StatusBar(Widget):
    """Status bar showing simulation progress and overhead."""

    DEFAULT_CSS = """
    StatusBar {
        height: 3;
        border: solid $accent;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("[b]Simulation Status[/b]  Waiting...", id="status-content")

    def update_status(
        self,
        timestep: int,
        total_timesteps: int | None,
        sim_time: float,
        wall_clock: float,
        paused: bool,
        num_violations: int,
    ) -> None:
        """Update the status bar with current simulation progress."""
        if total_timesteps is not None and total_timesteps > 0:
            pct = timestep / total_timesteps * 100
            progress = f"Step {timestep}/{total_timesteps} ({pct:.1f}%)"
            bar_width = 30
            filled = int(bar_width * timestep / total_timesteps)
            bar = "[green]" + "#" * filled + "[/green]" + "-" * (bar_width - filled)
            progress_bar = f"[{bar}]"
        else:
            progress = f"Step {timestep}"
            progress_bar = ""

        pause_label = "  [yellow][PAUSED][/yellow]" if paused else ""
        viol_label = (
            f"  [red]{num_violations} violation(s)[/red]"
            if num_violations > 0
            else "  [green]No violations[/green]"
        )

        text = (
            f"[b]Simulation Status[/b]{pause_label}\n"
            f"  {progress}  {progress_bar}  "
            f"sim_t={sim_time:.4e}  wall={wall_clock:.1f}s{viol_label}"
        )
        content = self.query_one("#status-content", Label)
        content.update(text)


# ---------------------------------------------------------------------------
# Helper: render a sparkline as Unicode block characters
# ---------------------------------------------------------------------------

_SPARK_BLOCKS = "".join(chr(0x2581 + i) for i in range(8))  # ▁▂▃▄▅▆▇█


def _render_sparkline(values: list[float], width: int = 30) -> str:
    """Render a sparkline string using Unicode block characters.

    Args:
        values: The data series.
        width: Maximum number of characters in the sparkline.

    Returns:
        A string of Unicode block characters representing the data.
    """
    if not values:
        return ""

    # Use the last `width` values
    data = values[-width:]
    if len(data) < 2:
        return _SPARK_BLOCKS[3]  # Mid-level block

    lo = min(data)
    hi = max(data)
    span = hi - lo

    if span < 1e-300:
        # All values the same -- flat line
        return _SPARK_BLOCKS[3] * len(data)

    result = []
    for v in data:
        idx = int((v - lo) / span * 7)
        idx = max(0, min(7, idx))
        result.append(_SPARK_BLOCKS[idx])

    return "".join(result)


# ---------------------------------------------------------------------------
# Main Dashboard App
# ---------------------------------------------------------------------------

class SimDebuggerDashboard(App[None]):
    """Real-time Textual TUI dashboard for sim-debugger.

    Displays live invariant values, violation alerts with colour coding,
    temporal history sparklines, and a simulation progress status bar.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #top-section {
        height: auto;
    }
    #middle-section {
        height: 1fr;
        layout: horizontal;
    }
    #violation-log {
        width: 1fr;
    }
    #detail-panel {
        width: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "pause", "Pause"),
        Binding("r", "resume", "Resume"),
        Binding("v", "toggle_verbose", "Toggle Verbose"),
    ]

    paused: reactive[bool] = reactive(False)
    verbose: reactive[bool] = reactive(False)

    def __init__(
        self,
        monitor: Monitor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._monitor = monitor
        self._all_violations: list[Violation] = []
        self._start_time = time.monotonic()
        self._latest_state: DashboardState | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="top-section"):
            yield InvariantPanel(id="invariant-panel")
        with Horizontal(id="middle-section"):
            yield ViolationLog(id="violation-log")
            yield DetailPanel(id="detail-panel")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "sim-debugger dashboard"
        self.sub_title = "monitoring invariants"

    # ------------------------------------------------------------------
    # Public API: push state updates from simulation thread
    # ------------------------------------------------------------------

    def push_state(self, state: DashboardState) -> None:
        """Push a new dashboard state from the simulation.

        This is the main entry point for feeding data into the dashboard.
        It should be called from the simulation loop (possibly via
        call_from_thread if running in a separate thread).

        Args:
            state: The current dashboard state snapshot.
        """
        self._latest_state = state
        self._update_display(state)

    def push_monitor_state(
        self,
        sim_state: SimulationState,
        violations: list[Violation],
        monitor: Monitor,
        total_timesteps: int | None = None,
    ) -> None:
        """Convenience: build DashboardState from a Monitor and push it.

        Args:
            sim_state: Current simulation state.
            violations: Violations detected at this step.
            monitor: The Monitor instance.
            total_timesteps: Total expected timesteps (for progress bar).
        """
        if self.paused:
            return

        values = monitor.get_current_values()
        _initial = monitor.get_initial_values()  # noqa: F841 -- reserved for drift display

        # Determine per-invariant status
        statuses: dict[str, str] = {}
        for name in monitor.active_invariants:
            # Check if any violation at this timestep matches this invariant
            matching = [v for v in violations if v.invariant_name == name]
            if matching:
                worst = max(matching, key=lambda v: _severity_rank(v.severity))
                statuses[name] = worst.severity.value
            else:
                statuses[name] = "ok"

        ds = DashboardState(
            timestep=sim_state.timestep,
            total_timesteps=total_timesteps,
            sim_time=sim_state.time,
            invariant_values=values,
            invariant_statuses=statuses,
            violations=violations,
            wall_clock_elapsed=time.monotonic() - self._start_time,
        )
        self.push_state(ds)

    # ------------------------------------------------------------------
    # Internal display update
    # ------------------------------------------------------------------

    def _update_display(self, state: DashboardState) -> None:
        """Update all dashboard panels from a DashboardState."""
        # Update invariant panel
        inv_panel = self.query_one("#invariant-panel", InvariantPanel)
        inv_panel.update_invariants(state.invariant_values, state.invariant_statuses)

        # Update violation log with new violations
        if state.violations:
            self._all_violations.extend(state.violations)
            vlog = self.query_one("#violation-log", ViolationLog)
            vlog.add_violations(state.violations)

            # Auto-select latest violation for detail view
            detail = self.query_one("#detail-panel", DetailPanel)
            detail.show_violation(state.violations[-1])

        # Update status bar
        status = self.query_one("#status-bar", StatusBar)
        status.update_status(
            timestep=state.timestep,
            total_timesteps=state.total_timesteps,
            sim_time=state.sim_time,
            wall_clock=state.wall_clock_elapsed,
            paused=self.paused,
            num_violations=len(self._all_violations),
        )

    # ------------------------------------------------------------------
    # Key binding actions
    # ------------------------------------------------------------------

    def action_pause(self) -> None:
        """Pause monitoring updates."""
        self.paused = True
        self.sub_title = "PAUSED"
        self.notify("Monitoring paused. Press 'r' to resume.")

    def action_resume(self) -> None:
        """Resume monitoring updates."""
        self.paused = False
        self.sub_title = "monitoring invariants"
        self.notify("Monitoring resumed.")

    def action_toggle_verbose(self) -> None:
        """Toggle verbose output mode."""
        self.verbose = not self.verbose
        mode = "ON" if self.verbose else "OFF"
        self.notify(f"Verbose mode: {mode}")


def _severity_rank(severity: ViolationSeverity) -> int:
    """Return a numeric rank for severity comparison (higher = worse)."""
    return {
        ViolationSeverity.WARNING: 1,
        ViolationSeverity.ERROR: 2,
        ViolationSeverity.CRITICAL: 3,
    }.get(severity, 0)


# ---------------------------------------------------------------------------
# Factory function for launching the dashboard
# ---------------------------------------------------------------------------

def create_dashboard(
    monitor: Monitor | None = None,
) -> SimDebuggerDashboard:
    """Create a new dashboard instance.

    Args:
        monitor: Optional Monitor instance. If provided, the dashboard
                 can be fed data via push_monitor_state().

    Returns:
        A SimDebuggerDashboard app instance (call .run() to start).
    """
    return SimDebuggerDashboard(monitor=monitor)
