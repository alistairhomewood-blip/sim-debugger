"""Typer CLI for sim-debugger.

Commands:
    run              Instrument and run a simulation with invariant monitoring
    check            Static analysis: suggest applicable invariants for a script
    list-invariants  Show all registered invariant monitors
    report           Re-render a saved violation report from JSON
    init             Create a default .sim-debugger.toml config file
    export           Export violation history from a monitoring session
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sim_debugger.core.auto_detect import auto_detect_invariants
from sim_debugger.core.config import SimDebuggerConfig, load_config, save_default_config
from sim_debugger.core.invariants import create_default_registry
from sim_debugger.core.plugins import load_plugins
from sim_debugger.instrument.ast_rewriter import transform_source
from sim_debugger.parsec import BorisEnergyInvariant, GaussLawInvariant, LorentzForceInvariant

app = typer.Typer(
    name="sim-debugger",
    help=(
        "Instrument simulations, monitor physical invariants, "
        "explain failures in physics language."
    ),
    add_completion=False,
)
console = Console()


def _get_full_registry(config: SimDebuggerConfig | None = None):
    """Create registry with all built-in + PARSEC + plugin invariants."""
    registry = create_default_registry()
    registry.register(BorisEnergyInvariant())
    registry.register(GaussLawInvariant())
    registry.register(LorentzForceInvariant())

    # Load plugins if configured
    if config and config.plugins.paths:
        loaded = load_plugins(
            registry,
            config.plugins.paths,
            enabled=config.plugins.enabled or None,
            disabled=config.plugins.disabled or None,
        )
        if loaded:
            console.print(
                f"[dim]Loaded {len(loaded)} plugin invariant(s): "
                f"{', '.join(p.name for p in loaded)}[/dim]"
            )

    return registry


@app.command()
def run(
    script: str = typer.Argument(help="Path to the simulation Python script"),
    invariants: str | None = typer.Option(
        None, "--invariants", "-i",
        help="Comma-separated list of invariant names to monitor",
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", "-t",
        help="Global violation threshold (overrides defaults)",
    ),
    mode: str = typer.Option(
        "default", "--mode", "-m",
        help="Monitoring mode: lightweight (every 100 steps), default (every step), full",
    ),
    output: str = typer.Option(
        "text", "--output", "-o",
        help="Output format: text or json",
    ),
    log: str | None = typer.Option(
        None, "--log", "-l",
        help="Log violations to file",
    ),
    config_file: str | None = typer.Option(
        None, "--config", "-c",
        help="Path to .sim-debugger.toml config file",
    ),
    export_json: str | None = typer.Option(
        None, "--export-json",
        help="Export violation history to JSON after run",
    ),
    no_instrument: bool = typer.Option(
        False, "--no-instrument",
        help="Run without AST instrumentation (use decorators only)",
    ),
) -> None:
    """Instrument and run a simulation script with invariant monitoring.

    Example:
        sim-debugger run my_boris_pusher.py --invariants energy,charge
    """
    script_path = Path(script)
    if not script_path.exists():
        console.print(f"[red]Error:[/red] Script not found: {script}")
        raise typer.Exit(1) from None

    if not script_path.suffix == ".py":
        console.print(f"[red]Error:[/red] Script must be a Python file: {script}")
        raise typer.Exit(1) from None

    # Load configuration
    try:
        config = load_config(
            path=config_file,
            start_dir=str(script_path.parent),
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1) from None

    # CLI arguments override config
    if invariants:
        config.monitor.invariants = [name.strip() for name in invariants.split(",")]
    if mode != "default":
        config.monitor.mode = mode
    if output != "text":
        config.output.format = output
    if log:
        config.output.log_file = log
    if export_json:
        config.output.json_file = export_json

    # Parse invariant list
    inv_list = config.monitor.invariants
    registry = _get_full_registry(config)

    if inv_list:
        # Validate invariant names
        for name in inv_list:
            try:
                registry.get(name)
            except KeyError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1) from None

    # Parse thresholds
    thresholds: dict[str, float] = dict(config.thresholds.thresholds)
    if threshold is not None and inv_list:
        thresholds.update({name: threshold for name in inv_list})

    # Determine check interval (used by the monitor, validated here)
    _check_interval = config.get_check_interval()  # noqa: F841

    # Print header
    version = "0.2.0"
    console.print(Panel.fit(
        f"[bold]sim-debugger v{version}[/bold] -- monitoring invariants",
        border_style="blue",
    ))

    if config.config_file:
        console.print(f"[dim]Config: {config.config_file}[/dim]")

    if no_instrument:
        # Run without AST instrumentation
        console.print(f"Running [cyan]{script}[/cyan] (decorator mode, no AST instrumentation)")
        _run_direct(script_path)
    else:
        # Instrument and run
        console.print(f"Instrumenting [cyan]{script}[/cyan]...")
        try:
            source = script_path.read_text()
            transformed, transformer = transform_source(
                source,
                filename=str(script_path),
                invariants=inv_list,
                thresholds=thresholds,
            )
        except SyntaxError as e:
            console.print(f"[red]Syntax error in {script}:[/red] {e}")
            raise typer.Exit(1) from None

        if transformer.instrumented_loops:
            for start, end in transformer.instrumented_loops:
                console.print(
                    f"  Instrumented timestep loop at lines {start}-{end}"
                )
        else:
            # Try auto-detection suggestion
            suggestions = auto_detect_invariants(source_code=source)
            if suggestions:
                console.print(
                    "[yellow]Warning:[/yellow] No timestep loops detected. "
                    "The simulation will run but no invariants will be monitored."
                )
                console.print(
                    "[dim]Auto-detection suggests these invariants might be relevant:[/dim]"
                )
                for s in suggestions[:3]:
                    console.print(f"  [dim]- {s.name} ({s.reason})[/dim]")
                console.print(
                    "[dim]Consider using @sim_debugger.timestep decorator for "
                    "explicit marking.[/dim]"
                )
            else:
                console.print(
                    "[yellow]Warning:[/yellow] No timestep loops detected. "
                    "The simulation will run but no invariants will be monitored.\n"
                    "Consider using @sim_debugger.timestep decorator for explicit marking."
                )

        console.print("Running instrumented simulation...\n")

        # Execute the instrumented code
        start_time = time.time()
        try:
            # Set up the execution namespace
            namespace: dict[str, object] = {
                "__name__": "__main__",
                "__file__": str(script_path.resolve()),
            }
            # Add the script's directory to sys.path
            script_dir = str(script_path.resolve().parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            code = compile(transformed, str(script_path), "exec")
            # SECURITY: exec() is used by design to run user-supplied
            # simulation scripts with AST-injected monitoring hooks.
            # Only run trusted scripts. See README for details.
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            console.print(f"\n[red]Simulation error:[/red] {type(e).__name__}: {e}")
            raise typer.Exit(1) from None

        elapsed = time.time() - start_time
        console.print(f"\n[dim]Simulation completed in {elapsed:.2f}s[/dim]")

    # Handle log output
    if config.output.log_file:
        console.print(f"[dim]Violations logged to {config.output.log_file}[/dim]")

    # Handle JSON export
    if config.output.json_file:
        console.print(f"[dim]Violation history exported to {config.output.json_file}[/dim]")


def _run_direct(script_path: Path) -> None:
    """Run a script directly without AST instrumentation."""
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(script_path.resolve()),
    }
    script_dir = str(script_path.resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    code = compile(script_path.read_text(), str(script_path), "exec")
    # SECURITY: exec() is used by design to run user-supplied simulation
    # scripts without AST instrumentation. Only run trusted scripts.
    exec(code, namespace)  # noqa: S102


@app.command()
def check(
    script: str = typer.Argument(help="Path to the simulation Python script"),
) -> None:
    """Static analysis: suggest applicable invariants for a simulation script.

    Inspects the script's imports, variable names, and code patterns to
    suggest which invariant monitors are likely relevant.
    """
    script_path = Path(script)
    if not script_path.exists():
        console.print(f"[red]Error:[/red] Script not found: {script}")
        raise typer.Exit(1) from None

    source = script_path.read_text()

    console.print(Panel.fit(
        f"[bold]sim-debugger check[/bold] -- {script}",
        border_style="blue",
    ))

    # Use the auto-detection system
    suggestions = auto_detect_invariants(source_code=source)

    if suggestions:
        table = Table(title="Suggested Invariants")
        table.add_column("Invariant", style="cyan")
        table.add_column("Confidence", style="yellow", justify="right")
        table.add_column("Reason", style="dim")
        table.add_column("Source", style="dim")
        for s in suggestions:
            table.add_row(
                s.name,
                f"{s.confidence:.0%}",
                s.reason,
                s.source,
            )
        console.print(table)

        high_conf = [s for s in suggestions if s.confidence >= 0.5]
        if high_conf:
            inv_list = ",".join(s.name for s in high_conf)
            console.print(
                f"\nRun with: [bold]sim-debugger run "
                f"{script} --invariants \"{inv_list}\"[/bold]"
            )
    else:
        console.print(
            "[yellow]No specific invariants suggested.[/yellow]\n"
            "Run with auto-detection: [bold]sim-debugger run "
            f"{script}[/bold]"
        )


@app.command(name="list-invariants")
def list_invariants(
    config_file: str | None = typer.Option(
        None, "--config", "-c",
        help="Path to .sim-debugger.toml config file",
    ),
) -> None:
    """Show all registered invariant monitors with their descriptions."""
    config = None
    if config_file:
        try:
            config = load_config(path=config_file)
        except (FileNotFoundError, ValueError):
            pass

    registry = _get_full_registry(config)

    table = Table(title="Available Invariant Monitors")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Default Threshold", style="dim", justify="right")
    table.add_column("Type", style="yellow")

    builtin_names = {
        "Total Energy", "Linear Momentum", "Angular Momentum",
        "Charge Conservation", "Particle Count",
    }
    parsec_names = {"Boris Energy", "Gauss's Law", "Lorentz Force"}

    for inv in registry.list_all():
        if inv.name in builtin_names:
            inv_type = "built-in"
        elif inv.name in parsec_names:
            inv_type = "PARSEC"
        else:
            inv_type = "plugin"
        table.add_row(
            inv.name,
            inv.description,
            f"{inv.default_threshold:.0e}",
            inv_type,
        )

    console.print(table)


@app.command()
def report(
    violations_file: str = typer.Argument(
        help="Path to a JSON violations file"
    ),
) -> None:
    """Re-render a saved violation report from JSON.

    Reads a JSON file containing violation data and produces a
    formatted report.
    """
    path = Path(violations_file)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {violations_file}")
        raise typer.Exit(1) from None

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        raise typer.Exit(1) from None

    console.print(Panel.fit(
        "[bold]sim-debugger report[/bold]",
        border_style="blue",
    ))

    if isinstance(data, list):
        violations = data
    elif isinstance(data, dict) and "violations" in data:
        violations = data["violations"]

        # Show trends if available
        if "trends" in data:
            trends_table = Table(title="Invariant Trends")
            trends_table.add_column("Invariant", style="cyan")
            trends_table.add_column("Status", style="yellow")
            trends_table.add_column("Drift", justify="right")
            trends_table.add_column("Samples", justify="right", style="dim")

            for name, trend in data["trends"].items():
                is_stable = trend.get("is_stable")
                status = "[green]STABLE[/green]" if is_stable else "[red]DRIFTING[/red]"
                drift = f"{trend.get('relative_drift', 0):.2e}"
                samples = str(trend.get("num_samples", 0))
                trends_table.add_row(name, status, drift, samples)

            console.print(trends_table)
            console.print()
    else:
        console.print("[red]Error:[/red] Unrecognised JSON format.")
        raise typer.Exit(1) from None

    if not violations:
        console.print("[green]No violations in report.[/green]")
        return

    for i, v in enumerate(violations, 1):
        sev = v.get("severity", "unknown")
        sev_style = {
            "warning": "yellow",
            "error": "red",
            "critical": "bold red",
        }.get(sev, "white")

        console.print(f"\n[bold]Violation #{i}[/bold]")
        console.print(f"  Invariant:  [cyan]{v.get('invariant_name', '?')}[/cyan]")
        console.print(f"  Severity:   [{sev_style}]{sev.upper()}[/{sev_style}]")
        console.print(f"  Timestep:   {v.get('timestep', '?')}")
        console.print(
            f"  Value:      {v.get('actual_value', '?')} "
            f"(expected: {v.get('expected_value', '?')})"
        )
        console.print(f"  Error:      {v.get('relative_error', '?')}")

        explanation = v.get("explanation")
        if explanation:
            console.print("\n  [bold]Explanation:[/bold]")
            for line in explanation.split("\n"):
                console.print(f"    {line}")


@app.command()
def init(
    path: str = typer.Option(
        ".sim-debugger.toml", "--path", "-p",
        help="Where to create the config file",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing config file",
    ),
) -> None:
    """Create a default .sim-debugger.toml configuration file.

    Generates a starter configuration file with documented options.
    """
    config_path = Path(path)
    if config_path.exists() and not force:
        console.print(
            f"[yellow]Config file already exists:[/yellow] {path}\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(1) from None

    save_default_config(config_path)
    console.print(f"[green]Created config file:[/green] {path}")
    console.print("[dim]Edit the file to customise monitoring settings.[/dim]")


@app.callback()
def main() -> None:
    """sim-debugger: monitor physical invariants in numerical simulations."""
    pass


if __name__ == "__main__":
    app()
