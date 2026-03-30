"""Violation history and trend analysis.

Tracks invariant values and violations across timesteps, enabling:
- Historical violation query (by invariant, severity, timestep range)
- Trend detection (drift rate, oscillation frequency, growth rate)
- Statistical summary of invariant stability
- Export to JSON and HDF5 for post-analysis

Usage::

    from sim_debugger.core.history import ViolationHistory

    history = ViolationHistory()
    history.record_values(timestep=100, values={"Total Energy": 1.0})
    history.record_violation(violation)

    # Query
    recent = history.get_violations(last_n=10)
    trend = history.compute_trend("Total Energy")

    # Export
    history.export_json("violations.json")
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim_debugger.core.violations import Violation, ViolationSeverity

# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

@dataclass
class InvariantTrend:
    """Trend analysis for a single invariant over its recorded history.

    Attributes:
        invariant_name: Name of the invariant.
        num_samples: Number of recorded values.
        mean: Mean value over the history.
        std: Standard deviation.
        min_value: Minimum recorded value.
        max_value: Maximum recorded value.
        drift_rate: Linear drift rate (slope of best-fit line) per timestep.
        drift_total: Total drift from first to last value.
        relative_drift: drift_total / |initial_value|.
        is_stable: True if relative drift is within expected bounds.
        growth_rate: Exponential growth rate (if divergent).
        oscillation_amplitude: Peak-to-peak amplitude of oscillation.
    """

    invariant_name: str
    num_samples: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    drift_rate: float = 0.0
    drift_total: float = 0.0
    relative_drift: float = 0.0
    is_stable: bool = True
    growth_rate: float = 0.0
    oscillation_amplitude: float = 0.0


# ---------------------------------------------------------------------------
# Violation History
# ---------------------------------------------------------------------------

class ViolationHistory:
    """Tracks invariant values and violations across the full simulation.

    Unlike StateHistory (which is a fixed-size ring buffer for temporal
    localisation), ViolationHistory retains summary data for the entire
    simulation run, enabling trend analysis and export.

    Memory usage scales with the number of recorded timesteps (values only,
    not full state arrays) and the number of violations.
    """

    def __init__(self, downsample_interval: int = 1) -> None:
        """
        Args:
            downsample_interval: Record values every N timesteps.
                Set > 1 for long simulations to limit memory usage.
        """
        self._downsample_interval = downsample_interval

        # Invariant value time series: name -> list of (timestep, value)
        self._value_series: dict[str, list[tuple[int, float]]] = defaultdict(list)

        # All violations in chronological order
        self._violations: list[Violation] = []

        # Per-invariant violation counts by severity
        self._violation_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self._record_count: int = 0

    def record_values(
        self,
        timestep: int,
        values: dict[str, float],
    ) -> None:
        """Record invariant values at a timestep.

        Args:
            timestep: The current timestep.
            values: Dict of invariant_name -> computed value.
        """
        self._record_count += 1
        if self._record_count % self._downsample_interval != 0:
            return

        for name, value in values.items():
            self._value_series[name].append((timestep, value))

    def record_violation(self, violation: Violation) -> None:
        """Record a detected violation.

        Args:
            violation: The violation to record.
        """
        self._violations.append(violation)
        self._violation_counts[violation.invariant_name][
            violation.severity.value
        ] += 1

    def record_violations(self, violations: list[Violation]) -> None:
        """Record multiple violations at once."""
        for v in violations:
            self.record_violation(v)

    # ----- Query methods -----

    def get_violations(
        self,
        invariant_name: str | None = None,
        severity: ViolationSeverity | None = None,
        timestep_min: int | None = None,
        timestep_max: int | None = None,
        last_n: int | None = None,
    ) -> list[Violation]:
        """Query violations with optional filters.

        Args:
            invariant_name: Filter by invariant name (None = all).
            severity: Filter by minimum severity (None = all).
            timestep_min: Include violations at or after this timestep.
            timestep_max: Include violations at or before this timestep.
            last_n: Return only the last N matching violations.

        Returns:
            List of matching violations in chronological order.
        """
        results = self._violations

        if invariant_name is not None:
            results = [v for v in results if v.invariant_name == invariant_name]

        if severity is not None:
            severity_rank = _severity_to_rank(severity)
            results = [
                v for v in results
                if _severity_to_rank(v.severity) >= severity_rank
            ]

        if timestep_min is not None:
            results = [v for v in results if v.timestep >= timestep_min]

        if timestep_max is not None:
            results = [v for v in results if v.timestep <= timestep_max]

        if last_n is not None:
            results = results[-last_n:]

        return results

    def get_value_series(
        self,
        invariant_name: str,
    ) -> list[tuple[int, float]]:
        """Get the full value time series for an invariant.

        Returns:
            List of (timestep, value) pairs.
        """
        return list(self._value_series.get(invariant_name, []))

    @property
    def total_violations(self) -> int:
        """Total number of violations recorded."""
        return len(self._violations)

    @property
    def invariant_names(self) -> list[str]:
        """Names of all invariants with recorded values."""
        return sorted(self._value_series.keys())

    # ----- Trend analysis -----

    def compute_trend(
        self,
        invariant_name: str,
        stability_threshold: float = 1e-6,
    ) -> InvariantTrend:
        """Compute trend analysis for a single invariant.

        Args:
            invariant_name: The invariant to analyse.
            stability_threshold: Relative drift below this is considered stable.

        Returns:
            InvariantTrend with statistical and trend information.
        """
        series = self._value_series.get(invariant_name, [])
        if not series:
            return InvariantTrend(invariant_name=invariant_name)

        timesteps = np.array([s[0] for s in series], dtype=float)
        values = np.array([s[1] for s in series], dtype=float)

        n = len(values)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values)) if n > 1 else 0.0
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        # Linear drift rate via least-squares fit
        drift_rate = 0.0
        if n >= 2:
            t_centered = timesteps - np.mean(timesteps)
            denom = np.sum(t_centered ** 2)
            if denom > 0:
                drift_rate = float(np.sum(t_centered * (values - mean_val)) / denom)

        drift_total = values[-1] - values[0]
        if abs(values[0]) > 1e-300:
            relative_drift = abs(drift_total) / abs(values[0])
        else:
            relative_drift = abs(drift_total)

        is_stable = relative_drift < stability_threshold

        # Exponential growth rate
        growth_rate = 0.0
        if n >= 4 and np.all(values > 0):
            log_values = np.log(values)
            t_centered = timesteps - np.mean(timesteps)
            denom = np.sum(t_centered ** 2)
            if denom > 0:
                growth_rate = float(
                    np.sum(t_centered * (log_values - np.mean(log_values))) / denom
                )

        # Oscillation amplitude: peak-to-peak of detrended signal
        oscillation_amplitude = 0.0
        if n >= 3:
            detrended = values - (mean_val + drift_rate * (timesteps - np.mean(timesteps)))
            oscillation_amplitude = float(np.max(detrended) - np.min(detrended))

        return InvariantTrend(
            invariant_name=invariant_name,
            num_samples=n,
            mean=mean_val,
            std=std_val,
            min_value=min_val,
            max_value=max_val,
            drift_rate=drift_rate,
            drift_total=drift_total,
            relative_drift=relative_drift,
            is_stable=is_stable,
            growth_rate=growth_rate,
            oscillation_amplitude=oscillation_amplitude,
        )

    def compute_all_trends(
        self,
        stability_threshold: float = 1e-6,
    ) -> dict[str, InvariantTrend]:
        """Compute trends for all recorded invariants.

        Returns:
            Dict of invariant_name -> InvariantTrend.
        """
        return {
            name: self.compute_trend(name, stability_threshold)
            for name in self._value_series
        }

    # ----- Export -----

    def export_json(self, path: str | Path) -> None:
        """Export violation history and trends to a JSON file.

        The JSON contains:
        - violations: list of violation dicts
        - trends: per-invariant trend analysis
        - value_series: downsampled invariant values

        Args:
            path: Output file path.
        """
        trends = self.compute_all_trends()

        data = {
            "version": "1.0",
            "total_violations": self.total_violations,
            "violations": [_violation_to_dict(v) for v in self._violations],
            "trends": {
                name: _trend_to_dict(trend)
                for name, trend in trends.items()
            },
            "value_series": {
                name: [{"timestep": ts, "value": val} for ts, val in series]
                for name, series in self._value_series.items()
            },
            "violation_counts": dict(self._violation_counts),
        }

        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def export_hdf5(self, path: str | Path) -> None:
        """Export violation history to an HDF5 file.

        Requires h5py. Stores value series as datasets and violations
        as a structured table.

        Args:
            path: Output file path.

        Raises:
            ImportError: If h5py is not installed.
        """
        try:
            import h5py  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "HDF5 export requires h5py. Install with: pip install h5py"
            ) from exc

        with h5py.File(str(path), "w") as f:
            # Store value series
            series_grp = f.create_group("value_series")
            for name, series in self._value_series.items():
                if not series:
                    continue
                timesteps = np.array([s[0] for s in series])
                values = np.array([s[1] for s in series])
                inv_grp = series_grp.create_group(name)
                inv_grp.create_dataset("timesteps", data=timesteps)
                inv_grp.create_dataset("values", data=values)

            # Store violations as a table
            if self._violations:
                viol_grp = f.create_group("violations")
                viol_grp.create_dataset(
                    "invariant_names",
                    data=[v.invariant_name.encode() for v in self._violations],
                )
                viol_grp.create_dataset(
                    "timesteps",
                    data=[v.timestep for v in self._violations],
                )
                viol_grp.create_dataset(
                    "severities",
                    data=[v.severity.value.encode() for v in self._violations],
                )
                viol_grp.create_dataset(
                    "relative_errors",
                    data=[v.relative_error for v in self._violations],
                )
                viol_grp.create_dataset(
                    "expected_values",
                    data=[v.expected_value for v in self._violations],
                )
                viol_grp.create_dataset(
                    "actual_values",
                    data=[v.actual_value for v in self._violations],
                )

            # Store trends as attributes
            trends_grp = f.create_group("trends")
            for name, trend in self.compute_all_trends().items():
                t_grp = trends_grp.create_group(name)
                t_grp.attrs["num_samples"] = trend.num_samples
                t_grp.attrs["mean"] = trend.mean
                t_grp.attrs["std"] = trend.std
                t_grp.attrs["drift_rate"] = trend.drift_rate
                t_grp.attrs["drift_total"] = trend.drift_total
                t_grp.attrs["relative_drift"] = trend.relative_drift
                t_grp.attrs["is_stable"] = trend.is_stable
                t_grp.attrs["growth_rate"] = trend.growth_rate

    def summary(self) -> str:
        """Generate a human-readable summary of the violation history.

        Returns:
            Multi-line string with the summary.
        """
        lines = [
            "Violation History Summary",
            "=" * 50,
            f"Total violations: {self.total_violations}",
            "",
        ]

        if self._violation_counts:
            lines.append("Violations by invariant:")
            for inv_name, counts in sorted(self._violation_counts.items()):
                total = sum(counts.values())
                detail = ", ".join(
                    f"{sev}: {cnt}" for sev, cnt in sorted(counts.items())
                )
                lines.append(f"  {inv_name}: {total} ({detail})")
            lines.append("")

        trends = self.compute_all_trends()
        if trends:
            lines.append("Invariant trends:")
            for name, trend in sorted(trends.items()):
                status = "STABLE" if trend.is_stable else "DRIFTING"
                lines.append(
                    f"  {name}: {status} "
                    f"(drift={trend.relative_drift:.2e}, "
                    f"mean={trend.mean:.6e}, std={trend.std:.2e})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_to_rank(severity: ViolationSeverity) -> int:
    return {
        ViolationSeverity.WARNING: 1,
        ViolationSeverity.ERROR: 2,
        ViolationSeverity.CRITICAL: 3,
    }.get(severity, 0)


def _violation_to_dict(v: Violation) -> dict[str, Any]:
    """Convert a Violation to a JSON-serialisable dict."""
    d: dict[str, Any] = {
        "invariant_name": v.invariant_name,
        "timestep": v.timestep,
        "time": v.time,
        "expected_value": v.expected_value,
        "actual_value": v.actual_value,
        "relative_error": v.relative_error,
        "absolute_error": v.absolute_error,
        "severity": v.severity.value,
    }
    if v.explanation:
        d["explanation"] = v.explanation
    if v.localisation:
        loc: dict[str, Any] = {}
        if v.localisation.temporal:
            t = v.localisation.temporal
            loc["temporal"] = {
                "first_violation_timestep": t.first_violation_timestep,
                "pattern": t.pattern.value,
                "duration": t.duration,
            }
        if v.localisation.source:
            s = v.localisation.source
            loc["source"] = {
                "file": s.file,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "function_name": s.function_name,
                "sub_step": s.sub_step,
            }
        d["localisation"] = loc
    return d


def _trend_to_dict(trend: InvariantTrend) -> dict[str, Any]:
    """Convert an InvariantTrend to a JSON-serialisable dict."""
    return {
        "invariant_name": trend.invariant_name,
        "num_samples": trend.num_samples,
        "mean": trend.mean,
        "std": trend.std,
        "min_value": trend.min_value,
        "max_value": trend.max_value,
        "drift_rate": trend.drift_rate,
        "drift_total": trend.drift_total,
        "relative_drift": trend.relative_drift,
        "is_stable": trend.is_stable,
        "growth_rate": trend.growth_rate,
        "oscillation_amplitude": trend.oscillation_amplitude,
    }
