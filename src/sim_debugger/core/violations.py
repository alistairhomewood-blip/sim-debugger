"""Violation data models for sim-debugger.

Defines the data structures for representing conservation law violations,
their severity classification, and localisation results.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np


class ViolationSeverity(enum.Enum):
    """Severity classification for invariant violations.

    Severity is determined by the ratio of the relative error to the threshold:
    - WARNING:  error in [threshold, 10 * threshold)
    - ERROR:    error in [10 * threshold, 100 * threshold)
    - CRITICAL: error >= 100 * threshold, or value is NaN/Inf
    """

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ViolationPattern(enum.Enum):
    """Classification of how a violation manifests over time."""

    SUDDEN = "sudden"         # Single-timestep jump
    GRADUAL = "gradual"       # Slow accumulation over many timesteps
    OSCILLATORY = "oscillatory"  # Growing oscillation
    DIVERGENT = "divergent"   # Exponential growth


@dataclass(frozen=True)
class TemporalLocalisation:
    """Result of temporal localisation: when the violation first appeared."""

    first_violation_timestep: int
    pattern: ViolationPattern
    violation_trajectory: list[tuple[int, float]] = field(default_factory=list)

    @property
    def duration(self) -> int:
        """Number of timesteps from first deviation to detection."""
        if not self.violation_trajectory:
            return 0
        return self.violation_trajectory[-1][0] - self.first_violation_timestep


@dataclass(frozen=True)
class SpatialLocalisation:
    """Result of spatial localisation: where the violation is concentrated.

    Phase 2 feature -- included here for data model completeness.
    """

    region_type: str  # "cells" | "particles" | "boundary"
    indices: np.ndarray | None = None
    bounding_box: tuple[float, ...] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpatialLocalisation):
            return NotImplemented
        if self.region_type != other.region_type:
            return False
        if self.bounding_box != other.bounding_box:
            return False
        if self.indices is None and other.indices is None:
            return True
        if self.indices is None or other.indices is None:
            return False
        return np.array_equal(self.indices, other.indices)

    def __hash__(self) -> int:
        return hash((self.region_type, self.bounding_box))


@dataclass(frozen=True)
class SourceLocalisation:
    """Result of source-code localisation: which code caused the violation.

    Phase 3 feature -- included here for data model completeness.
    """

    file: str
    line_start: int
    line_end: int
    function_name: str
    sub_step: str | None = None


@dataclass(frozen=True)
class LocalisationResult:
    """Combined localisation result across all dimensions."""

    temporal: TemporalLocalisation | None = None
    spatial: SpatialLocalisation | None = None
    source: SourceLocalisation | None = None


@dataclass(frozen=True)
class Violation:
    """A detected conservation law violation.

    Attributes:
        invariant_name: Name of the invariant that was violated.
        timestep: Timestep at which the violation was detected.
        time: Simulation time at which the violation was detected.
        expected_value: The invariant value at the previous timestep (or initial value).
        actual_value: The invariant value at the current timestep.
        relative_error: |(actual - expected) / expected| (or absolute if expected is near zero).
        absolute_error: |actual - expected|.
        severity: Classification of how severe the violation is.
        localisation: Where/when the violation originated.
        explanation: Physics-language explanation (filled in by the explanation generator).
    """

    invariant_name: str
    timestep: int
    time: float
    expected_value: float
    actual_value: float
    relative_error: float
    absolute_error: float
    severity: ViolationSeverity
    localisation: LocalisationResult | None = None
    explanation: str | None = None

    @property
    def signed_relative_error(self) -> float:
        """Relative error with sign (positive = increase, negative = decrease)."""
        if abs(self.expected_value) < 1e-300:
            return self.actual_value - self.expected_value
        return (self.actual_value - self.expected_value) / abs(self.expected_value)


def classify_severity(
    relative_error: float,
    threshold: float,
    value: float = 0.0,
) -> ViolationSeverity:
    """Classify violation severity based on relative error and threshold.

    Args:
        relative_error: The absolute relative error |delta / prev|.
        threshold: The invariant's detection threshold.
        value: The current invariant value (used to detect NaN/Inf).

    Returns:
        The appropriate ViolationSeverity level.
    """
    if not math.isfinite(value) or not math.isfinite(relative_error):
        return ViolationSeverity.CRITICAL
    if relative_error >= 100 * threshold:
        return ViolationSeverity.CRITICAL
    if relative_error >= 10 * threshold:
        return ViolationSeverity.ERROR
    return ViolationSeverity.WARNING
