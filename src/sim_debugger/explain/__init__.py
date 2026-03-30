"""Explanation module: translate violations into physics-language descriptions."""

from sim_debugger.explain.generator import generate_explanation
from sim_debugger.explain.templates import ExplanationTemplate, get_template

__all__ = [
    "generate_explanation",
    "ExplanationTemplate",
    "get_template",
]
