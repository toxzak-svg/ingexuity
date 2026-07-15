"""Synthetic prediction-training data tools for IngExuity."""

from .schema import SCHEMA_VERSION, normalize_probabilities, validate_record
from .scenarios import generate_scenarios
from .render import TeacherRenderer, TemplateRenderer

__all__ = [
    "SCHEMA_VERSION",
    "TeacherRenderer",
    "TemplateRenderer",
    "generate_scenarios",
    "normalize_probabilities",
    "validate_record",
]
