"""Synthetic prediction-training data tools for IngExuity."""

from .schema import SCHEMA_VERSION, normalize_probabilities, validate_record
from .scenarios import generate_scenarios
from .render import TeacherRenderer, TemplateRenderer
from .build import build_dataset

__all__ = [
    "SCHEMA_VERSION",
    "TeacherRenderer",
    "TemplateRenderer",
    "build_dataset",
    "generate_scenarios",
    "normalize_probabilities",
    "validate_record",
]
