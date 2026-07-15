"""Synthetic prediction-training data tools for IngExuity."""

from .schema import SCHEMA_VERSION, normalize_probabilities, validate_record
from .scenarios import generate_scenarios
from .render import TeacherRenderer, TemplateRenderer
from .build import build_dataset
from .curriculum import CATEGORY_COUNTS, build_curriculum

__all__ = [
    "SCHEMA_VERSION",
    "TeacherRenderer",
    "TemplateRenderer",
    "build_dataset",
    "CATEGORY_COUNTS",
    "build_curriculum",
    "generate_scenarios",
    "normalize_probabilities",
    "validate_record",
]
