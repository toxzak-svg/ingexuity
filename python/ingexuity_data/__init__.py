"""Synthetic prediction-training data tools for IngExuity."""

from .schema import SCHEMA_VERSION, normalize_probabilities, validate_record
from .scenarios import generate_scenarios

__all__ = [
    "SCHEMA_VERSION",
    "generate_scenarios",
    "normalize_probabilities",
    "validate_record",
]
