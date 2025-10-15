"""
Moon phase feature group package.

This package contains the base and compute-framework specific implementations for
calculating moon phase features. The base implementation defines the common
interface and parsing logic, while the Pandas implementation provides the
concrete logic for working with pandas DataFrames.
"""

from .base import MoonPhaseFeatureGroup  # noqa: F401
from .pandas import PandasMoonPhaseFeatureGroup  # noqa: F401

__all__ = ["MoonPhaseFeatureGroup", "PandasMoonPhaseFeatureGroup"]
