"""
Pandas implementation for moon phase feature groups.

This module implements the computation of moon phase features for data stored
in pandas DataFrames. It inherits from the base MoonPhaseFeatureGroup and
implements the necessary methods to check the existence of the time column,
perform the lunar phase calculation, and add the result back to the DataFrame.

The lunar phase is approximated using a simple astronomical formula: the phase
is determined by the number of days since a known new moon divided by the
length of the synodic month (approximately 29.53058867 days). The
implementation uses a fixed reference date (6 January 2000 at 18:14 UTC),
which corresponds to a known new moon. For each timestamp, the algorithm
computes the phase fraction, then converts it to degrees or a binary full
moon indicator based on the requested representation.
"""

from __future__ import annotations

import datetime
from typing import Any, Set, Type

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional in some environments
    pd = None  # type: ignore

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.moon_phase.base import MoonPhaseFeatureGroup


class PandasMoonPhaseFeatureGroup(MoonPhaseFeatureGroup):
    """Pandas implementation of the MoonPhaseFeatureGroup."""

    # Threshold for considering a phase as a full moon when using the "is_full"
    # representation. The phase fraction for a full moon is 0.5. Values within
    # this absolute difference are considered a full moon.
    FULL_MOON_THRESHOLD: float = 0.05
    # Length of a synodic month (new moon to new moon) in days
    SYNODIC_MONTH: float = 29.53058867
    # Reference date for a known new moon (UTC). This is used as the baseline
    # for calculating phase fractions. Timestamps are assumed to be either
    # timezone-naive or in UTC. Users should convert their timestamps
    # accordingly before using this feature group.
    REFERENCE_DATE: datetime.datetime = datetime.datetime(2000, 1, 6, 18, 14)

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFrameWork]]:
        """
        Specify that this feature group is compatible with pandas DataFrames.

        Returns
        -------
        Set[type[ComputeFrameWork]]
            A set containing PandasDataframe, indicating support for pandas.
        """
        return {PandasDataframe}

    @classmethod
    def _check_time_feature_exists(cls, data: pd.DataFrame, time_feature: str) -> None:
        """Verify that the specified time feature exists in the DataFrame."""
        if time_feature not in data.columns:
            raise ValueError(f"Time feature '{time_feature}' not found in the data")

    @classmethod
    def _add_result_to_data(
        cls, data: pd.DataFrame, feature_name: str, result: pd.Series
    ) -> pd.DataFrame:
        """Add the computed result to the DataFrame under the given feature name."""
        data[feature_name] = result
        return data

    @classmethod
    def _calculate_moon_phase(
        cls, data: pd.DataFrame, representation: str, time_feature: str
    ) -> pd.Series:
        """
        Compute the moon phase for each row in the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the timestamp column.
        representation : str
            The desired representation ("fraction", "degrees" or "is_full").
        time_feature : str
            The name of the timestamp column on which to base the calculation.

        Returns
        -------
        pd.Series
            A Series containing the computed moon phase values.

        Raises
        ------
        ValueError
            If an unsupported representation is requested.
        """
        # Ensure we have pandas available
        if pd is None:
            raise ImportError(
                "Pandas is required for PandasMoonPhaseFeatureGroup but is not installed"
            )

        # Retrieve the time series
        time_series = data[time_feature]
        # Convert to datetime if necessary
        if not np.issubdtype(time_series.dtype, np.datetime64):
            time_series = pd.to_datetime(time_series, errors="coerce")
        # Compute the difference in days from the reference date
        # `.dt` accessor is used for datetime64[ns] dtype; ensures timezone naive
        diff_days = (time_series - cls.REFERENCE_DATE).dt.total_seconds() / 86400.0
        # Compute the phase fraction in [0, 1)
        phase_fraction = (diff_days % cls.SYNODIC_MONTH) / cls.SYNODIC_MONTH

        if representation == "fraction":
            return phase_fraction.astype(float)
        elif representation == "degrees":
            # Convert fraction to degrees
            return (phase_fraction * 360.0).astype(float)
        elif representation == "is_full":
            # Identify full moon: phase fraction near 0.5 (full moon)
            return (abs(phase_fraction - 0.5) < cls.FULL_MOON_THRESHOLD).astype(int)
        else:
            raise ValueError(f"Unsupported phase representation: {representation}")
