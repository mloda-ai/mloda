"""
Tests for the Pandas implementation of the MoonPhaseFeatureGroup.

These tests verify that the PandasMoonPhaseFeatureGroup correctly checks
for the existence of the time column, computes the moon phase in
different representations, and integrates with the generic
`calculate_feature` API to add columns to a DataFrame.
"""

import unittest
import datetime
import pandas as pd

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet

from mloda_plugins.feature_group.experimental.moon_phase.pandas import (
    PandasMoonPhaseFeatureGroup,
)


class TestPandasMoonPhaseFeatureGroup(unittest.TestCase):
    """Test cases for the PandasMoonPhaseFeatureGroup."""

    def setUp(self) -> None:
        """Set up a small DataFrame with timestamp values."""
        # Four timestamps: reference new moon, one day later, near full moon (~14.5 days later),
        # and a second near full moon point 12 hours later.  All timestamps are naive UTC.
        self.timestamps = [
            datetime.datetime(2000, 1, 6, 18, 14),
            datetime.datetime(2000, 1, 7, 18, 14),
            datetime.datetime(2000, 1, 21, 6, 14),
            datetime.datetime(2000, 1, 21, 18, 14),
        ]
        self.df = pd.DataFrame({"ts": self.timestamps})

    def test_check_time_feature_exists(self) -> None:
        """Ensure that missing time columns raise a ValueError."""
        # existing column should not raise
        PandasMoonPhaseFeatureGroup._check_time_feature_exists(self.df, "ts")
        # missing column should raise
        with self.assertRaises(ValueError):
            PandasMoonPhaseFeatureGroup._check_time_feature_exists(self.df, "missing")

    def test_calculate_moon_phase_fraction(self) -> None:
        """Test computation of moon phase fraction values."""
        result = PandasMoonPhaseFeatureGroup._calculate_moon_phase(self.df, "fraction", "ts")
        # Compute expected fractions manually using the same algorithm
        synodic_month = PandasMoonPhaseFeatureGroup.SYNODIC_MONTH
        ref = PandasMoonPhaseFeatureGroup.REFERENCE_DATE
        expected = []
        for ts in self.timestamps:
            diff_days = (ts - ref).total_seconds() / 86400.0
            phase_fraction = (diff_days % synodic_month) / synodic_month
            expected.append(phase_fraction)
        # Compare result values to expected
        for res, exp in zip(result, expected):
            self.assertAlmostEqual(res, exp, delta=1e-6)

    def test_calculate_moon_phase_degrees(self) -> None:
        """Test computation of moon phase angle in degrees."""
        result = PandasMoonPhaseFeatureGroup._calculate_moon_phase(self.df, "degrees", "ts")
        synodic_month = PandasMoonPhaseFeatureGroup.SYNODIC_MONTH
        ref = PandasMoonPhaseFeatureGroup.REFERENCE_DATE
        expected = []
        for ts in self.timestamps:
            diff_days = (ts - ref).total_seconds() / 86400.0
            phase_fraction = (diff_days % synodic_month) / synodic_month
            expected.append(phase_fraction * 360.0)
        for res, exp in zip(result, expected):
            self.assertAlmostEqual(res, exp, delta=1e-6)

    def test_calculate_moon_phase_is_full(self) -> None:
        """Test computation of the full moon indicator."""
        result = PandasMoonPhaseFeatureGroup._calculate_moon_phase(self.df, "is_full", "ts")
        # The first two timestamps are near new moon and should not be full; the last two are near full moon
        expected = [0, 0, 1, 1]
        self.assertEqual(list(result), expected)

    def test_calculate_feature_api(self) -> None:
        """Test integration with the calculate_feature API to add features to a DataFrame."""
        feature_set = FeatureSet()
        # Add three features using string-based names
        feature_set.add(Feature("moon_phase_fraction__ts"))
        feature_set.add(Feature("moon_phase_degrees__ts"))
        feature_set.add(Feature("moon_phase_is_full__ts"))
        # Calculate features on a copy of the DataFrame
        result_df = PandasMoonPhaseFeatureGroup.calculate_feature(self.df.copy(), feature_set)
        # Verify that new columns exist
        self.assertIn("moon_phase_fraction__ts", result_df.columns)
        self.assertIn("moon_phase_degrees__ts", result_df.columns)
        self.assertIn("moon_phase_is_full__ts", result_df.columns)
        # Spot-check one value for each representation
        # Index 2 (near full moon) should have fraction close to ~0.49, degrees ~176-183, and is_full == 1
        frac_val = result_df.loc[2, "moon_phase_fraction__ts"]
        deg_val = result_df.loc[2, "moon_phase_degrees__ts"]
        full_val = result_df.loc[2, "moon_phase_is_full__ts"]
        self.assertAlmostEqual(frac_val, 0.4910, delta=0.01)
        self.assertAlmostEqual(deg_val, 176.76, delta=3.0)
        self.assertEqual(full_val, 1)
