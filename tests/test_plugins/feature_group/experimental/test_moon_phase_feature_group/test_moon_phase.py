"""
Basic tests for the MoonPhaseFeatureGroup base class.

These tests verify that the moon phase feature group correctly parses
feature names to extract the representation type and source time
feature, matches feature name patterns, and returns the appropriate
input features.  Invalid feature names should trigger exceptions.
"""

import unittest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.moon_phase.base import MoonPhaseFeatureGroup


class TestMoonPhaseFeatureGroup(unittest.TestCase):
    """Test cases for the MoonPhaseFeatureGroup base implementation."""

    def test_extract_parameters_from_string_name(self) -> None:
        """Test extracting representation and source feature from string definitions."""
        # fraction representation
        feature = Feature("moon_phase_fraction__ts")
        rep, src = MoonPhaseFeatureGroup._extract_moon_phase_parameters(feature)
        self.assertEqual(rep, "fraction")
        self.assertEqual(src, "ts")

        # degrees representation
        feature = Feature("moon_phase_degrees__timestamp")
        rep, src = MoonPhaseFeatureGroup._extract_moon_phase_parameters(feature)
        self.assertEqual(rep, "degrees")
        self.assertEqual(src, "timestamp")

        # is_full representation
        feature = Feature("moon_phase_is_full__my_time")
        rep, src = MoonPhaseFeatureGroup._extract_moon_phase_parameters(feature)
        self.assertEqual(rep, "is_full")
        self.assertEqual(src, "my_time")

    def test_extract_parameters_from_options(self) -> None:
        """Test extracting representation and source feature via configuration options."""
        # prepare feature with options specifying source feature and representation
        options = Options()
        # set the source feature via mloda context
        options.context["mloda_source_feature"] = {Feature("dt")}
        options.context[MoonPhaseFeatureGroup.PHASE_REPRESENTATION] = "degrees"
        feature = Feature("moon_phase", options=options)
        rep, src = MoonPhaseFeatureGroup._extract_moon_phase_parameters(feature)
        self.assertEqual(rep, "degrees")
        self.assertEqual(src, "dt")

    def test_input_features_from_string(self) -> None:
        """Test that input_features returns the correct source feature when parsing string definitions."""
        feature_group = MoonPhaseFeatureGroup()
        feature_name = FeatureName("moon_phase_fraction__event_time")
        input_feats = feature_group.input_features(Options(), feature_name)
        # input_feats should be a set containing exactly one Feature
        self.assertIsNotNone(input_feats)
        if input_feats is None:
            raise AssertionError("input_features returned None")
        self.assertEqual(len(input_feats), 1)
        self.assertIn(Feature("event_time"), input_feats)

    def test_input_features_from_options(self) -> None:
        """Test that input_features returns the correct source feature when using options."""
        feature_group = MoonPhaseFeatureGroup()
        feature_name = FeatureName("moon_phase")  # name without source encoded
        options = Options()
        options.context["mloda_source_feature"] = {Feature("my_timestamp")}
        options.context[MoonPhaseFeatureGroup.PHASE_REPRESENTATION] = "fraction"
        input_feats = feature_group.input_features(options, feature_name)
        self.assertIsNotNone(input_feats)
        if input_feats is None:
            raise AssertionError("input_features returned None")
        self.assertEqual(len(input_feats), 1)
        self.assertIn(Feature("my_timestamp"), input_feats)

    def test_invalid_representation_raises(self) -> None:
        """Test that invalid representations raise a ValueError."""
        feature = Feature("moon_phase_invalid__ts")
        with self.assertRaises(ValueError):
            MoonPhaseFeatureGroup._extract_moon_phase_parameters(feature)
