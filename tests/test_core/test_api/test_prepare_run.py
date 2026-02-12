"""
Tests for the prepare() / run() public API on mlodaAPI.

These tests define the contract for a two-phase execution model:
  1. prepare() - classmethod that builds the execution plan and returns a reusable mlodaAPI instance
  2. run() - instance method that executes with fresh api_data, reusing the cached plan
"""

from typing import Any, List, Optional, Set, Union

from mloda.user import mloda, mlodaAPI, Feature, PluginCollector
from mloda.provider import FeatureGroup, FeatureSet, ApiDataFeatureGroup
from mloda.user import Options, FeatureName, Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class PrepareRunApiFeature(FeatureGroup):
    """A simple feature that consumes api data for prepare/run tests."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature(name="api_id", index=Index(("api_id",))),
            Feature(name="api_value", index=Index(("api_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PrepareRunApiFeature"] = data["api_id"].astype(str) + "_" + data["api_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {cls.get_class_name()}


_enabled = PluginCollector.enabled_feature_groups(
    {
        ApiDataFeatureGroup,
        PrepareRunApiFeature,
    }
)


class TestPrepareReturnsInstance:
    """Test 1: prepare() returns an mlodaAPI instance."""

    def test_prepare_returns_mloda_api_instance(self) -> None:
        """Calling mlodaAPI.prepare() should return an mlodaAPI instance with a cached execution plan."""
        api_data = {
            "PrepareExample": {
                "api_id": [1, 2, 3],
                "api_value": ["a", "b", "c"],
            }
        }

        features: List[Union[Feature, str]] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks={PandasDataFrame},
            api_data=api_data,
            plugin_collector=_enabled,
        )

        assert isinstance(session, mlodaAPI)


class TestRunReturnsResults:
    """Test 2: run() returns a list of results."""

    def test_run_returns_results(self) -> None:
        """After prepare(), calling run() should execute the plan and return a list of results."""
        api_data = {
            "PrepareExample": {
                "api_id": [1, 2, 3],
                "api_value": ["a", "b", "c"],
            }
        }

        features: List[Union[Feature, str]] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks={PandasDataFrame},
            api_data=api_data,
            plugin_collector=_enabled,
        )

        result = session.run(api_data=api_data)

        assert isinstance(result, list)
        assert len(result) == 1
        df = result[0]
        assert "PrepareRunApiFeature" in df.columns
        assert len(df) == 3
        assert df["PrepareRunApiFeature"].tolist() == ["1_a", "2_b", "3_c"]


class TestRunMatchesRunAllOutput:
    """Test 3: run() output matches run_all() output."""

    def test_run_matches_run_all_output(self) -> None:
        """The prepare()+run() path must produce the same results as run_all()."""
        api_data = {
            "PrepareExample": {
                "api_id": [10, 20],
                "api_value": ["x", "y"],
            }
        }

        features: List[Union[Feature, str]] = [Feature(name="PrepareRunApiFeature")]

        run_all_result = mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            api_data=api_data,
            plugin_collector=_enabled,
        )

        session = mloda.prepare(
            features,
            compute_frameworks={PandasDataFrame},
            api_data=api_data,
            plugin_collector=_enabled,
        )
        prepare_run_result = session.run(api_data=api_data)

        assert len(run_all_result) == len(prepare_run_result)

        df_run_all = run_all_result[0]
        df_prepare_run = prepare_run_result[0]

        assert df_run_all["PrepareRunApiFeature"].tolist() == df_prepare_run["PrepareRunApiFeature"].tolist()


class TestMultipleSequentialRuns:
    """Test 4: Multiple sequential runs with different api_data produce correct, independent results."""

    def test_multiple_sequential_runs_with_different_api_data(self) -> None:
        """prepare() once, then run() twice with different api_data.
        Each run should produce results matching only its own input data.
        """
        initial_api_data = {
            "PrepareExample": {
                "api_id": [1],
                "api_value": ["initial"],
            }
        }

        features: List[Union[Feature, str]] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks={PandasDataFrame},
            api_data=initial_api_data,
            plugin_collector=_enabled,
        )

        first_api_data = {
            "PrepareExample": {
                "api_id": [1, 2],
                "api_value": ["a", "b"],
            }
        }
        first_result = session.run(api_data=first_api_data)

        assert len(first_result) == 1
        df_first = first_result[0]
        assert df_first["PrepareRunApiFeature"].tolist() == ["1_a", "2_b"]

        second_api_data = {
            "PrepareExample": {
                "api_id": [10, 20, 30],
                "api_value": ["x", "y", "z"],
            }
        }
        second_result = session.run(api_data=second_api_data)

        assert len(second_result) == 1
        df_second = second_result[0]
        assert df_second["PrepareRunApiFeature"].tolist() == ["10_x", "20_y", "30_z"]


class TestStepStateDoesNotLeakBetweenRuns:
    """Test 5: Internal step state does not leak between runs."""

    def test_step_state_does_not_leak_between_runs(self) -> None:
        """prepare() once, run() twice with the same data.
        Both runs must succeed and produce identical results.
        If internal step state (e.g. step_is_done flags) leaked from the first run,
        the second run would fail or produce wrong results.
        """
        api_data = {
            "PrepareExample": {
                "api_id": [5, 6],
                "api_value": ["p", "q"],
            }
        }

        features: List[Union[Feature, str]] = [Feature(name="PrepareRunApiFeature")]

        session = mloda.prepare(
            features,
            compute_frameworks={PandasDataFrame},
            api_data=api_data,
            plugin_collector=_enabled,
        )

        first_result = session.run(api_data=api_data)
        second_result = session.run(api_data=api_data)

        assert len(first_result) == 1
        assert len(second_result) == 1

        df_first = first_result[0]
        df_second = second_result[0]

        assert df_first["PrepareRunApiFeature"].tolist() == ["5_p", "6_q"]
        assert df_second["PrepareRunApiFeature"].tolist() == ["5_p", "6_q"]
        assert df_first["PrepareRunApiFeature"].tolist() == df_second["PrepareRunApiFeature"].tolist()
