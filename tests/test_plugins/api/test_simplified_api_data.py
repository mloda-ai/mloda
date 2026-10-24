"""
Test simplified mloda data parameter for mloda.run_all().

These tests verify that run_all() accepts api_data as a simple dict.
The ApiDataFG source group serves the dict sets automatically.

Simplified mloda:
    result = mloda.run_all(
        features,
        api_data={"KeyName": {"col1": [1, 2], "col2": ["a", "b"]}}
    )
"""

from typing import Any

import pytest

from mloda.user import mloda
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Link, JoinSpec
from mloda.user import GlobalFilter, Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda.provider import ApiDataFG, ClaimRoute, NamePolicy, SourceFeatureGroup, SourceMatch
from mloda.user import ParallelizationMode
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG  # noqa: F401
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.api_data_scope import ApiDataRef, api_data_scope, read_api_data


# ============================================================================
# Test Feature: Simple mloda Consumer
# ============================================================================
class SimpleApiFeature(FeatureGroup):
    """A simple feature that consumes mloda data."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Define input features from mloda data."""
        return {
            Feature(name="api_id", index=Index(("api_id",))),
            Feature(name="api_value", index=Index(("api_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Concatenate mloda columns."""
        data["SimpleApiFeature"] = data["api_id"].astype(str) + "_" + data["api_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Feature: Multi-key mloda Consumer
# ============================================================================
class MultiKeyApiFeature(FeatureGroup):
    """A feature that consumes data from multiple mloda keys."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Define input features from multiple mloda data sources."""
        # This will require data from both FirstKey and SecondKey
        return {
            Feature(name="first_id", index=Index(("first_id",))),
            Feature(name="second_value", index=Index(("second_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Combine data from multiple keys."""
        data["MultiKeyApiFeature"] = data["first_id"].astype(str) + "_" + data["second_value"]
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Feature: Joined with Creator
# ============================================================================
class CreatorDataFeature(FeatureGroup):
    """Creates its own data via DataCreator."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"creator_id", "creator_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"creator_id": [1, 2, 3], "creator_value": ["a", "b", "c"]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"creator_id", "creator_value"}


class SimplifiedApiJoinFeature(FeatureGroup):
    """Joins simplified mloda data with Creator data using LEFT join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Define input features with a LEFT join link."""
        # Create the link: LEFT join on api_id = creator_id
        link = Link.left(JoinSpec(ApiDataFG, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",))))

        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Combine the joined data
        data["SimplifiedApiJoinFeature"] = data["api_value"].fillna("") + "_" + data["creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Class
# ============================================================================
class TestSimplifiedApiData:
    """Tests for simplified api_data parameter in mloda.run_all()."""

    _enabled_simple = PluginCollector.enabled_feature_groups(
        {
            ApiDataFG,
            SimpleApiFeature,
        }
    )

    _enabled_multikey = PluginCollector.enabled_feature_groups(
        {
            ApiDataFG,
            MultiKeyApiFeature,
        }
    )

    _enabled_join = PluginCollector.enabled_feature_groups(
        {
            ApiDataFG,
            CreatorDataFeature,
            SimplifiedApiJoinFeature,
        }
    )

    def test_simplified_api_data_single_key(self) -> None:
        """
        Test that run_all() accepts simplified api_data dict with a single key.

        This test verifies the NEW simplified mloda:
            api_data={"KeyName": {"col1": [...], "col2": [...]}}

        Instead of the OLD verbose mloda:
            collection = <verbose collection setup>
            run_all(..., <collection>, api_data=...)

        Expected behavior:
        - run_all() should serve the dict through ApiDataFG
        - api_data dict keys become the collection key names
        - api_data dict values contain the actual column data
        """
        # NEW simplified mloda - just pass a dict!
        api_data = {
            "ApiExample": {
                "api_id": [1, 2, 3, 4],
                "api_value": ["w", "x", "y", "z"],
            }
        }

        feature_list: list[Feature | str] = [Feature(name="SimpleApiFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_simple,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "SimpleApiFeature" in df.columns
        assert len(df) == 4
        assert df["SimpleApiFeature"].tolist() == ["1_w", "2_x", "3_y", "4_z"]

    def test_simplified_api_data_multiple_keys(self) -> None:
        """
        Test that run_all() accepts simplified api_data dict with multiple keys.

        This verifies that multiple mloda data sources can be registered in a single dict.
        The test uses only one key's data - combining data from multiple keys requires Links.
        """
        api_data = {
            "FirstKey": {
                "api_id": [1, 2],
                "api_value": ["a", "b"],
            },
            "SecondKey": {
                "second_id": [3, 4],
                "second_value": ["c", "d"],
            },
        }

        # Use SimpleApiFeature which only needs api_id and api_value (from FirstKey)
        feature_list: list[Feature | str] = [Feature(name="SimpleApiFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_simple,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
        )

        # Validate result - should work even with extra unused keys
        assert len(result) == 1
        df = result[0]
        assert "SimpleApiFeature" in df.columns
        assert len(df) == 2
        assert df["SimpleApiFeature"].tolist() == ["1_a", "2_b"]

    def test_requested_features_from_two_api_data_sets_return_separate_results(self) -> None:
        result = mloda.run_all(
            ["a", "b"],
            compute_frameworks=[PythonDictFramework],
            api_data={"first": {"a": ["one"]}, "second": {"b": ["one", "two"]}},
        )

        frames = [dict(frame) for frame in result]
        assert len(frames) == 2, f"both requested features must be returned in separate frames, got {frames!r}"
        assert {next(iter(frame)) for frame in frames} == {"a", "b"}
        assert {tuple(frame): tuple(next(iter(frame.values()))) for frame in frames} == {
            ("a",): ("one",),
            ("b",): ("one", "two"),
        }

    def test_features_from_same_api_data_set_share_a_frame(self) -> None:
        result = mloda.run_all(
            ["a", "a2", "b"],
            compute_frameworks=[PythonDictFramework],
            api_data={"first": {"a": [1], "a2": [2]}, "second": {"b": [1, 2]}},
        )

        frames = sorted((dict(frame) for frame in result), key=lambda frame: sorted(frame))
        assert frames == [{"a": [1], "a2": [2]}, {"b": [1, 2]}]

    def test_derived_feature_needing_two_api_data_sets_raises_without_link(self) -> None:
        with pytest.raises(ValueError, match=r"(?s)MultiKeyApiFeature.*missing Links"):
            mloda.run_all(
                [Feature(name="MultiKeyApiFeature")],
                plugin_collector=self._enabled_multikey,
                compute_frameworks=[PandasDataFrame],
                api_data={"First": {"first_id": [1, 2]}, "Second": {"second_id": [1, 2], "second_value": ["x", "y"]}},
            )

    def test_missing_links_error_ignores_unrelated_api_source_partition(self) -> None:
        """The missing-Links error must not blame an option that the spanned source-key partitions never differed by."""
        with pytest.raises(ValueError, match=r"(?s)MultiKeyApiFeature.*missing Links") as exc_info:
            mloda.run_all(
                [
                    Feature(name="MultiKeyApiFeature"),
                    Feature(name="first_id", options={"unit": "x"}, index=Index(("first_id",))),
                ],
                plugin_collector=self._enabled_multikey,
                compute_frameworks=[PandasDataFrame],
                api_data={"First": {"first_id": [1, 2]}, "Second": {"second_id": [1, 2], "second_value": ["x", "y"]}},
            )

        assert "differing option" not in str(exc_info.value).lower(), (
            "MultiKeyApiFeature spans two ApiDataFG source-key partitions with identical "
            "Options; the unrelated 'unit' bucket from a third, disjoint request must not be blamed"
        )

    def test_duplicate_column_across_api_data_sets_raises(self) -> None:
        with pytest.raises(ValueError, match=r"(?s)(?=.*'a')(?=.*first)(?=.*second)"):
            mloda.run_all(
                ["a"],
                compute_frameworks=[PythonDictFramework],
                api_data={"first": {"a": [1]}, "second": {"a": [1, 2]}},
            )

    def test_simplified_api_data_two_keys_first_key_used(self) -> None:
        """
        Test that run_all() can register multiple mloda keys and use the first key's data.

        This verifies that:
        - Multiple mloda keys can be registered in a single api_data dict
        - The first key's data is accessible via its column names
        - Extra unused keys don't break the registration
        """
        api_data = {
            "FirstKey": {
                "first_id": [1, 2, 3],
                "first_value": ["a", "b", "c"],
            },
            "SecondKey": {
                "second_id": [10, 20],
                "second_value": ["x", "y"],
            },
        }

        # Request feature from first key
        feature_list: list[Feature | str] = ["first_value"]

        result = mloda.run_all(
            feature_list,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
        )

        assert len(result) == 1
        df = result[0]
        assert "first_value" in df.columns
        assert len(df) == 3
        assert df["first_value"].tolist() == ["a", "b", "c"]

    def test_simplified_api_data_two_keys_second_key_used(self) -> None:
        """
        Test that run_all() can register multiple mloda keys and use the second key's data.

        This verifies that:
        - Multiple mloda keys can be registered in a single api_data dict
        - The second key's data is accessible via its column names
        - Extra unused keys don't break the registration
        """
        api_data = {
            "FirstKey": {
                "first_id": [1, 2, 3],
                "first_value": ["a", "b", "c"],
            },
            "SecondKey": {
                "second_id": [10, 20],
                "second_value": ["x", "y"],
            },
        }

        # Request feature from second key
        feature_list: list[Feature | str] = ["second_value"]

        result = mloda.run_all(
            feature_list,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
        )

        assert len(result) == 1
        df = result[0]
        assert "second_value" in df.columns
        assert len(df) == 2
        assert df["second_value"].tolist() == ["x", "y"]

    def test_simplified_api_data_with_joins(self) -> None:
        """
        Test that simplified api_data works with Links/Joins.

        This ensures the simplified mloda is compatible with existing
        Link functionality (LEFT join, APPEND, etc).
        """
        api_data = {
            "ApiExample": {
                "api_id": [1, 2, 3, 4],
                "api_value": ["w", "x", "y", "z"],
            }
        }

        feature_list: list[Feature | str] = [Feature(name="SimplifiedApiJoinFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_join,
            compute_frameworks=[PandasDataFrame],
            api_data=api_data,
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "SimplifiedApiJoinFeature" in df.columns
        assert len(df) == 4  # LEFT join keeps all mloda rows

    def test_none_api_data_still_works(self) -> None:
        """
        Test that passing None for api_data still works (no regression).

        This ensures backward compatibility for features that don't use mloda data.
        """
        feature_list: list[Feature | str] = [Feature(name="creator_id")]

        _enabled_creator_only = PluginCollector.enabled_feature_groups({CreatorDataFeature})

        result = mloda.run_all(
            feature_list,
            plugin_collector=_enabled_creator_only,
            compute_frameworks=[PandasDataFrame],
            api_data=None,  # Explicitly None
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "creator_id" in df.columns
        assert len(df) == 3

    def test_empty_api_data_dict(self) -> None:
        """
        Test that passing an empty api_data dict behaves correctly.

        Should either work (no mloda data) or raise a clear error.
        """
        feature_list: list[Feature | str] = [Feature(name="creator_id")]

        _enabled_creator_only = PluginCollector.enabled_feature_groups({CreatorDataFeature})

        result = mloda.run_all(
            feature_list,
            plugin_collector=_enabled_creator_only,
            compute_frameworks=[PandasDataFrame],
            api_data={},  # Empty dict
        )

        # Validate result
        assert len(result) == 1
        df = result[0]
        assert "creator_id" in df.columns
        assert len(df) == 3


class AmbiguousApiColumnFeature(FeatureGroup):
    """Computes a name that an api_data set also provides."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"simplified_api_ambiguous_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"simplified_api_ambiguous_col": [9]}


class SimplifiedApiRogueFG(SourceFeatureGroup):
    """Not an ApiDataFG, yet hands out a match carrying an ApiDataRef."""

    CLAIM_ROUTES = (ClaimRoute("api_data", NamePolicy.CHECKED, True),)

    @classmethod
    def find_sources(cls, route: Any, feature_name: str, options: Any, data_access_collection: Any) -> list[Any]:
        if feature_name != "simplified_api_rogue_col":
            return []
        return [SourceMatch("api_data:Rogue", ApiDataRef("Rogue", ("simplified_api_rogue_col",)))]

    @classmethod
    def columns(cls, match: SourceMatch) -> Any:
        return ("simplified_api_rogue_col",)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return read_api_data("Rogue")


class SimplifiedApiDoublingFG(ApiDataFG):
    """Custom transport: serves the api values doubled; inert for other names."""

    @classmethod
    def is_pointed(cls, feature_name: str, options: Any, data_access_collection: Any) -> bool:
        return feature_name.startswith("simplified_api_double")

    @classmethod
    def find_sources(
        cls, route: Any, feature_name: str, options: Any, data_access_collection: Any
    ) -> list[SourceMatch]:
        if not feature_name.startswith("simplified_api_double"):
            return []
        return super().find_sources(route, feature_name, options, data_access_collection)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        values = cls.api_values(match)
        return {name: [v * 2 for v in column] for name, column in values.items()}


def _compute_step(session: Any) -> Any:
    return next(step for step in session.resolved_plan() if step.step_kind == "compute")


class TestApiDataSourceGroupBehavior:
    """Resolution, introspection, filtering and sibling roots for api_data served by ApiDataFG."""

    _api_only = PluginCollector.enabled_feature_groups({ApiDataFG})

    def test_describe_columns_lists_the_prepare_time_columns(self) -> None:
        session = mloda.prepare(
            ["api_id"],
            compute_frameworks=[PandasDataFrame],
            api_data={"Cols": {"api_id": [1], "api_value": ["a"]}},
            plugin_collector=self._api_only,
        )

        assert set(session.describe_columns(_compute_step(session))) == {"api_id", "api_value"}

    def test_explain_resolves_api_data_features(self) -> None:
        steps = mloda.explain(
            ["api_id"],
            compute_frameworks=[PandasDataFrame],
            api_data={"Cols": {"api_id": [1], "api_value": ["a"]}},
            plugin_collector=self._api_only,
        )

        assert any(step.feature_group is ApiDataFG and "api_id" in step.feature_names for step in steps)

    def test_group_computing_an_api_column_name_is_ambiguous(self) -> None:
        collector = PluginCollector.enabled_feature_groups({ApiDataFG, AmbiguousApiColumnFeature})

        with pytest.raises(ValueError, match=r"(?s)(?=.*ApiDataFG)(?=.*AmbiguousApiColumnFeature)"):
            mloda.run_all(
                ["simplified_api_ambiguous_col"],
                compute_frameworks=[PandasDataFrame],
                api_data={"Cols": {"simplified_api_ambiguous_col": [1, 2]}},
                plugin_collector=collector,
            )

    def test_column_in_no_api_data_set_fails_resolution_naming_the_column(self) -> None:
        with pytest.raises(ValueError, match="missing_col"):
            mloda.run_all(
                ["missing_col"],
                compute_frameworks=[PandasDataFrame],
                api_data={"Cols": {"api_id": [1]}},
                plugin_collector=self._api_only,
            )

    def test_pointer_narrows_to_the_named_set(self) -> None:
        result = mloda.run_all(
            [Feature(name="only_second", options=Options({"ApiDataFG": "second"}))],
            compute_frameworks=[PythonDictFramework],
            api_data={"first": {"other": [1]}, "second": {"only_second": [5, 6]}},
        )

        assert [dict(frame) for frame in result] == [{"only_second": [5, 6]}]

    def test_unknown_pointer_key_aborts_naming_the_known_keys(self) -> None:
        with pytest.raises(ValueError, match=r"(?s)(?=.*nope)(?=.*first)(?=.*second)"):
            mloda.run_all(
                [Feature(name="only_second", options=Options({"ApiDataFG": "nope"}))],
                compute_frameworks=[PythonDictFramework],
                api_data={"first": {"other": [1]}, "second": {"only_second": [5, 6]}},
            )

    def test_global_filter_filters_api_data_rows(self) -> None:
        global_filter = GlobalFilter()
        global_filter.add_filter("api_num", "min", {"value": 2})

        result = mloda.run_all(
            ["api_num"],
            compute_frameworks=[PandasDataFrame],
            api_data={"Nums": {"api_num": [1, 2, 3]}},
            global_filter=global_filter,
            plugin_collector=self._api_only,
        )

        assert result[0]["api_num"].tolist() == [2, 3]

    def test_sibling_root_group_and_api_data_set_stay_in_separate_frames(self) -> None:
        collector = PluginCollector.enabled_feature_groups({ApiDataFG, CreatorDataFeature})

        result = mloda.run_all(
            ["creator_id", "api_num"],
            compute_frameworks=[PandasDataFrame],
            api_data={"Nums": {"api_num": [7, 8]}},
            plugin_collector=collector,
        )

        frames = {tuple(frame.columns): frame for frame in result}
        assert set(frames) == {("creator_id",), ("api_num",)}
        assert frames[("creator_id",)]["creator_id"].tolist() == [1, 2, 3]
        assert frames[("api_num",)]["api_num"].tolist() == [7, 8]


class TestApiDataScopeAndShapes:
    """Scope binding, subclass transport, THREADING, non-mapping values and foreign-pointer claims."""

    def test_read_outside_any_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="Cols"):
            read_api_data("Cols")

    def test_read_with_a_different_key_than_the_bound_one_raises(self) -> None:
        with api_data_scope(lambda: {"x": [1]}, "bound_key"):
            assert read_api_data("bound_key") == {"x": [1]}
            with pytest.raises(ValueError, match=r"(?s)(?=.*other_key)"):
                read_api_data("other_key")

    def test_non_api_data_group_cannot_read_the_store(self) -> None:
        with pytest.raises(Exception, match="Rogue"):
            mloda.run_all(
                ["simplified_api_rogue_col"],
                compute_frameworks=[PythonDictFramework],
                api_data={"Rogue": {"simplified_api_rogue_col": [1]}},
                plugin_collector=PluginCollector.enabled_feature_groups({SimplifiedApiRogueFG}),
            )

    def test_subclass_transforms_values_through_api_values(self) -> None:
        result = mloda.run_all(
            ["simplified_api_double_col"],
            compute_frameworks=[PythonDictFramework],
            api_data={"Dbl": {"simplified_api_double_col": [1, 2]}},
            plugin_collector=PluginCollector.enabled_feature_groups({SimplifiedApiDoublingFG}),
        )

        assert dict(result[0]) == {"simplified_api_double_col": [2, 4]}

    def test_threading_mode_run_returns_the_data(self) -> None:
        result = mloda.run_all(
            ["simplified_api_thread_col"],
            compute_frameworks=[PythonDictFramework],
            api_data={"Thr": {"simplified_api_thread_col": [3, 4]}},
            plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG}),
            parallelization_modes={ParallelizationMode.THREADING},
        )

        assert dict(result[0]) == {"simplified_api_thread_col": [3, 4]}

    def test_prepare_refuses_non_mapping_set_without_leaking_it(self) -> None:
        secret = "simplified_api_secret_value_z9"

        with pytest.raises(ValueError, match="Bad") as raised:
            mloda.prepare(
                ["simplified_api_bad_col"],
                compute_frameworks=[PythonDictFramework],
                api_data={"Bad": secret},  # type: ignore[dict-item]
                plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG}),
            )

        assert secret not in str(raised.value)
        assert "str" in str(raised.value)

    def test_column_names_in_messages_are_sorted_as_strings(self) -> None:
        with pytest.raises(ValueError, match=r"(?s)PrepareKey.*differs"):
            session = mloda.prepare(
                ["simplified_api_x"],
                compute_frameworks=[PythonDictFramework],
                api_data={"PrepareKey": {"simplified_api_x": [1]}},
                plugin_collector=PluginCollector.enabled_feature_groups({ApiDataFG}),
            )
            session.run(api_data={"PrepareKey": {"simplified_api_x": [1], 2: [2], "b": [3]}})  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        "options",
        [{"CsvFG": "some.csv"}, {"query_text": "select 1"}],
        ids=["foreign_pointer", "query_text"],
    )
    def test_column_name_is_not_pointed_with_foreign_pointer_or_query_text(self, options: dict[str, Any]) -> None:
        dac = DataAccessCollection()
        dac._api_data_columns = {"Cols": ("simplified_api_pointed_col",)}

        assert ApiDataFG.is_pointed("simplified_api_pointed_col", Options(options), dac) is False

    def test_column_name_without_pointer_stays_pointed(self) -> None:
        dac = DataAccessCollection()
        dac._api_data_columns = {"Cols": ("simplified_api_pointed_col",)}

        assert ApiDataFG.is_pointed("simplified_api_pointed_col", Options({}), dac) is True
