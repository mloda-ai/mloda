from pathlib import Path
from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.user import DataAccessCollection, FeatureName, PluginCollector, mloda
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Options
from mloda.user import GlobalFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.user import ParallelizationMode
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Features
from mloda.provider import FeatureSet
from mloda.provider import ComputeFramework
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_ALL


class GlobalFilterBasicTest(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        for filter in features.filters:  # type: ignore
            if filter.filter_type != "equal" or filter.parameter.value != 1:
                raise ValueError("Test Filter not found")
        return pa.table({cls.get_class_name(): [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class GlobalFilterFromDifferentColumnTest(GlobalFilterBasicTest):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"GlobalFilterFromDifferentColumn1", "GlobalFilterFromDifferentColumn2"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        if "GlobalFilterFromDifferentColumn2" not in features.get_all_names():
            raise ValueError("Filter feature not found.")

        for feat in features.features:
            if feat.name == "GlobalFilterFromDifferentColumn2":
                if feat.initial_requested_data is not False:
                    raise ValueError("Filter should not lead to automatic requested data.")

        return pa.table({"GlobalFilterFromDifferentColumn1": [1, 2, 3], "GlobalFilterFromDifferentColumn2": [1, 2, 3]})


class GlobalFilterHasDifferentNameTest(GlobalFilterBasicTest):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"GlobalFilterHasDifferentName1", "GlobalFilterHasDifferentName2"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if len(features.filters) != 1:  # type: ignore
            raise ValueError("Test Filter not found")

        if next(iter(features.filters)).filter_feature.name != "GlobalFilterHasDifferentNameTest":  # type: ignore
            raise ValueError("Filter feature name is not equal to eature name.")

        if len(features.get_all_names()) != 1:
            raise ValueError("Filter feature is same like normal feature.")

        return pa.table({"GlobalFilterHasDifferentNameTest": [1, 2, 3], "GlobalFilterHasDifferentName2": [1, 2, 3]})

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        return FeatureName(name="GlobalFilterHasDifferentNameTest")


@PARALLELIZATION_MODES_ALL
class TestGlobalFilter:
    def get_features(self, feature_list: list[str], options: dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def test_basic_global_filter(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        global_filter_test_basic = "GlobalFilterBasicTest"

        features = self.get_features([global_filter_test_basic])

        global_filter = GlobalFilter()
        global_filter.add_filter(global_filter_test_basic, "equal", {"value": 1})

        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {global_filter_test_basic: [1]}

        result_global_filter = runner.execution_planner.global_filter
        assert result_global_filter.filters == global_filter.filters  # type: ignore
        assert isinstance(result_global_filter, GlobalFilter)

        # test registration of used filters
        for key, value in result_global_filter.collection.items():
            assert issubclass(key[0], FeatureGroup)
            assert isinstance(key[1], FeatureName)
            assert key[0].get_class_name() == global_filter_test_basic
            assert str(key[1]) == global_filter_test_basic

            assert isinstance(value, set)
            assert len(value) == 1

            single_feature = next(iter(value))
            assert single_feature.filter_feature.name == global_filter_test_basic
            assert single_feature.filter_feature.name == global_filter_test_basic
            assert next(iter(global_filter.filters)).uuid == single_feature.uuid

    def test_global_filter_filter_requests_other_column(
        self, modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        base_feature_name = "GlobalFilterFromDifferentColumn"

        features = self.get_features([f"{base_feature_name}1"])

        global_filter = GlobalFilter()

        # We could have parametized this test, but we don t want to add too many tests for no reason.
        if ParallelizationMode.MULTIPROCESSING in modes:
            filter_feat: str | Feature = f"{base_feature_name}2"
        else:
            filter_feat = Feature(name=f"{base_feature_name}2")

        global_filter.add_filter(filter_feat, "equal", {"value": 1})
        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {"GlobalFilterFromDifferentColumn1": [1]}

    def test_global_filter_filter_has_different_name(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        base_feature_name = "GlobalFilterHasDifferentName"

        features = self.get_features([f"{base_feature_name}1"])

        global_filter = GlobalFilter()

        # We could have parametized this test, but we don t want to add too many tests for no reason.
        if ParallelizationMode.MULTIPROCESSING not in modes:
            filter_feat: str | Feature = f"{base_feature_name}2"
        else:
            filter_feat = Feature(name=f"{base_feature_name}2")

        global_filter.add_filter(filter_feat, "equal", {"value": 1})
        runner = MlodaTestRunner.run_engine(
            features, parallelization_modes=modes, flight_server=flight_server, global_filter=global_filter
        )

        for result in runner.get_result():
            res = result.to_pydict()
            assert res == {"GlobalFilterHasDifferentNameTest": [1]}


class TestGlobalFilterOnReaderBackedColumn:
    """A filter on a CSV column keeps its reader match off the options and still filters."""

    @staticmethod
    def _csv(tmp_path: Path) -> str:
        path = tmp_path / "gf_reader_backed.csv"
        path.write_text("gf_csv_key,gf_csv_val\n1,10\n2,20\n3,30\n")
        return str(path)

    def test_filter_on_csv_column_loads_filtered_rows(self, tmp_path: Path) -> None:
        global_filter = GlobalFilter()
        global_filter.add_filter("gf_csv_val", "equal", {"value": 20})

        results = mloda.run_all(
            [Feature("gf_csv_val")],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(files={self._csv(tmp_path)}),
            plugin_collector=PluginCollector.enabled_feature_groups({ReadFileFeature}),
            global_filter=global_filter,
        )

        assert len(results) == 1
        assert results[0].to_pydict() == {"gf_csv_val": [20]}

    def test_matched_filter_feature_carries_the_pair_on_input_data_match(self, tmp_path: Path) -> None:
        csv_path = self._csv(tmp_path)
        global_filter = GlobalFilter()
        global_filter.add_filter("gf_csv_val", "equal", {"value": 20})

        matched = global_filter.identify_matched_filters(
            ReadFileFeature, Feature("gf_csv_val"), DataAccessCollection(files={csv_path})
        )

        assert len(matched) == 1
        filter_feature = next(iter(matched)).filter_feature
        assert filter_feature.input_data_match == (CsvReader, csv_path)
        assert RESERVED_READER_OPTION_KEY not in filter_feature.options.group
        assert RESERVED_READER_OPTION_KEY not in filter_feature.options.context


class TestGlobalFilterFromAnotherSource:
    """A filter whose column lives in a different reader source than the host is dropped, not attached."""

    @staticmethod
    def _two_files(tmp_path: Path) -> tuple[str, str]:
        one = tmp_path / "gf_other_one.csv"
        one.write_text("gf_other_a,gf_other_k\n1,1\n2,2\n3,3\n")
        two = tmp_path / "gf_other_two.csv"
        two.write_text("gf_other_b,gf_other_k2\n10,1\n20,2\n30,3\n")
        return str(one), str(two)

    def _run(self, global_filter: GlobalFilter, feature: str, files: set[str]) -> list[Any]:
        return list(
            mloda.run_all(
                [Feature(feature)],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=DataAccessCollection(files=files),
                plugin_collector=PluginCollector.enabled_feature_groups({ReadFileFeature}),
                global_filter=global_filter,
            )
        )

    def test_filter_on_other_file_is_dropped_and_recorded(self, tmp_path: Path) -> None:
        one, two = self._two_files(tmp_path)
        global_filter = GlobalFilter()
        global_filter.add_filter("gf_other_b", "equal", {"value": 20})

        results = self._run(global_filter, "gf_other_a", {one, two})

        assert results[0].to_pydict() == {"gf_other_a": [1, 2, 3]}
        assert len(global_filter.dropped_filters) == 1
        reason = " ".join(str(getattr(r, "reason", r)) for r in global_filter.dropped_filters.values())
        assert one in reason
        assert two in reason

    def test_filter_on_same_file_still_filters(self, tmp_path: Path) -> None:
        one, two = self._two_files(tmp_path)
        global_filter = GlobalFilter()
        global_filter.add_filter("gf_other_k", "equal", {"value": 2})

        results = self._run(global_filter, "gf_other_a", {one, two})

        assert results[0].to_pydict() == {"gf_other_a": [2]}
        assert global_filter.dropped_filters == {}


class TestGlobalFilterOnFormatGroupColumn:
    """A filter on a format-group column keeps its source match off the options."""

    def test_matched_filter_feature_carries_the_source_match(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
            ToyFormatFG,
            toy_dac,
        )

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_val", "equal", {"value": 20})

        matched = global_filter.identify_matched_filters(
            ToyFormatFG, Feature("gf_toy_val"), toy_dac(h1={"gf_toy_val": [10, 20]})
        )

        assert len(matched) == 1
        filter_feature = next(iter(matched)).filter_feature
        pair = filter_feature.input_data_match
        assert pair is not None
        assert pair[0] is ToyFormatFG
        assert pair[1].source == "h1:toy"
        assert RESERVED_READER_OPTION_KEY not in filter_feature.options.group
        assert RESERVED_READER_OPTION_KEY not in filter_feature.options.context

    def test_filter_on_a_column_absent_from_every_source_is_not_matched(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
            ToyFormatFG,
            toy_dac,
        )

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_missing", "equal", {"value": 1})

        matched = global_filter.identify_matched_filters(
            ToyFormatFG, Feature("gf_toy_val"), toy_dac(h1={"gf_toy_val": [10]})
        )

        assert len(matched) == 0

    def test_class_name_key_carried_over_with_a_missing_filter_column_drops_the_filter(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import ToyFormatFG

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_missing", "equal", {"value": 1})
        host = Feature("gf_toy_val", Options({"ToyFormatFG": {"gf_toy_val": [10]}}))

        matched = global_filter.identify_matched_filters(ToyFormatFG, host, None)

        assert len(matched) == 0
        assert [e.stage for (fg, _, _), e in global_filter.dropped_filters.items() if fg is ToyFormatFG] == [
            "input_data"
        ]

    def test_filter_column_in_two_sources_drops_the_filter_instead_of_raising(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import (
            ToyFormatFG,
            toy_dac,
        )

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_val", "equal", {"value": 10})

        matched = global_filter.identify_matched_filters(
            ToyFormatFG, Feature("gf_toy_val"), toy_dac(h1={"gf_toy_val": [10]}, h2={"gf_toy_val": [10]})
        )

        assert len(matched) == 0
        assert [e.stage for (fg, _, _), e in global_filter.dropped_filters.items() if fg is ToyFormatFG] == [
            "input_data"
        ]
