import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.user import DataAccessCollection, FeatureName, PluginCollector, mloda
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG  # noqa: F401
from mloda.provider import DataCreatorFeatureGroup, SourceMatch
from mloda.user import Options
from mloda.user import GlobalFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.user import ParallelizationMode
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Features
from mloda.user import Index
from mloda.provider import FeatureSet
from mloda.provider import ComputeFramework
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_ALL


class GlobalFilterBasicTest(DataCreatorFeatureGroup):
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
    def feature_names_supported(cls) -> set[str]:
        return {"GlobalFilterFromDifferentColumn1", "GlobalFilterFromDifferentColumn2"}

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


class GlobalFilterKeepIndexDerivedTest(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == cls.get_class_name()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("GlobalFilterFromDifferentColumn1")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        doubled = pa.array([v * 2 for v in data["GlobalFilterFromDifferentColumn1"].to_pylist()])
        return data.append_column(cls.get_class_name(), doubled)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class GlobalFilterHasDifferentNameTest(GlobalFilterBasicTest):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"GlobalFilterHasDifferentName1", "GlobalFilterHasDifferentName2"}

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

    @pytest.mark.parametrize("shape", ["plain", "keep_root", "keep_derived"])
    def test_global_filter_filter_requests_other_column(
        self, modes: set[ParallelizationMode], flight_server: Any, shape: str
    ) -> None:
        base_feature_name = "GlobalFilterFromDifferentColumn"
        key = Index((f"{base_feature_name}2",))

        if shape == "keep_root":
            features = Features(
                [Feature(f"{base_feature_name}1", index=key, keep_index=True, initial_requested_data=True)]
            )
            expected = {f"{base_feature_name}1": [1], f"{base_feature_name}2": [1]}
        elif shape == "keep_derived":
            derived = GlobalFilterKeepIndexDerivedTest.get_class_name()
            features = Features([Feature(derived, index=key, keep_index=True, initial_requested_data=True)])
            expected = {derived: [2], f"{base_feature_name}2": [1]}
        else:
            features = self.get_features([f"{base_feature_name}1"])
            expected = {f"{base_feature_name}1": [1]}

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

        results = runner.get_result()
        assert len(results) == 1
        for result in results:
            assert result.to_pydict() == expected

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
            plugin_collector=PluginCollector.enabled_feature_groups({CsvFG}),
            global_filter=global_filter,
        )

        assert len(results) == 1
        assert results[0].to_pydict() == {"gf_csv_val": [20]}

    def test_matched_filter_feature_carries_the_pair_on_input_data_match(self, tmp_path: Path) -> None:
        csv_path = self._csv(tmp_path)
        global_filter = GlobalFilter()
        global_filter.add_filter("gf_csv_val", "equal", {"value": 20})

        matched = global_filter.identify_matched_filters(
            CsvFG, Feature("gf_csv_val"), DataAccessCollection(files={csv_path})
        )

        assert len(matched) == 1
        filter_feature = next(iter(matched)).filter_feature
        assert filter_feature.input_data_match == (CsvFG, SourceMatch(source=csv_path, access=csv_path))
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
                plugin_collector=PluginCollector.enabled_feature_groups({CsvFG}),
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


class TestGlobalFilterOnSourceGroupColumn:
    """A filter on a source-group column keeps its source match off the options."""

    def test_matched_filter_feature_carries_the_source_match(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
            ToySourceFG,
            toy_dac,
        )

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_val", "equal", {"value": 20})

        matched = global_filter.identify_matched_filters(
            ToySourceFG, Feature("gf_toy_val"), toy_dac(h1={"gf_toy_val": [10, 20]})
        )

        assert len(matched) == 1
        filter_feature = next(iter(matched)).filter_feature
        pair = filter_feature.input_data_match
        assert pair is not None
        assert pair[0] is ToySourceFG
        assert pair[1].source == "h1:toy"
        assert RESERVED_READER_OPTION_KEY not in filter_feature.options.group
        assert RESERVED_READER_OPTION_KEY not in filter_feature.options.context

    def test_filter_on_a_column_absent_from_every_source_is_not_matched(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
            ToySourceFG,
            toy_dac,
        )

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_missing", "equal", {"value": 1})

        matched = global_filter.identify_matched_filters(
            ToySourceFG, Feature("gf_toy_val"), toy_dac(h1={"gf_toy_val": [10]})
        )

        assert len(matched) == 0

    def test_class_name_key_carried_over_with_a_missing_filter_column_drops_the_filter(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_missing", "equal", {"value": 1})
        host = Feature("gf_toy_val", Options({"ToySourceFG": {"gf_toy_val": [10]}}))

        matched = global_filter.identify_matched_filters(ToySourceFG, host, None)

        assert len(matched) == 0
        assert [e.stage for (fg, _, _), e in global_filter.dropped_filters.items() if fg is ToySourceFG] == [
            "input_data"
        ]

    def test_filter_column_in_two_sources_drops_the_filter_instead_of_raising(self) -> None:
        from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
            ToySourceFG,
            toy_dac,
        )

        global_filter = GlobalFilter()
        global_filter.add_filter("gf_toy_val", "equal", {"value": 10})

        matched = global_filter.identify_matched_filters(
            ToySourceFG, Feature("gf_toy_val"), toy_dac(h1={"gf_toy_val": [10]}, h2={"gf_toy_val": [10]})
        )

        assert len(matched) == 0
        assert [e.stage for (fg, _, _), e in global_filter.dropped_filters.items() if fg is ToySourceFG] == [
            "input_data"
        ]


class TestGlobalFilterOnColumnInSeveralFiles:
    """A filter column held by several sources resolves against the host's source, else the filter is dropped with a warning."""

    @staticmethod
    def _write(tmp_path: Path, name: str, text: str) -> str:
        path = tmp_path / name
        path.write_text(text)
        return str(path)

    @staticmethod
    def _run(tmp_path: Path, feature: str) -> list[Any]:
        global_filter = GlobalFilter()
        global_filter.add_filter("sevfilt_id", "equal", {"value": 1})
        return list(
            mloda.run_all(
                [Feature(feature)],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=DataAccessCollection(folders={"gf_folder": str(tmp_path)}),
                global_filter=global_filter,
            )
        )

    def test_filter_resolves_against_the_host_source(self, tmp_path: Path) -> None:
        self._write(tmp_path, "gf_sev_a.csv", "sevfilt_id,sevfilt_amount\n1,10\n2,20\n3,30\n")
        self._write(tmp_path, "gf_sev_b.csv", "sevfilt_id,sevfilt_other\n1,5\n2,6\n")

        results = self._run(tmp_path, "sevfilt_amount")

        assert results[0].to_pydict() == {"sevfilt_amount": [10]}

    def test_filter_is_evaluated_on_the_host_file_not_on_another_file_holding_the_column(self, tmp_path: Path) -> None:
        """Both files hold the filter column, with disjoint ids: only a.csv has id 1, so [10] proves a.csv was used."""
        self._write(tmp_path, "gf_disc_a.csv", "sevfilt_id,sevfilt_amount\n1,10\n2,20\n3,30\n")
        self._write(tmp_path, "gf_disc_b.csv", "sevfilt_id,sevfilt_other\n7,5\n8,6\n")

        results = self._run(tmp_path, "sevfilt_amount")

        assert results[0].to_pydict() == {"sevfilt_amount": [10]}

    def test_filter_column_only_in_other_sources_keeps_the_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._write(tmp_path, "gf_unres_a.csv", "sevfilt_id,sevfilt_other_a\n1,5\n2,6\n")
        self._write(tmp_path, "gf_unres_b.csv", "sevfilt_id,sevfilt_other_b\n1,7\n2,8\n")
        self._write(tmp_path, "gf_unres_c.csv", "sevfilt_amount\n10\n20\n")

        with caplog.at_level(logging.WARNING):
            results = self._run(tmp_path, "sevfilt_amount")

        assert results[0].to_pydict() == {"sevfilt_amount": [10, 20]}
        assert any(
            r.levelno == logging.WARNING and "matched no feature group" in r.getMessage() for r in caplog.records
        )

    def test_one_group_filters_on_the_host_source_while_another_group_still_filters(self, tmp_path: Path) -> None:
        self._write(tmp_path, "gf_mix_a.csv", "sevfilt_id,sevfilt_amount\n1,10\n2,20\n")
        self._write(tmp_path, "gf_mix_b.csv", "sevfilt_id,sevfilt_other\n1,5\n2,6\n")
        (tmp_path / "gf_mix_c.json").write_text(
            '{"sevfilt_id": 1, "sevfilt_val": 7}\n{"sevfilt_id": 2, "sevfilt_val": 8}\n'
        )
        global_filter = GlobalFilter()
        global_filter.add_filter("sevfilt_id", "equal", {"value": 1})

        results = mloda.run_all(
            [Feature("sevfilt_amount"), Feature("sevfilt_val")],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(folders={"gf_folder": str(tmp_path)}),
            global_filter=global_filter,
        )

        merged: dict[str, Any] = {}
        for result in results:
            merged.update(result.to_pydict())
        assert merged == {"sevfilt_amount": [10], "sevfilt_val": [7]}

    def test_filter_matching_no_group_for_another_reason_warns_and_stays_unfiltered(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._write(tmp_path, "gf_nomatch.csv", "sevfilt_nm_key,sevfilt_amount\n1,10\n2,20\n")

        with caplog.at_level(logging.WARNING):
            results = self._run(tmp_path, "sevfilt_amount")

        assert results[0].to_pydict() == {"sevfilt_amount": [10, 20]}
        assert any(
            r.levelno == logging.WARNING and "matched no feature group" in r.getMessage() for r in caplog.records
        )

    @staticmethod
    def _folder_with_two_files(tmp_path: Path) -> Path:
        folder = tmp_path / "sevfilt_data"
        folder.mkdir()
        (folder / "gf_ptr_a.csv").write_text("sevfilt_id,sevfilt_amount\n1,10\n2,20\n3,30\n")
        (folder / "gf_ptr_b.csv").write_text("sevfilt_id,sevfilt_other\n1,5\n2,6\n")
        return folder

    def test_host_pointed_at_a_folder_filters_the_host_file(self, tmp_path: Path) -> None:
        folder = self._folder_with_two_files(tmp_path)
        global_filter = GlobalFilter()
        global_filter.add_filter("sevfilt_id", "equal", {"value": 1})

        results = mloda.run_all(
            [Feature("sevfilt_amount", options={"CsvFG": str(folder)})],
            compute_frameworks=["PyArrowTable"],
            global_filter=global_filter,
        )

        assert results[0].to_pydict() == {"sevfilt_amount": [10]}

    @staticmethod
    def _run_with_pointed_filter(folder: Path) -> list[Any]:
        global_filter = GlobalFilter()
        global_filter.add_filter(Feature("sevfilt_id", options={"CsvFG": str(folder)}), "equal", {"value": 1})
        return list(
            mloda.run_all(
                [Feature("sevfilt_amount")],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=DataAccessCollection(folders={"gf_folder": str(folder)}),
                global_filter=global_filter,
            )
        )

    def test_filter_feature_pointed_at_a_folder_does_not_raise(self, tmp_path: Path) -> None:
        """The result equals the single-file counterpart, so the ambiguity changes nothing here."""
        two_files = self._folder_with_two_files(tmp_path)
        single = tmp_path / "sevfilt_single"
        single.mkdir()
        (single / "gf_ptr_a.csv").write_text((two_files / "gf_ptr_a.csv").read_text())

        results = self._run_with_pointed_filter(two_files)

        expected = self._run_with_pointed_filter(single)
        assert [r.to_pydict() for r in results] == [r.to_pydict() for r in expected]

    def test_database_column_in_two_tables_drops_the_filter_naming_the_ambiguity(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "sevfilt.db"
        write_sqlite(
            path,
            {
                "sevfilt_t1": {"sevfilt_dbpid": [1, 2], "sevfilt_dbamount": [10, 20]},
                "sevfilt_t2": {"sevfilt_dbpid": [1], "sevfilt_dbother": [5]},
            },
        )
        global_filter = GlobalFilter()
        global_filter.add_filter("sevfilt_dbpid", "equal", {"value": 1})

        with caplog.at_level(logging.WARNING):
            results = mloda.run_all(
                [Feature("sevfilt_dbamount")],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=DataAccessCollection(credentials={"sevfilt_h": {"sqlite": str(path)}}),
                global_filter=global_filter,
            )

        assert results[0].to_pydict() == {"sevfilt_dbamount": [10, 20]}
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        reasons = [e.reason for (fg, _, _), e in global_filter.dropped_filters.items() if fg is SqliteFG]
        assert len(reasons) == 1
        assert "several sources" in reasons[0]
        assert "credential mapping" not in reasons[0]
