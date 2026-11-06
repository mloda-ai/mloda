"""Run-all driven policy suites: pipeline result policies exercised end-to-end through ``mloda.run_all``.

A new pipeline result policy ships a suite on ``RunAllSuite`` that covers every framework in SYNC and a worker mode,
each case a ``PolicySuccess`` or a ``PolicyRaises``. Hosts only set ``adapter_class``.
Driving the policy through the whole pipeline is mandatory: unit tests of single enforcement surfaces miss defects.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.compute_framework.merge_suite import ASOF_SCENARIOS
from mloda._testing.compute_framework_shape import serves_asof_merge
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.source import reclaim
from mloda.core.abstract_plugins.compute_framework import ComputeFramework, EmptyResultError
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.optional_dependency import require
from mloda.provider import DataCreatorFeatureGroup, FeatureChainParserMixin, FeatureGroup, FeatureSet, property_spec
from mloda.user import (
    DataAccessCollection,
    Feature,
    GlobalFilter,
    Index,
    JoinSpec,
    Link,
    MissingInputFeatureError,
    Options,
    ParallelizationMode,
    PluginCollector,
    RunResult,
    mloda,
)


@dataclass(frozen=True)
class PolicySuccess:
    """run_all returns results; ``assert_result`` is called on them."""

    assert_result: Callable[[RunResult], None]


@dataclass(frozen=True)
class PolicyRaises:
    """run_all raises an Exception whose ``str()`` contains ``match_substring`` (and ``also_contains``)."""

    match_substring: str
    error_type: type[BaseException] | None = None
    cause_type: type[BaseException] | None = None
    also_contains: tuple[str, ...] = ()


PolicyExpectation = PolicySuccess | PolicyRaises

_WORKER_MODES: set[ParallelizationMode] = {ParallelizationMode.SYNC, ParallelizationMode.THREADING}


class RunAllSuite(ComputeFrameworkSuite, suite_base=True):
    """Drives a FeatureGroup-level policy end-to-end through ``run_all`` on the host's framework."""

    aspect = "run all"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    @pytest.fixture(autouse=True)
    # The "_a_" prefix sorts it first among autouse fixtures, so its mark is the outermost.
    def _a_reclaim_suite_built_classes(self) -> Iterator[None]:
        yield from reclaim.reclaimed_scope()

    @pytest.fixture(scope="session")
    def flight_server(self) -> Iterator[Any]:
        if importlib.util.find_spec("pyarrow") is None:
            yield None
            return
        from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer

        server = ParallelRunnerFlightServer()
        yield server
        server.end_flight_server_process()

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        if "mode" not in metafunc.fixturenames:
            return
        adapter_class = getattr(metafunc.cls, "adapter_class", None)
        modes: set[ParallelizationMode] = set(_WORKER_MODES)
        if adapter_class is not None:
            supported = adapter_class().compute_framework().supported_parallelization_modes()
            modes = {mode for mode in _WORKER_MODES if mode in supported}
        metafunc.parametrize("mode", sorted(modes, key=str))

    def run_all_connection(self, adapter: FrameworkAdapter) -> Any | None:
        """The connection travelling in the DataAccessCollection; None runs without one."""
        return adapter.connection

    def assert_policy_case(
        self,
        adapter: FrameworkAdapter,
        *,
        feature_name: str | Sequence[str],
        plugin_collector: PluginCollector,
        expectation: PolicyExpectation,
        mode: ParallelizationMode,
        flight_server: Any,
        global_filter: GlobalFilter | None = None,
        links: set[Link] | None = None,
    ) -> RunResult | BaseException:
        names = [feature_name] if isinstance(feature_name, str) else list(feature_name)
        features = [Feature(name=name) for name in names]
        connection = self.run_all_connection(adapter)
        dac = None if connection is None else DataAccessCollection(connections={connection})
        framework = self.served_framework(adapter)

        def run() -> RunResult:
            return mloda.run_all(
                features,
                compute_frameworks=[framework],
                plugin_collector=plugin_collector,
                parallelization_modes={mode},
                flight_server=flight_server,
                data_access_collection=dac,
                global_filter=global_filter,
                links=links,
            )

        if isinstance(expectation, PolicyRaises):
            with pytest.raises(Exception) as excinfo:
                run()
            assert expectation.match_substring in str(excinfo.value)
            for substring in expectation.also_contains:
                assert substring in str(excinfo.value)
            if expectation.error_type is not None:
                assert isinstance(excinfo.value, expectation.error_type)
            if expectation.cause_type is not None:
                assert isinstance(excinfo.value.__cause__, expectation.cause_type)
            return excinfo.value

        result = run()
        expectation.assert_result(result)
        return result


def _root_class(
    names: set[str],
    rows: Any,
    framework: type[ComputeFramework] | None = None,
    name: str | None = None,
) -> type[FeatureGroup]:
    """A tracked root FeatureGroup supporting ``names`` that returns ``rows``, optionally pinned to ``framework``."""

    class _RootFeatureGroup(DataCreatorFeatureGroup):
        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return set(names)

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            if isinstance(rows, dict):
                return {key: list(values) for key, values in rows.items()}
            return rows

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return None if framework is None else {framework}

    if name is not None:
        _RootFeatureGroup.__name__ = name
    return reclaim.track(_RootFeatureGroup)


def _root_collector(feature_name: str, rows: dict[str, list[Any]]) -> PluginCollector:
    """A collector enabling a tracked root FeatureGroup that returns ``rows`` for ``feature_name``."""
    return PluginCollector.enabled_feature_groups({_root_class({feature_name}, rows)})


class EmptyResultRunAllSuite(RunAllSuite, suite_base=True):
    """A schema-bearing zero-row result succeeds; a zero-column result raises EmptyResultError."""

    aspect = "empty result run all"

    def _assert_single_empty_result(self, adapter: FrameworkAdapter) -> PolicySuccess:
        def check(result: RunResult) -> None:
            assert len(result) == 1
            assert self._result_row_count(adapter, result[0]) == 0
            for step, frame in result.frames():
                assert step.result_framework is not None
                assert isinstance(frame, step.result_framework.expected_data_framework())

        return PolicySuccess(assert_result=check)

    def test_empty_result_default(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A zero-row result with one empty column succeeds on every framework."""
        self.assert_policy_case(
            adapter,
            feature_name="empty_result_default_col",
            plugin_collector=_root_collector("empty_result_default_col", {"empty_result_default_col": []}),
            expectation=self._assert_single_empty_result(adapter),
            mode=mode,
            flight_server=flight_server,
        )

    def test_empty_result_allowed_succeeds(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A FeatureGroup returning a schema-bearing zero-row result succeeds on every backend."""
        self.assert_policy_case(
            adapter,
            feature_name="empty_result_allowed_col",
            plugin_collector=_root_collector("empty_result_allowed_col", {"empty_result_allowed_col": []}),
            expectation=self._assert_single_empty_result(adapter),
            mode=mode,
            flight_server=flight_server,
        )

    def zero_column_expectation(self) -> PolicyRaises:
        """Override where the backend rejects a zero-column result before the schema guard."""
        return PolicyRaises(match_substring="Result carries no schema", error_type=EmptyResultError)

    def test_zero_column_result_raises(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A zero-column ``{}`` result is schema-less and raises ``zero_column_expectation``."""
        self.assert_policy_case(
            adapter,
            feature_name="empty_result_schemaless_col",
            plugin_collector=_root_collector("empty_result_schemaless_col", {}),
            expectation=self.zero_column_expectation(),
            mode=mode,
            flight_server=flight_server,
        )


SD_A_VALUES = [1, 2, 3]
_PARTIAL_MESSAGE = "PartialResultProbeFirstOnly did not produce the feature(s) ['sd_a__partial_result_probe_first_2']"


def _group(name: str, bases: tuple[type, ...], members: dict[str, Any]) -> type[FeatureGroup]:
    """A tracked FeatureGroup subclass named ``name``."""
    return reclaim.track(type(name, bases, members))


def _inputs_consumer(name: str, inputs: tuple[str, ...]) -> type[FeatureGroup]:
    def input_features(self: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(n) for n in inputs}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 1, 1]}

    return _group(
        name, (FeatureGroup,), {"input_features": input_features, "calculate_feature": classmethod(calculate_feature)}
    )


class SiblingDictRunAllSuite(RunAllSuite, suite_base=True):
    """Dict-returning sibling consumers end-to-end through run_all."""

    aspect = "sibling dict run all"

    def _column(self, adapter: FrameworkAdapter, data: Any, name: str) -> list[Any]:
        return [row[name] for row in self.rows(adapter, data)]

    def _sibling_groups(self, adapter: FrameworkAdapter) -> dict[str, type[FeatureGroup]]:
        column = self._column

        def input_features(self_: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature("sd_a")}

        def scaled(name: str, factor: int) -> type[FeatureGroup]:
            def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
                return {name: [v * factor for v in column(adapter, data, "sd_a")]}

            return _group(
                name,
                (FeatureGroup,),
                {"input_features": input_features, "calculate_feature": classmethod(calculate_feature)},
            )

        def overlap(cls: Any, data: Any, features: FeatureSet) -> Any:
            values = column(adapter, data, "sd_a")
            return {"sd_a": values, "SiblingDictOverlap": [v + 10 for v in values]}

        def aggregate(cls: Any, data: Any, features: FeatureSet) -> Any:
            return {"SiblingDictAggregate": [sum(column(adapter, data, "sd_a"))]}

        def case_fold(cls: Any, data: Any, features: FeatureSet) -> Any:
            return {"SD_A": [v * 2 for v in column(adapter, data, "sd_a")]}

        def case_fold_names(cls: Any) -> set[str]:
            return {"SD_A"}

        groups = {
            "SiblingDictDouble": scaled("SiblingDictDouble", 2),
            "SiblingDictTriple": scaled("SiblingDictTriple", 3),
        }
        for name, fn in (
            ("SiblingDictOverlap", overlap),
            ("SiblingDictAggregate", aggregate),
            ("SiblingDictCaseFold", case_fold),
        ):
            members: dict[str, Any] = {"input_features": input_features, "calculate_feature": classmethod(fn)}
            if name == "SiblingDictCaseFold":
                members["feature_names_supported"] = classmethod(case_fold_names)
            groups[name] = _group(name, (FeatureGroup,), members)
        return groups

    def _sd_root(self, adapter: FrameworkAdapter) -> type[FeatureGroup]:
        return _root_class({"sd_a"}, {"sd_a": SD_A_VALUES})

    def _assert_column_values(self, adapter: FrameworkAdapter, expected: dict[str, list[Any]]) -> PolicySuccess:
        def check(result: RunResult) -> None:
            columns: dict[str, list[Any]] = {}
            for frame in result:
                for row in self._result_rows(adapter, frame):
                    for key, value in row.items():
                        columns.setdefault(key, []).append(value)
            for name, values in expected.items():
                assert name in columns, f"missing {name}, got {sorted(columns)}"
                assert columns[name] == values

        return PolicySuccess(assert_result=check)

    def _run_sibling(
        self,
        adapter: FrameworkAdapter,
        names: list[str],
        expected: dict[str, list[Any]],
        mode: ParallelizationMode,
        flight_server: Any,
        global_filter: GlobalFilter | None = None,
    ) -> None:
        groups = self._sibling_groups(adapter)
        enabled = {self._sd_root(adapter), *groups.values()}
        self.assert_policy_case(
            adapter,
            feature_name=names,
            plugin_collector=PluginCollector.enabled_feature_groups(enabled),
            expectation=self._assert_column_values(adapter, expected),
            mode=mode,
            flight_server=flight_server,
            global_filter=global_filter,
        )

    def _partial_first_only(self) -> type[FeatureGroup]:
        def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
            return {features.get_sorted_features()[0].name: [0, 0, 0]}

        return _group(
            "PartialResultProbeFirstOnly",
            (FeatureChainParserMixin, FeatureGroup),
            {
                "PREFIX_PATTERN": r".*__partial_result_probe_first_(?P<n>\d+)$",
                "MIN_IN_FEATURES": 1,
                "MAX_IN_FEATURES": 1,
                "PROPERTY_MAPPING": {
                    "n": property_spec("n", name_converter=int),
                    "in_features": property_spec("src"),
                },
                "calculate_feature": classmethod(calculate_feature),
            },
        )

    def test_consumer_whose_declared_input_was_replaced_raises(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A consumer whose declared input was replaced by a one-row dict raises."""
        groups = self._sibling_groups(adapter)
        lost = _inputs_consumer("SiblingDictLostInputConsumer", ("sd_a", "SiblingDictAggregate"))
        self.assert_policy_case(
            adapter,
            feature_name=["SiblingDictLostInputConsumer"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {self._sd_root(adapter), groups["SiblingDictAggregate"], lost}
            ),
            expectation=PolicyRaises(
                match_substring="SiblingDictLostInputConsumer",
                error_type=MissingInputFeatureError,
                also_contains=("declared input(s) ['sd_a'] missing", "dropped by feature group 'SiblingDictAggregate'"),
            ),
            mode=mode,
            flight_server=flight_server,
        )

    def test_intermediate_group_missing_a_feature_raises(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """An intermediate group omitting a requested feature raises."""
        consumer = _inputs_consumer(
            "PartialResultProbeConsumer",
            ("sd_a__partial_result_probe_first_1", "sd_a__partial_result_probe_first_2"),
        )
        self.assert_policy_case(
            adapter,
            feature_name=["PartialResultProbeConsumer"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {self._sd_root(adapter), self._partial_first_only(), consumer}
            ),
            expectation=PolicyRaises(match_substring=_PARTIAL_MESSAGE),
            mode=mode,
            flight_server=flight_server,
        )

    def test_source_group_missing_a_feature_raises(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A source group omitting a requested feature raises."""
        creator = _root_class(
            {"sd_src_kept", "sd_src_omitted"}, {"sd_src_kept": [1, 2, 3]}, name="SourceOmittingCreator"
        )
        consumer = _inputs_consumer("SourceOmittingConsumer", ("sd_src_kept", "sd_src_omitted"))
        self.assert_policy_case(
            adapter,
            feature_name=["SourceOmittingConsumer"],
            plugin_collector=PluginCollector.enabled_feature_groups({creator, consumer}),
            expectation=PolicyRaises(
                match_substring="SourceOmittingCreator did not produce the feature(s) ['sd_src_omitted']"
            ),
            mode=mode,
            flight_server=flight_server,
        )

    def test_group_missing_a_requested_feature_raises(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A group omitting one of two requested features raises."""
        self.assert_policy_case(
            adapter,
            feature_name=["sd_a__partial_result_probe_first_1", "sd_a__partial_result_probe_first_2"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {self._sd_root(adapter), self._partial_first_only()}
            ),
            expectation=PolicyRaises(match_substring=_PARTIAL_MESSAGE),
            mode=mode,
            flight_server=flight_server,
        )

    def test_case_differing_dict_key_succeeds(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A dict key differing from the root column only in case succeeds."""
        self._run_sibling(adapter, ["SD_A"], {"SD_A": [2, 4, 6]}, mode, flight_server)

    def test_sibling_dict_consumers_return_both_outputs(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """Two sibling dict consumers both return their output column."""
        expected = {"SiblingDictDouble": [2, 4, 6], "SiblingDictTriple": [3, 6, 9]}
        self._run_sibling(adapter, ["SiblingDictDouble", "SiblingDictTriple"], expected, mode, flight_server)

    def test_overlapping_dict_still_replaces_frame(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A dict repeating the existing column plus its output replaces the frame."""
        self._run_sibling(adapter, ["SiblingDictOverlap"], {"SiblingDictOverlap": [11, 12, 13]}, mode, flight_server)

    def test_different_row_count_dict_still_replaces_frame(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A one-row aggregate dict replaces the three-row frame."""
        self._run_sibling(adapter, ["SiblingDictAggregate"], {"SiblingDictAggregate": [6]}, mode, flight_server)

    def test_sibling_dict_consumers_return_both_outputs_under_final_filter(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """Sibling outputs are filtered by a final global filter."""
        global_filter = GlobalFilter()
        global_filter.add_filter("sd_a", "min", {"value": 2})
        expected = {"SiblingDictDouble": [4, 6], "SiblingDictTriple": [6, 9]}
        self._run_sibling(
            adapter, ["SiblingDictDouble", "SiblingDictTriple"], expected, mode, flight_server, global_filter
        )


UNDECLARED_COLUMN = "mc_undeclared_col"
UndeclaredRead = Callable[[Any], Any]
CauseType = Callable[[], type[BaseException]]


def key_read(data: Any) -> Any:
    """Subscripts the undeclared column (dict or pandas frame)."""
    return data[UNDECLARED_COLUMN]


def select_read(data: Any) -> Any:
    """Selects the undeclared column on a SQL relation."""
    return data.select(UNDECLARED_COLUMN)


def pyarrow_key_read(data: Any) -> Any:
    """Reads the undeclared column from a PyArrow table by name."""
    return data.column(UNDECLARED_COLUMN)


def pyarrow_filter_read(data: Any) -> Any:
    """Filters a PyArrow table on the undeclared column."""
    pc = require("pyarrow.compute", "filtering PyArrow test tables")
    return data.filter(pc.field(UNDECLARED_COLUMN) > 0)


def attribute_read(data: Any) -> Any:
    """Reads the undeclared column as an attribute (``df.x``)."""
    return getattr(data, UNDECLARED_COLUMN)


def _misspelled_read(data: Any) -> Any:
    return data.fliter


def _consumer_class(name: str, read: UndeclaredRead) -> type[FeatureGroup]:
    def input_features(self: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mc_declared")}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        return read(data)

    return _group(
        name, (FeatureGroup,), {"input_features": input_features, "calculate_feature": classmethod(calculate_feature)}
    )


class MissingColumnRunAllSuite(RunAllSuite, suite_base=True):
    """Each undeclared read case raises an explanatory ``ValueError`` through ``run_all``."""

    aspect = "missing column run all"
    required_hooks = ("undeclared_reads",)

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        """Case name -> (read function, lazy cause type); import no optional backend here, only in the callables."""
        raise NotImplementedError

    def root_values(self, adapter: FrameworkAdapter) -> Any:
        return {"mc_declared": [1, 2, 3]}

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        if "case" in metafunc.fixturenames:
            cases = list(metafunc.cls.undeclared_reads()) if metafunc.cls is not None else []
            metafunc.parametrize("case", cases, ids=cases)
        if "mode" in metafunc.fixturenames:
            metafunc.parametrize("mode", [ParallelizationMode.SYNC])

    def _root(self, adapter: FrameworkAdapter) -> type[FeatureGroup]:
        return _root_class({"mc_declared"}, self.root_values(adapter), adapter.compute_framework())

    def test_undeclared_column_read_raises_explanatory_error(
        self, adapter: FrameworkAdapter, case: str, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """An undeclared column read raises an error naming the consumer and its input_features."""
        self.assert_undeclared_read_raises(adapter, case, mode, flight_server)

    def test_misspelled_method_is_not_translated(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """A misspelled method stays an AttributeError and is not blamed on input_features."""
        consumer = _consumer_class("MisspelledMethodConsumer", _misspelled_read)
        error = self.assert_policy_case(
            adapter,
            feature_name="MisspelledMethodConsumer",
            plugin_collector=PluginCollector.enabled_feature_groups({self._root(adapter), consumer}),
            expectation=PolicyRaises(match_substring="fliter", error_type=AttributeError),
            mode=mode,
            flight_server=flight_server,
        )
        assert "input_features" not in str(error)

    def assert_undeclared_read_raises(
        self,
        adapter: FrameworkAdapter,
        case: str,
        mode: ParallelizationMode,
        flight_server: Any,
        global_filter: GlobalFilter | None = None,
    ) -> BaseException:
        read, cause_factory = self.undeclared_reads()[case]
        cause_type = cause_factory()
        consumer = _consumer_class(case, read)
        error = self.assert_policy_case(
            adapter,
            feature_name=case,
            plugin_collector=PluginCollector.enabled_feature_groups({self._root(adapter), consumer}),
            expectation=PolicyRaises(
                match_substring=case,
                error_type=ValueError,
                cause_type=cause_type,
                also_contains=(cause_type.__name__, UNDECLARED_COLUMN, "input_features"),
            ),
            mode=mode,
            flight_server=flight_server,
            global_filter=global_filter,
        )
        assert isinstance(error, BaseException)
        return error


class AsofRunAllSuite(RunAllSuite, suite_base=True):
    """A backward, single-by-key as-of join end-to-end through run_all."""

    aspect = "asof run all"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return serves_asof_merge(framework)

    def test_backward_single_key(
        self, adapter: FrameworkAdapter, mode: ParallelizationMode, flight_server: Any
    ) -> None:
        """The joined group sees each left row matched to the latest earlier right row."""
        scenario = ASOF_SCENARIOS["backward_single_key"]
        left = _root_class({"k", "t", "lv"}, scenario["left"].to_python_dict(), name="AsofLeftFeature")
        right = _root_class({"k", "t", "rv"}, scenario["right"].to_python_dict(), name="AsofRightFeature")
        link = Link.asof(
            JoinSpec(left, Index(("k",))),
            JoinSpec(right, Index(("k",))),
            left_time_column="t",
            right_time_column="t",
            direction="backward",
        )

        def input_features(self_: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return {Feature(name="lv", link=link, index=Index(("k",))), Feature(name="rv", index=Index(("k",)))}

        def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
            rows = self.rows(adapter, data)
            return {cls.get_class_name(): [f"{r['k']}|{r['t']}|{r['rv']}" for r in rows]}

        joined = _group(
            "AsofJoinedFeature",
            (FeatureGroup,),
            {"input_features": input_features, "calculate_feature": classmethod(calculate_feature)},
        )

        def check(result: RunResult) -> None:
            assert len(result) == 1
            rows = self._result_rows(adapter, result[0])
            assert sorted(str(r["AsofJoinedFeature"]) for r in rows) == ["1|10|1", "1|20|2", "2|15|3"]

        self.assert_policy_case(
            adapter,
            feature_name="AsofJoinedFeature",
            plugin_collector=PluginCollector.enabled_feature_groups({left, right, joined}),
            expectation=PolicySuccess(assert_result=check),
            mode=mode,
            flight_server=flight_server,
            links={link},
        )
