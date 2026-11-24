"""Hand-set context splits fixed-name sources and follows their downstream consumers."""

import itertools
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from mloda.provider import (
    ComputeFramework,
    DataCreatorFeatureGroup,
    FeatureChainParser,
    FeatureGroup,
    FeatureSet,
    PropertySpec,
)
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

ROOT = "explicit_candidates"
ALTERNATE = "explicit_candidates_alternate"
CONSUMERS = ("explicit_small", "explicit_large")
OUTER = {"explicit_outer_small": (CONSUMERS[0], 1), "explicit_outer_large": (CONSUMERS[1], 2)}
FRAMEWORKS = [PythonDictFramework, PyArrowTable]


def _values(data: Any, name: str) -> list[int]:
    return list(data[name].to_pylist()) if isinstance(data, pa.Table) else list(data[name])


def _columns(results: Any) -> dict[str, list[list[int]]]:
    columns: dict[str, list[list[int]]] = {}
    for frame in results:
        data = frame.to_pydict() if isinstance(frame, pa.Table) else frame
        for name, values in data.items():
            columns.setdefault(name, []).append(list(values))
    return {name: sorted(values) for name, values in columns.items()}


class ExplicitContextRoot(DataCreatorFeatureGroup):
    PROPERTY_MAPPING = {
        "top_k": PropertySpec("The hand-set source parameter.", context=True),
        "padding": PropertySpec("A materialized context default.", context=True, default="filled"),
    }
    calls: ClassVar[list[tuple[int, ...]]] = []
    observed: ClassVar[list[Options]] = []

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {ROOT, ALTERNATE}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls.append(tuple(sorted(int(feature.options.get("top_k")) for feature in features.features)))
        cls.observed.extend(feature.options for feature in features.features)
        return {str(feature.name): [int(feature.options.get("top_k")) * 10] for feature in features.features}


class RenamingContextConsumer(FeatureGroup):
    PROPERTY_MAPPING = {"candidates_top_k": PropertySpec("Renamed when handed to the source.", context=True)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(CONSUMERS)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        source_options = Options(context={"top_k": options.get("candidates_top_k")})
        return {Feature(ROOT, source_options, split_context_keys={"top_k"})}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): [value + 1 for value in _values(data, ROOT)] for feature in features.features}


class SecondLevelContextConsumer(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(OUTER)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        source, top_k = OUTER[str(feature_name)]
        return {Feature(source, Options(context={"candidates_top_k": top_k}))}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            str(feature.name): [value + 1 for value in _values(data, OUTER[str(feature.name)][0])]
            for feature in features.features
        }


NAMINGS = [(True, "no_patterns"), (False, "no_patterns"), (False, "unreadable_name"), (False, "patterns_raise")]
NAMING_IDS = ["marked", "unmarked", "unmarked-unreadable_name", "unmarked-patterns_raise"]


def _raise_patterns(cls: type) -> Any:
    raise RuntimeError("unreadable patterns")


def _apply_naming(naming: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the root's name patterns unreadable as the naming requires."""
    if naming == "unreadable_name":
        monkeypatch.setattr(ExplicitContextRoot, "PREFIX_PATTERN", ROOT, raising=False)
        # ROOT has no "__" source, so the pattern matches but resolve_name raises.
        with pytest.raises(ValueError):
            FeatureChainParser.resolve_name(
                ROOT, ExplicitContextRoot._get_prefix_patterns(), ExplicitContextRoot.PROPERTY_MAPPING
            )
    elif naming == "patterns_raise":
        monkeypatch.setattr(ExplicitContextRoot, "_get_prefix_patterns", classmethod(_raise_patterns))


def _groups(
    calls: list[tuple[int, ...]], observed: list[Options]
) -> tuple[type[FeatureGroup], type[FeatureGroup], type[FeatureGroup]]:
    ExplicitContextRoot.calls = calls
    ExplicitContextRoot.observed = observed

    return ExplicitContextRoot, RenamingContextConsumer, SecondLevelContextConsumer


def _run(
    requests: list[Feature | str], groups: set[type[FeatureGroup]], frameworks: list[type[ComputeFramework]]
) -> Any:
    return mloda.run_all(
        requests,
        compute_frameworks=frameworks,
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
        parallelization_modes={ParallelizationMode.SYNC},
    )


def _assert_explicit_source_marker(observed: list[Options], top_k: int, marked: bool = True) -> None:
    matching = [options for options in observed if options.get("top_k") == top_k]
    assert matching
    if marked:
        assert all("top_k" in options.split_context_keys for options in matching)
    assert all(options.inherited_context_keys == frozenset() for options in observed)
    assert all(options.get("padding") == "filled" for options in observed)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("reverse", [False, True], ids=["direct-first", "consumer-first"])
@pytest.mark.parametrize("direct_top_k", [2, 3], ids=["equal-values", "different-values"])
@pytest.mark.parametrize("marked, naming", NAMINGS, ids=NAMING_IDS)
def test_renamed_input_context_shares_equal_values_and_splits_different_values(
    framework: type[ComputeFramework],
    reverse: bool,
    direct_top_k: int,
    marked: bool,
    naming: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_naming(naming, monkeypatch)
    if not marked:

        def input_features(self: RenamingContextConsumer, options: Options, feature_name: FeatureName) -> set[Feature]:
            return {Feature(ROOT, Options(context={"top_k": options.get("candidates_top_k")}))}

        monkeypatch.setattr(RenamingContextConsumer, "input_features", input_features)
    calls: list[tuple[int, ...]] = []
    observed: list[Options] = []
    root, consumer, _ = _groups(calls, observed)
    requests: list[Feature | str] = [
        Feature(ROOT, Options(context={"top_k": direct_top_k})),
        Feature(CONSUMERS[0], Options(context={"candidates_top_k": 2})),
    ]
    if reverse:
        requests.reverse()

    results = _run(requests, {root, consumer}, [framework])

    assert _columns(results) == {ROOT: [[direct_top_k * 10]], CONSUMERS[0]: [[21]]}
    assert sorted(calls) == ([(2,)] if direct_top_k == 2 else [(2,), (3,)])
    _assert_explicit_source_marker(observed, 2, marked)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("marked, naming", NAMINGS, ids=NAMING_IDS)
def test_direct_requests_differing_in_a_declared_key_are_rejected_unless_marked(
    framework: type[ComputeFramework], marked: bool, naming: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_naming(naming, monkeypatch)
    calls: list[tuple[int, ...]] = []
    root, _, _ = _groups(calls, [])
    keys = {"top_k"} if marked else set()
    requests: list[Feature | str] = [
        Feature(ROOT, Options(context={"top_k": k}), split_context_keys=keys) for k in (2, 3)
    ]

    if not marked:
        with pytest.raises(ValueError, match="is reached twice with differing values for context key"):
            _run(requests, {root}, [framework])
        return

    results = _run(requests, {root}, [framework])

    assert sorted(calls) == [(2,), (3,)]
    assert sorted(v for column in _columns(results)[ROOT] for v in column) == [20, 30]


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("merge", ["intake", "narrower-fold"])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_equal_marked_and_plain_requests_keep_the_marker_when_they_merge(
    framework: type[ComputeFramework], merge: str, order: tuple[int, ...]
) -> None:
    calls: list[tuple[int, ...]] = []
    observed: list[Options] = []
    root, consumer, _ = _groups(calls, observed)
    pin = framework.get_class_name() if merge == "narrower-fold" else None
    requests: list[Feature | str] = [
        Feature(ROOT, Options(context={"top_k": 2}), compute_framework=pin),
        Feature(CONSUMERS[0], Options(context={"candidates_top_k": 2}), compute_framework=pin),
        Feature(ROOT, Options(context={"top_k": 3})),
    ]
    frameworks = [framework]
    if merge == "narrower-fold":
        frameworks.extend(candidate for candidate in FRAMEWORKS if candidate is not framework)

    results = _run([requests[index] for index in order], {root, consumer}, frameworks)

    assert _columns(results) == {ROOT: [[20], [30]], CONSUMERS[0]: [[21]]}
    assert sorted(calls) == [(2,), (3,)]
    _assert_explicit_source_marker(observed, 2)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("reverse", [False, True])
def test_second_level_consumers_follow_explicitly_split_root_variants(
    framework: type[ComputeFramework], reverse: bool
) -> None:
    calls: list[tuple[int, ...]] = []
    observed: list[Options] = []
    groups = _groups(calls, observed)
    requests: list[Feature | str] = list(OUTER)
    if reverse:
        requests.reverse()

    results = _run(requests, set(groups), [framework])

    assert _columns(results) == {"explicit_outer_small": [[12]], "explicit_outer_large": [[22]]}
    assert sorted(calls) == [(1,), (2,)]
    _assert_explicit_source_marker(observed, 1)
    _assert_explicit_source_marker(observed, 2)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_context_metadata_without_a_split_marker_keeps_one_call(framework: type[ComputeFramework]) -> None:
    calls: list[tuple[int, ...]] = []
    observed: list[Options] = []
    root, _, _ = _groups(calls, observed)
    requests = [
        Feature(ROOT, Options(context={"top_k": 2, "metadata": "one"})),
        Feature(ALTERNATE, Options(context={"top_k": 2, "metadata": "two"})),
    ]

    results = _run(list(requests), {root}, [framework])

    assert _columns(results) == {ROOT: [[20]], ALTERNATE: [[20]]}
    assert calls == [(2, 2)]
    assert all(options.split_context_keys == frozenset() for options in observed)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("marker_location", ["chain-root", "third-direct-root"])
def test_one_marker_in_the_root_scope_keeps_both_plain_downstream_paths_separate(
    framework: type[ComputeFramework], reverse: bool, marker_location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def input_features(self: RenamingContextConsumer, options: Options, feature_name: FeatureName) -> set[Feature]:
        value = options.get("candidates_top_k")
        keys = {"top_k"} if marker_location == "chain-root" and value == 1 else frozenset[str]()
        return {Feature(ROOT, Options(context={"top_k": value}), split_context_keys=keys)}

    monkeypatch.setattr(RenamingContextConsumer, "input_features", input_features)
    calls: list[tuple[int, ...]] = []
    observed: list[Options] = []
    groups = _groups(calls, observed)
    requests: list[Feature | str] = list(OUTER)
    expected = {"explicit_outer_small": [[12]], "explicit_outer_large": [[22]]}
    expected_calls = [(1,), (2,)]
    marked_value = 1
    if marker_location == "third-direct-root":
        requests.insert(0, Feature(ROOT, Options(context={"top_k": 3}), split_context_keys={"top_k"}))
        expected[ROOT] = [[30]]
        expected_calls.append((3,))
        marked_value = 3
    if reverse:
        requests.reverse()

    results = _run(requests, set(groups), [framework])

    assert _columns(results) == expected
    assert sorted(calls) == expected_calls
    _assert_explicit_source_marker(observed, marked_value)


FORWARDED = ("forwarded_small", "forwarded_large")


class ForwardingConsumer(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(FORWARDED)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(ROOT)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): [value + 1 for value in _values(data, ROOT)] for feature in features.features}


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("reverse", [False, True])
def test_forwarded_context_key_splits_a_fixed_name_root_without_a_marker(
    framework: type[ComputeFramework], reverse: bool
) -> None:
    calls: list[tuple[int, ...]] = []
    observed: list[Options] = []
    root, _, _ = _groups(calls, observed)
    requests: list[Feature | str] = [
        Feature(name, Options(context={"top_k": top_k}, propagate_context_keys=frozenset({"top_k"})))
        for name, top_k in zip(FORWARDED, (1, 2))
    ]
    if reverse:
        requests.reverse()

    results = _run(requests, {root, ForwardingConsumer}, [framework])

    assert _columns(results) == {"forwarded_small": [[11]], "forwarded_large": [[21]]}
    assert sorted(calls) == [(1,), (2,)]


MID = "marked_mid"
BASE = "marked_base"
TWIN_CONSUMERS = {"marked_twin_one": 1, "marked_twin_two": 2}
MARKING_TWINS: set[str] = set(TWIN_CONSUMERS)


class MarkedBaseRoot(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {BASE}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {BASE: [1]}


class MarkedMid(FeatureGroup):
    PROPERTY_MAPPING = {"level": PropertySpec("The marked value.", context=True)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {MID}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(BASE)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {MID: [1]}


class MarkedTwinConsumer(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(TWIN_CONSUMERS)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        level = TWIN_CONSUMERS[str(feature_name)]
        marked: set[str] = {"level"} if str(feature_name) in MARKING_TWINS else set()
        return {Feature(MID, Options(context={"level": level}), split_context_keys=marked)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {str(feature.name): [1] for feature in features.features}


@pytest.mark.parametrize(
    "marking",
    [{"marked_twin_one", "marked_twin_two"}, {"marked_twin_one"}, {"marked_twin_two"}, set()],
    ids=["both", "only_first", "only_second", "none"],
)
def test_non_root_input_reached_with_differing_values_is_rejected_at_plan_time(
    marking: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{__name__}.MARKING_TWINS", marking)
    with pytest.raises(ValueError, match="is reached twice with differing values for context key"):
        _run(list(TWIN_CONSUMERS), {MarkedBaseRoot, MarkedMid, MarkedTwinConsumer}, [PythonDictFramework])
