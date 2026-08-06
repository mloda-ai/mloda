"""One compute framework wins every reduction, whatever the set iteration order.
Set iteration over class objects is id-based, so the reduction pins to the lowest class name.
"""

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import ResolveLinks
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

_PROBE = Path(__file__).with_name("determinism_probe.py")
_PROBE_TIMEOUT = 30.0
# Each probe is a fresh interpreter importing PyArrowTable, so the count is what the gate budget allows.
_PROBE_PROCESSES = 5
_PROBE_EXPECTED = {"feature": "PyArrowTable", "trekker_left": "PyArrowTable", "trekker_right": "PyArrowTable"}


class DeterminismLeftFeatureGroup(FeatureGroup):
    pass


class DeterminismRightFeatureGroup(FeatureGroup):
    pass


def _throwaway_frameworks() -> tuple[type[ComputeFramework], ...]:
    """Defined per call and unavailable, so discovery never sees them; Zz sorts behind every shipped name."""

    class ZzZuluThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    class ZzAlfaThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    class ZzTangoThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    class ZzBravoThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

    # Definition order deliberately disagrees with name order.
    return (ZzZuluThrowawayFramework, ZzAlfaThrowawayFramework, ZzTangoThrowawayFramework, ZzBravoThrowawayFramework)


def _same_name_frameworks() -> tuple[type[ComputeFramework], type[ComputeFramework]]:
    """Two frameworks sharing a class name, separated only by qualname."""

    def alfa() -> type[ComputeFramework]:
        class ZzSharedNameThrowawayFramework(ComputeFramework):
            @staticmethod
            def is_available() -> bool:
                return False

        return ZzSharedNameThrowawayFramework

    def bravo() -> type[ComputeFramework]:
        class ZzSharedNameThrowawayFramework(ComputeFramework):
            @staticmethod
            def is_available() -> bool:
                return False

        return ZzSharedNameThrowawayFramework

    return alfa(), bravo()


def _link() -> Link:
    return Link.inner(
        JoinSpec(DeterminismLeftFeatureGroup, "idx"),
        JoinSpec(DeterminismRightFeatureGroup, "idx"),
    )


def _run_probes(count: int) -> list[dict[str, str]]:
    assert _PROBE.is_file(), f"{_PROBE} does not exist"

    outputs: list[dict[str, str]] = []
    for _ in range(count):
        # Safe: fixed argv, no shell, no user input.
        completed = subprocess.run(  # nosec B603
            [sys.executable, str(_PROBE)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
        assert completed.returncode == 0, f"probe interpreter failed:\n{completed.stderr}"
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        assert len(lines) == 1, f"probe printed {len(lines)} lines, expected exactly one: {completed.stdout!r}"
        parsed: dict[str, str] = json.loads(lines[0])
        outputs.append(parsed)
    return outputs


def test_select_deterministic_ignores_input_order() -> None:
    forward = ComputeFramework.select_deterministic([PandasDataFrame, PyArrowTable])
    backward = ComputeFramework.select_deterministic([PyArrowTable, PandasDataFrame])

    assert forward is backward
    assert forward is PandasDataFrame


def test_select_deterministic_returns_the_expected_name() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], ...], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
        ((zulu, tango, bravo), "ZzBravoThrowawayFramework"),
        ((zulu, alfa, tango, bravo), "ZzAlfaThrowawayFramework"),
    ]

    for group, expected in expectations:
        assert ComputeFramework.select_deterministic(set(group)).get_class_name() == expected


def test_select_deterministic_breaks_a_shared_class_name_by_qualname() -> None:
    """A shared class name would otherwise fall back to input order, which for a set is id-based."""
    alfa, bravo = _same_name_frameworks()

    forward = ComputeFramework.select_deterministic([alfa, bravo])
    backward = ComputeFramework.select_deterministic([bravo, alfa])

    assert forward is backward
    assert forward is alfa
    assert ".alfa." in forward.__qualname__, forward.__qualname__


def test_select_deterministic_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        ComputeFramework.select_deterministic([])


def test_the_throwaway_frameworks_stay_out_of_plugin_discovery() -> None:
    """get_cfw_subclasses is what planning consults; nothing this module defines may reach it."""
    held = set(_throwaway_frameworks()) | set(_same_name_frameworks())

    discovered = PreFilterPlugins.get_cfw_subclasses()

    assert not discovered & held, f"test-only frameworks reached discovery: {discovered & held}"
    leaked = sorted(cfw.get_class_name() for cfw in discovered if cfw.__module__ == __name__)
    assert leaked == [], f"test-only frameworks leaked into plugin discovery: {leaked}"


def test_link_trekker_key_reduces_both_sides_to_the_expected_framework() -> None:
    link = _link()

    key = ResolveLinks(Graph()).create_link_trekker_key(
        link, {PyArrowTable, PandasDataFrame}, {PyArrowTable, PandasDataFrame}
    )

    assert key == (link, PandasDataFrame, PandasDataFrame)


def test_link_trekker_key_reduces_every_framework_pair_the_same_way() -> None:
    resolver = ResolveLinks(Graph())
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], type[ComputeFramework]], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
    ]

    for (left, right), expected in expectations:
        link = _link()
        key = resolver.create_link_trekker_key(link, {left, right}, {left, right})

        assert key[0] is link
        assert key[1].get_class_name() == expected
        assert key[2].get_class_name() == expected


def test_link_trekker_key_keeps_single_framework_sides() -> None:
    link = _link()

    key = ResolveLinks(Graph()).create_link_trekker_key(link, {PandasDataFrame}, {PyArrowTable})

    assert key == (link, PandasDataFrame, PyArrowTable)


def test_feature_compute_framework_is_the_expected_framework() -> None:
    feature = Feature("determinism_feature")
    feature.compute_frameworks = {PyArrowTable, PandasDataFrame}

    assert feature.get_compute_framework() is PandasDataFrame


def test_feature_compute_framework_is_the_expected_framework_for_every_pair() -> None:
    zulu, alfa, tango, bravo = _throwaway_frameworks()
    expectations: list[tuple[tuple[type[ComputeFramework], type[ComputeFramework]], str]] = [
        ((zulu, alfa), "ZzAlfaThrowawayFramework"),
        ((zulu, tango), "ZzTangoThrowawayFramework"),
        ((tango, bravo), "ZzBravoThrowawayFramework"),
    ]

    for (left, right), expected in expectations:
        feature = Feature("determinism_feature")
        feature.compute_frameworks = {left, right}

        assert feature.get_compute_framework().get_class_name() == expected


def test_feature_compute_framework_keeps_a_single_framework() -> None:
    feature = Feature("determinism_single_feature")
    feature.compute_frameworks = {PyArrowTable}

    assert feature.get_compute_framework() is PyArrowTable


def test_feature_compute_framework_rejects_an_empty_set() -> None:
    feature = Feature("determinism_empty_feature")
    feature.compute_frameworks = set()

    with pytest.raises(ValueError, match="determinism_empty_feature"):
        feature.get_compute_framework()


# Fresh interpreters cost roughly a second each, so this one needs more than the suite-wide per-test budget.
@pytest.mark.timeout(60)
def test_fresh_interpreters_reduce_to_the_same_frameworks() -> None:
    outputs = _run_probes(_PROBE_PROCESSES)

    assert len(outputs) == _PROBE_PROCESSES, f"expected {_PROBE_PROCESSES} probe results, got {len(outputs)}"
    for position, output in enumerate(outputs):
        assert output == _PROBE_EXPECTED, f"probe {position} reduced to {output}, expected {_PROBE_EXPECTED}"
