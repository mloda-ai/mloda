"""A conftest-free host of the source and column discovery suites leaves no suite-built subclass behind."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

INNER = """
import gc

import pytest

from mloda._testing.source.column_discovery_suite import ColumnDiscoveryContractTestMixin
from mloda._testing.source.source_suite import SourceFeatureGroupTestMixin, _chain_consumer
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from tests.test_core.test_abstract_plugins.test_components.test_input_data import toy_source_group
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG

SUITE_BUILT = ("SuiteTakeover", "SuiteChainConsumer", "_CountRowsProbeFG")


@pytest.fixture(autouse=True)
def _no_suite_built_class_remains():
    gc.disable()
    yield
    leaked = [c.__name__ for c in get_all_subclasses(FeatureGroup) if any(t in c.__name__ for t in SUITE_BUILT)]
    gc.enable()
    gc.collect()
    assert not leaked, f"suite-built classes left registered: {leaked}"


class TestToyHost(SourceFeatureGroupTestMixin):
    feature_group_class = ToySourceFG
    present_column = "toyfmt_col"
    missing_column = "toyfmt_missing"
    expected_source = "h1:toy"
    pointed_source = "scoped:toy"

    def own_dac(self):
        return toy_source_group.toy_dac(h1={"toyfmt_col": [1, 2]})

    def foreign_dac(self):
        return toy_source_group.foreign_dac()

    def own_pointer(self):
        return {"toyfmt_col": [1, 2]}

    def other_pointer(self):
        return {"toyfmt_col": [9]}, "scoped:toy"

    def foreign_handle_dac(self):
        return toy_source_group.foreign_dac(), "toyfmt_foreign"

    def known_handles_dac(self):
        return toy_source_group.toy_dac(h1={"toyfmt_col": [1]}, h2={"toyfmt_col": [2]}), ["h1", "h2"]

    def test_host_builds_a_chain_consumer(self):
        _chain_consumer("toyfmt_col")


class TestJsonDiscoveryHost(ColumnDiscoveryContractTestMixin):
    group_cls = JsonFG
    dependency_module = "pyarrow.json"
    expected_row_count = None

    @pytest.fixture
    def data_file(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text("".join('{"c1": %d, "a1": %d, "b1": %d}\\n' % (i, i, i) for i in range(3)))
        return str(path)
"""

SELECTED = "takes_over or another_concrete_group or host_builds_a_chain_consumer or load_neutral_overriding"


def test_hosts_without_a_conftest_leave_no_suite_built_class(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_inner=INNER)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly", "-k", SELECTED)

    result.assert_outcomes(passed=4, errors=0)
