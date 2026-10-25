"""Shared base for undeclared column reads driven through ``mloda.run_all``."""

from typing import Any

import pytest

from mloda.provider import DataCreatorFeatureGroup
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import GlobalFilter
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.core.abstract_plugins.components.feature_name import FeatureName

from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import (
    PolicyRaises,
    PolicyRunAllTestBase,
)

UNDECLARED_COLUMN = "mc_undeclared_col"


class MissingColumnRoot(DataCreatorFeatureGroup):
    """Root FeatureGroup producing the one declared column ``mc_declared``."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"mc_declared"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"mc_declared": [1, 2, 3]}


class MissingColumnConsumer(FeatureGroup):
    """Consumer declaring ``mc_declared`` whose subclass reads an undeclared column."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("mc_declared")}

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def cause_type(cls) -> Any:
        """Original exception type for this read style."""
        raise NotImplementedError

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.read_undeclared(data)


class KeyReadConsumer(MissingColumnConsumer):
    """Subscripts the undeclared column (dict or pandas frame)."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        return data[UNDECLARED_COLUMN]

    @classmethod
    def cause_type(cls) -> Any:
        return KeyError


class SelectReadConsumer(MissingColumnConsumer):
    """Selects the undeclared column on a SQL relation."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        return data.select(UNDECLARED_COLUMN)

    @classmethod
    def cause_type(cls) -> Any:
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import MissingRelationColumnError

        return MissingRelationColumnError


class PyArrowKeyReadConsumer(MissingColumnConsumer):
    """Reads the undeclared column from a PyArrow table by name."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        return data.column(UNDECLARED_COLUMN)

    @classmethod
    def cause_type(cls) -> Any:
        return KeyError


class PyArrowFilterReadConsumer(MissingColumnConsumer):
    """Filters a PyArrow table on the undeclared column."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        import pyarrow.compute as pc

        return data.filter(pc.field(UNDECLARED_COLUMN) > 0)

    @classmethod
    def cause_type(cls) -> Any:
        import pyarrow as pa

        return pa.ArrowInvalid


class AttributeReadConsumer(MissingColumnConsumer):
    """Reads the undeclared column as an attribute (``df.x``)."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        return getattr(data, UNDECLARED_COLUMN)

    @classmethod
    def cause_type(cls) -> Any:
        return AttributeError


class MisspelledMethodConsumer(MissingColumnConsumer):
    """Calls a misspelled method (``data.fliter``), which is a typo and not a column read."""

    @classmethod
    def read_undeclared(cls, data: Any) -> Any:
        return data.fliter

    @classmethod
    def cause_type(cls) -> Any:
        return AttributeError


class MissingColumnRunAllTestBase(PolicyRunAllTestBase):
    """Each consumer FG from ``consumer_fgs`` raises an explanatory ``ValueError`` through ``run_all``."""

    @classmethod
    def root_fg(cls) -> type[FeatureGroup]:
        """Root FeatureGroup producing ``mc_declared``; override for a framework-native root."""
        return MissingColumnRoot

    @classmethod
    def consumer_fgs(cls) -> list[type[MissingColumnConsumer]]:
        raise NotImplementedError

    @classmethod
    def pytest_generate_tests(cls, metafunc: Any) -> None:
        if "consumer_fg" in metafunc.fixturenames:
            fgs = cls.consumer_fgs()
            metafunc.parametrize("consumer_fg", fgs, ids=[fg.__name__ for fg in fgs])

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC])
    def test_undeclared_column_read_raises_explanatory_error(
        self, consumer_fg: type[MissingColumnConsumer], mode: ParallelizationMode, flight_server: Any
    ) -> None:
        self.assert_undeclared_read_raises(consumer_fg, mode, flight_server)

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC])
    def test_misspelled_method_is_not_translated(self, mode: ParallelizationMode, flight_server: Any) -> None:
        error = self.assert_policy_case(
            feature_name=MisspelledMethodConsumer.__name__,
            plugin_collector=PluginCollector.enabled_feature_groups({self.root_fg(), MisspelledMethodConsumer}),
            expectation=PolicyRaises(match_substring="fliter", error_type=AttributeError),
            mode=mode,
            flight_server=flight_server,
        )
        assert "input_features" not in str(error)

    def assert_undeclared_read_raises(
        self,
        consumer_fg: type[MissingColumnConsumer],
        mode: ParallelizationMode,
        flight_server: Any,
        global_filter: GlobalFilter | None = None,
    ) -> BaseException:
        error = self.assert_policy_case(
            feature_name=consumer_fg.__name__,
            plugin_collector=PluginCollector.enabled_feature_groups({self.root_fg(), consumer_fg}),
            expectation=PolicyRaises(
                match_substring=consumer_fg.__name__,
                error_type=ValueError,
                cause_type=consumer_fg.cause_type(),
                also_contains=(consumer_fg.cause_type().__name__, UNDECLARED_COLUMN, "input_features"),
            ),
            mode=mode,
            flight_server=flight_server,
            global_filter=global_filter,
        )
        assert isinstance(error, BaseException)
        return error
