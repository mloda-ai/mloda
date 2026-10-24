"""End-to-end TFS connection-propagation test for Iceberg.

IcebergPyArrowTransformer.transform_other_fw_to_fw currently passes PyArrow
data through (no real catalog write yet — the framework only consumes it). So
the destination FG operates on a pa.Table and verifies that the catalog from
the DataAccessCollection has been bound on the IcebergFramework instance by
asserting it from inside calculate_feature via a per-test sentinel."""

import contextlib
import sqlite3
from typing import Any
from unittest.mock import Mock

import pytest

try:
    import pyiceberg  # noqa: F401
    from pyiceberg.catalog import Catalog
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pyiceberg = None  # type: ignore[assignment]
    Catalog = None  # type: ignore[assignment,misc]
    pa = None  # type: ignore[assignment, unused-ignore]
    pc = None

from mloda.user import Feature, DataAccessCollection
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
    TfsRawValPyArrowSource,
)

_observed_catalogs: list[Any] = []


class ConnectionRecordingIcebergFramework(IcebergFramework):
    """Test-only subclass that records the bound catalog so the e2e test can
    assert the setup-time precompute reached this CFW instance."""

    def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
        super().set_framework_connection_object(framework_connection_object)
        _observed_catalogs.append(self.framework_connection_object)

    @classmethod
    def pick_connection_from_dac(cls, data_access_collection: Any, options: Any | None = None) -> Any | None:
        return IcebergFramework.pick_connection_from_dac(data_access_collection, options)


class TfsDoubledIcebergFG(FeatureGroup):
    """Doubles raw_val via PyArrow compute (Iceberg transformer is a passthrough)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {ConnectionRecordingIcebergFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        doubled = pc.multiply(data["raw_val"], 2)
        return data.append_column("tfs_doubled", doubled)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


class TfsHandleDoubledIcebergFG(FeatureGroup):
    """Doubles raw_val on Iceberg, forwarding its options (incl. connection_handle) to the input."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("raw_val", options=options)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.append_column("tfs_handle_doubled", pc.multiply(data["raw_val"], 2))

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_handle_doubled"}


@pytest.mark.skipif(pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed.")
class TestIcebergTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = ConnectionRecordingIcebergFramework
    destination_fg_class = TfsDoubledIcebergFG

    def test_a_root_runs_on_the_one_dac_connection_without_a_class_name_option(self, live_connection: Any) -> None:
        pytest.skip("Iceberg is read-only (no root conversion on a bound relation) and has no .connection on data")

    def test_a_connection_on_the_feature_needs_no_data_access_collection(self, live_connection: Any) -> None:
        pytest.skip("Iceberg is read-only (no root conversion on a bound relation) and has no .connection on data")

    def test_a_hop_into_iceberg_with_a_foreign_handle_runs_without_a_connection(self) -> None:
        with contextlib.closing(sqlite3.connect(":memory:")) as foreign:
            result = mloda.run_all(
                [Feature("tfs_handle_doubled", Options(group={"connection_handle": "d"}))],
                compute_frameworks=[PyArrowTable, IcebergFramework],
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {TfsRawValPyArrowSource, TfsHandleDoubledIcebergFG}
                ),
                data_access_collection=DataAccessCollection({"d": foreign}),
            )

        assert result[0].column("tfs_handle_doubled").to_pylist() == [2, 4, 6]

    def test_a_hop_into_iceberg_with_two_catalogs_and_no_handle_is_ambiguous(self) -> None:
        catalogs = {"c1": Mock(spec=Catalog), "c2": Mock(spec=Catalog)}
        for catalog in catalogs.values():
            catalog.load_table = Mock()

        with pytest.raises(ValueError, match="Ambiguous resolve") as exc:
            mloda.run_all(
                [Feature("tfs_handle_doubled")],
                compute_frameworks=[PyArrowTable, IcebergFramework],
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {TfsRawValPyArrowSource, TfsHandleDoubledIcebergFG}
                ),
                data_access_collection=DataAccessCollection(catalogs),
            )
        assert "IcebergFramework: Ambiguous resolve" in str(exc.value)

    @pytest.fixture
    def live_connection(self) -> Any:
        _observed_catalogs.clear()
        mock_catalog = Mock(spec=Catalog)
        mock_catalog.load_table = Mock()
        return mock_catalog

    def extract_doubled(self, final: Any) -> list[int]:
        assert _observed_catalogs, "Setup-time precompute did not bind a catalog on the destination CFW"
        assert isinstance(final, pa.Table)
        assert "tfs_doubled" in final.column_names
        return list(final.column("tfs_doubled").to_pylist())
