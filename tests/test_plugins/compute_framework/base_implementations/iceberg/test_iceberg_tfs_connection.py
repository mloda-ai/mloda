"""End-to-end TFS connection-propagation test for Iceberg.

IcebergPyArrowTransformer.transform_other_fw_to_fw currently passes PyArrow
data through (no real catalog write yet — the framework only consumes it). So
the destination FG operates on a pa.Table and verifies that the catalog from
the DataAccessCollection has been bound on the IcebergFramework instance by
asserting it from inside calculate_feature via a per-test sentinel."""

from typing import Any, Optional
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
    pa = None  # type: ignore[assignment]
    pc = None

from mloda.user import Feature, DataAccessCollection
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
)


_observed_catalogs: list[Any] = []


class ConnectionRecordingIcebergFramework(IcebergFramework):
    """Test-only subclass that records the bound catalog so the e2e test can
    assert the setup-time precompute reached this CFW instance."""

    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        super().set_framework_connection_object(framework_connection_object)
        _observed_catalogs.append(self.framework_connection_object)

    @classmethod
    def pick_connection_from_dac(cls, data_access_collection: Any, options: Optional[Any] = None) -> Optional[Any]:
        return IcebergFramework.pick_connection_from_dac(data_access_collection, options)


class TfsDoubledIcebergFG(FeatureGroup, MatchData):
    """Doubles raw_val via PyArrow compute (Iceberg transformer is a passthrough)."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if pyiceberg is None:
            return None
        if feature_name not in cls.feature_names_supported():
            return None
        if framework_connection_object is not None and (
            hasattr(framework_connection_object, "load_table") or isinstance(framework_connection_object, Mock)
        ):
            return framework_connection_object
        if data_access_collection is None:
            return None
        for conn in data_access_collection.connections.values():
            if hasattr(conn, "load_table") or isinstance(conn, Mock):
                return conn
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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


@pytest.mark.skipif(pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed.")
class TestIcebergTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = ConnectionRecordingIcebergFramework
    destination_fg_class = TfsDoubledIcebergFG

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
