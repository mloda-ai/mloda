"""Connection-seam tests for IcebergFramework.open_connection() and ensure_connection()."""

from unittest.mock import Mock, patch

import pytest

try:
    import pyiceberg  # noqa: F401
    from pyiceberg.catalog import Catalog
except ImportError:
    pyiceberg = None  # type: ignore[assignment]
    Catalog = None  # type: ignore[assignment,misc]

from mloda.user import ConnectionSpec, DataAccessCollection, ParallelizationMode
from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework

pytestmark = pytest.mark.skipif(pyiceberg is None, reason="PyIceberg is not installed.")

_LOAD_CATALOG_TARGET = "mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework.load_catalog"


def _mock_catalog() -> Mock:
    catalog = Mock(spec=Catalog)
    catalog.load_table = Mock()
    return catalog


def test_set_framework_connection_object_none_raises() -> None:
    fw = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    with pytest.raises(ValueError, match="Iceberg catalog or table is required"):
        fw.set_framework_connection_object(None)


def test_open_connection_none_returns_none() -> None:
    assert IcebergFramework.open_connection(None) is None


def test_open_connection_with_spec_calls_load_catalog_and_returns_result() -> None:
    spec = ConnectionSpec(IcebergFramework, name="lake", type="rest", uri="http://x")
    mock_catalog = _mock_catalog()
    mock_load_catalog = Mock(return_value=mock_catalog)

    with patch(_LOAD_CATALOG_TARGET, mock_load_catalog):
        result = IcebergFramework.open_connection(spec)

    mock_load_catalog.assert_called_once_with(name="lake", type="rest", uri="http://x")
    assert result is mock_catalog


def test_ensure_connection_from_spec_source_binds_the_mocked_catalog() -> None:
    fw = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    spec = ConnectionSpec(IcebergFramework, name="lake", type="rest", uri="http://x")
    fw.connection_source = ConnectionSource(spec=spec)
    mock_catalog = _mock_catalog()
    mock_load_catalog = Mock(return_value=mock_catalog)

    with patch(_LOAD_CATALOG_TARGET, mock_load_catalog):
        result = fw.ensure_connection()

    assert result is mock_catalog
    assert fw.framework_connection_object is mock_catalog


def test_open_connection_raises_import_error_when_load_catalog_missing() -> None:
    spec = ConnectionSpec(IcebergFramework, name="lake", type="rest", uri="http://x")

    with patch(_LOAD_CATALOG_TARGET, None):
        with pytest.raises(ImportError):
            IcebergFramework.open_connection(spec)


def test_pick_connection_from_dac_returns_matching_spec() -> None:
    spec = ConnectionSpec(IcebergFramework, name="lake", type="rest", uri="http://x")
    dac = DataAccessCollection(connections={spec})
    assert IcebergFramework.pick_connection_from_dac(dac) is spec
