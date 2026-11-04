from collections.abc import Iterator
from typing import Any

import pytest

from tests.test_plugins.compute_framework.base_implementations.iceberg.iceberg_test_table import adapter_handle


@pytest.fixture
def iceberg_adapter() -> Iterator[Any]:
    """An opened IcebergAdapter (real SQL catalog in a temp warehouse), closed on teardown."""
    from mloda._testing.framework_adapter import IcebergAdapter

    adapter = IcebergAdapter()
    if not adapter.is_available():
        pytest.skip("Iceberg backend unavailable")
    with adapter:
        yield adapter


@pytest.fixture
def iceberg_table_source(iceberg_adapter: Any) -> Iterator[None]:
    """Publish the open adapter for real_iceberg_table and clear it on teardown."""
    adapter_handle.append(iceberg_adapter)
    yield
    adapter_handle.clear()
