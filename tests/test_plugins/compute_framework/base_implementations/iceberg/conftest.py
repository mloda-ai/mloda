from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def iceberg_adapter() -> Iterator[Any]:
    """An opened IcebergAdapter (real SQL catalog in a temp warehouse), closed on teardown."""
    from mloda._testing.framework_adapter import IcebergAdapter

    adapter = IcebergAdapter()
    if not adapter.is_available():
        pytest.skip("Iceberg backend unavailable")
    with adapter:
        yield adapter
