"""
Characterization test for IcebergFramework merge support.

Iceberg has no merge engine at all: merge_engine() raises NotImplementedError
unconditionally. This pins the current behavior so the ASOF extension does not
accidentally claim Iceberg support.
"""

import pytest

from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework


def test_merge_engine_not_implemented() -> None:
    """IcebergFramework.merge_engine() raises NotImplementedError (no merge engine)."""
    with pytest.raises(NotImplementedError):
        IcebergFramework.merge_engine()
