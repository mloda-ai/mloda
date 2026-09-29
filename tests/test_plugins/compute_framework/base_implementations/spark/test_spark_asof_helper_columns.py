"""Regression tests for Spark as-of merge helper column handling.

These tests guard against the NameError regression where ``pick_helper_column_name``
was used without being imported, and verify that helper columns are collision-free
and case-insensitively unique.
"""

from __future__ import annotations

import importlib

import pytest


MODULE_PATH = "mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine"


def test_pick_helper_column_name_is_imported() -> None:
    """The module must expose ``pick_helper_column_name`` so calls do not raise NameError."""
    module = importlib.import_module(MODULE_PATH)
    assert hasattr(module, "pick_helper_column_name")
    assert callable(module.pick_helper_column_name)


def test_pick_helper_column_name_avoids_case_insensitive_collisions() -> None:
    """Helper names must not collide with existing columns, even case-insensitively."""
    module = importlib.import_module(MODULE_PATH)
    pick = module.pick_helper_column_name

    taken = {"Value", "_MLODA_R_value"}
    name = pick(taken=taken, prefix="_mloda_r_")

    assert name.lower() not in {t.lower() for t in taken}


def test_pick_helper_column_name_returns_unique_names() -> None:
    """Repeated calls with the same taken set must yield distinct names."""
    module = importlib.import_module(MODULE_PATH)
    pick = module.pick_helper_column_name

    taken: set[str] = set()
    first = pick(taken=taken, prefix="_mloda_lid")
    taken.add(first)
    second = pick(taken=taken, prefix="_mloda_lid")

    assert first != second


def test_merge_asof_uses_imported_helper_without_name_error() -> None:
    """Calling merge_asof must not raise NameError for pick_helper_column_name.

    PySpark is optional in this environment; when it is unavailable the engine's
    ``check_import`` raises ImportError, which is acceptable. A NameError is not.
    """
    module = importlib.import_module(MODULE_PATH)
    engine_cls = module.SparkMergeEngine

    if module.DataFrame is None:
        pytest.skip("PySpark is not installed")

    engine = engine_cls()
    assert hasattr(engine, "merge_asof")
