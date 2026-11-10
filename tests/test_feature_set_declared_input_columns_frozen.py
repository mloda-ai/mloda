from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet


def test_freeze_makes_declared_input_columns_read_only() -> None:
    feature_set = FeatureSet()
    feature_set.declared_input_columns = {"consumer": ("input",)}
    feature_set.freeze()

    assert feature_set.declared_input_columns is not None
    with pytest.raises(TypeError):
        feature_set.declared_input_columns["consumer"] = ("missing_column",)  # type: ignore[index]


def test_for_run_view_shares_frozen_declared_input_columns() -> None:
    feature_set = FeatureSet()
    feature_set.declared_input_columns = {"consumer": ("input",)}
    feature_set.freeze()

    view = feature_set.for_run(None)

    assert view.declared_input_columns is feature_set.declared_input_columns
    with pytest.raises(TypeError):
        view.declared_input_columns["consumer"] = ("missing_column",)  # type: ignore[index]
    assert feature_set.declared_input_columns == {"consumer": ("input",)}
