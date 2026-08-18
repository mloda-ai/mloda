"""Pins the PropertySpec module location to components/, not components/feature_chainer/.

property_spec.py is shared by PROPERTY_MAPPING (FeatureGroup) and READER_OPTIONS
(BaseInputData, which does not chain features), so nesting it under feature_chainer/
misrepresents what it is.
"""

import importlib

import pytest


class TestPropertySpecNewLocation:
    """PropertySpec exports are importable from the relocated module path."""

    def test_all_exports_importable_from_new_path(self) -> None:
        from mloda.core.abstract_plugins.components.property_spec import (
            NO_DEFAULT,
            PropertySpec,
            is_no_default,
            is_positive_int,
            property_spec,
        )

        assert PropertySpec is not None
        assert property_spec is not None
        assert NO_DEFAULT is not None
        assert is_no_default is not None
        assert is_positive_int is not None


class TestPropertySpecOldLocationRemoved:
    """The old feature_chainer-nested module path no longer exists."""

    def test_old_module_path_raises_module_not_found(self) -> None:
        feature_chainer = importlib.import_module("mloda.core.abstract_plugins.components.feature_chainer")
        assert feature_chainer is not None

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mloda.core.abstract_plugins.components.feature_chainer.property_spec")
