"""Pins the PropertySpec module location to components/, not components/feature_chainer/.

property_spec.py is shared by PROPERTY_MAPPING (FeatureGroup) and READER_OPTIONS
(BaseInputData, which does not chain features), so nesting it under feature_chainer/
misrepresents what it is.
"""

import importlib

import pytest


class TestPropertySpecNewLocation:
    """PropertySpec exports are importable from the relocated module path."""

    def test_property_spec_class_importable_from_new_path(self) -> None:
        from mloda.core.abstract_plugins.components.property_spec import PropertySpec

        assert PropertySpec is not None

    def test_property_spec_builder_importable_from_new_path(self) -> None:
        from mloda.core.abstract_plugins.components.property_spec import property_spec

        assert property_spec is not None

    def test_no_default_importable_from_new_path(self) -> None:
        from mloda.core.abstract_plugins.components.property_spec import NO_DEFAULT

        assert NO_DEFAULT is not None

    def test_is_no_default_importable_from_new_path(self) -> None:
        from mloda.core.abstract_plugins.components.property_spec import is_no_default

        assert is_no_default is not None

    def test_is_positive_int_importable_from_new_path(self) -> None:
        from mloda.core.abstract_plugins.components.property_spec import is_positive_int

        assert is_positive_int is not None


class TestPropertySpecOldLocationRemoved:
    """The old feature_chainer-nested module path no longer exists."""

    def test_old_module_path_raises_module_not_found(self) -> None:
        with pytest.raises((ModuleNotFoundError, ImportError)):
            importlib.import_module("mloda.core.abstract_plugins.components.feature_chainer.property_spec")
