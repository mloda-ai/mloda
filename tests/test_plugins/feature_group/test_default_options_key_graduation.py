"""Tests for DefaultOptionKeys graduation into mloda core."""

import warnings


class TestDefaultOptionKeysCoreLocation:
    """Verify DefaultOptionKeys lives in mloda core and is importable from mloda.provider."""

    def test_import_from_core(self) -> None:
        from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

        assert DefaultOptionKeys is not None

    def test_import_from_provider(self) -> None:
        from mloda.provider import DefaultOptionKeys

        assert DefaultOptionKeys is not None

    def test_all_keys_exist(self) -> None:
        from mloda.provider import DefaultOptionKeys

        expected_keys = [
            "in_features",
            "feature_chainer_parser_key",
            "reference_time",
            "time_travel",
            "default",
            "context",
            "group",
            "order_by",
            "strict_validation",
            "validation_function",
            "strict_type_enforcement",
        ]
        for key in expected_keys:
            assert hasattr(DefaultOptionKeys, key), f"Missing key: {key}"

    def test_values_match(self) -> None:
        from mloda.provider import DefaultOptionKeys

        assert DefaultOptionKeys.reference_time.value == "reference_time"
        assert DefaultOptionKeys.time_travel.value == "time_travel_filter"
        assert DefaultOptionKeys.group.value == "group"
        assert DefaultOptionKeys.order_by.value == "order_by"

    def test_core_and_provider_same_class(self) -> None:
        from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys as Core
        from mloda.provider import DefaultOptionKeys as Provider

        assert Core is Provider


class TestDefaultOptionKeysDeprecatedPaths:
    """Verify deprecated import paths still work with warnings."""

    def test_plugins_path_emits_deprecation(self) -> None:
        import importlib

        mod = importlib.import_module("mloda_plugins.feature_group.default_options_key")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(mod)

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "Expected a DeprecationWarning from plugins path"
        assert "mloda.provider" in str(deprecation_warnings[0].message)

    def test_experimental_path_emits_deprecation(self) -> None:
        import importlib

        mod = importlib.import_module("mloda_plugins.feature_group.experimental.default_options_key")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(mod)

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "Expected a DeprecationWarning from experimental path"
        assert "mloda.provider" in str(deprecation_warnings[0].message)

    def test_deprecated_paths_return_same_class(self) -> None:
        from mloda.provider import DefaultOptionKeys as Provider

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from mloda.provider import DefaultOptionKeys as Plugins
            from mloda.provider import DefaultOptionKeys as Experimental

        assert Provider is Plugins
        assert Provider is Experimental
