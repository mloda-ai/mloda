"""Tests for DefaultOptionKeys graduation out of experimental namespace."""

import warnings


class TestDefaultOptionKeysNewPath:
    """Verify DefaultOptionKeys is importable from the new stable path."""

    def test_import_from_new_path(self) -> None:
        from mloda_plugins.feature_group.default_options_key import DefaultOptionKeys

        assert DefaultOptionKeys is not None

    def test_all_keys_exist_at_new_path(self) -> None:
        from mloda_plugins.feature_group.default_options_key import DefaultOptionKeys

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

    def test_values_match_at_new_path(self) -> None:
        from mloda_plugins.feature_group.default_options_key import DefaultOptionKeys

        assert DefaultOptionKeys.reference_time.value == "reference_time"
        assert DefaultOptionKeys.time_travel.value == "time_travel_filter"
        assert DefaultOptionKeys.group.value == "group"
        assert DefaultOptionKeys.order_by.value == "order_by"


class TestDefaultOptionKeysDeprecatedPath:
    """Verify the old experimental path still works but emits a DeprecationWarning."""

    def test_old_import_returns_same_class(self) -> None:
        from mloda_plugins.feature_group.default_options_key import DefaultOptionKeys as New

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from mloda_plugins.feature_group.experimental.default_options_key import (
                DefaultOptionKeys as Old,
            )

        assert New is Old

    def test_old_import_emits_deprecation_warning(self) -> None:
        import importlib

        mod = importlib.import_module(
            "mloda_plugins.feature_group.experimental.default_options_key"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(mod)

        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0, "Expected a DeprecationWarning from old import path"
        assert "mloda_plugins.feature_group.default_options_key" in str(
            deprecation_warnings[0].message
        )
