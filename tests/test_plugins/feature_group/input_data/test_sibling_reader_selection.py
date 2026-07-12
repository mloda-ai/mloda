"""
Pins the input-data reader SELECTION contract (issue #565).

A feature selects a specific reader with an Option whose key equals the reader's
class name (``BaseInputData.data_access_name()``, i.e. ``cls.__name__``). The key
may also be the reader class itself. The matched ``(ReaderClass, data_access)``
pair is stored under the reserved ``"BaseInputData"`` options key and consumed by
``init_reader`` at load time. For non-file sources (e.g. HTTP), subclassing
``ReadFile`` and overriding ``match_subclass_data_access`` plus ``load_data`` is
the sanctioned pattern; ``suffix()`` is inert on that path.

Isolation: test readers defined here are discovered process-wide via
``get_all_subclasses``. Every reader's ``match_subclass_data_access`` therefore
requires a unique marker value (or marker options key) and returns None otherwise,
so it can never hijack matching in other tests running in the same worker process.
"""

from typing import Any, cast

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature


_ACCESS_A = "sibling_sel_565_access_a"
_ACCESS_B = "sibling_sel_565_access_b"
_URL_MARKER_KEY = "sibling_sel_565_url_marker"
_URL_ACCESS = "fake-http://example/sibling_sel_565/data"
_URL_FEATURE_NAME = "sibling_sel_565_url_value"
_URL_FEATURE_VALUES = [11, 22, 33]


class SiblingSel565ReaderA(ReadFile):
    """Final reader (wholesale load_data override) that only matches its own marker access string."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if data_access == _ACCESS_A:
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_565_a": [1]}


class SiblingSel565ReaderB(ReadFile):
    """Sibling of SiblingSel565ReaderA; only matches its own marker access string."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if data_access == _ACCESS_B:
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_565_b": [2]}


class SiblingSel565UrlReader(ReadFile):
    """URL-style reader recipe: overrides match_subclass_data_access and load_data wholesale.

    Deliberately does NOT override suffix(): the option-key selection path never
    consults it, which the end-to-end tests pin. Matching requires this test's
    unique marker option key so the reader never matches in other tests.
    """

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if options.get(_URL_MARKER_KEY) is None:
            return None
        if isinstance(data_access, str) and data_access.startswith("fake-http://"):
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pa.table({_URL_FEATURE_NAME: _URL_FEATURE_VALUES})


class SiblingSel565NotAReader:
    """Has get_class_name but is not a BaseInputData subclass."""

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__


class TestSiblingSelectionViaClassNameStringKey:
    """Group A: sibling selection via class-name string option key (unit level)."""

    def test_string_key_routes_to_reader_a(self) -> None:
        options = Options({SiblingSel565ReaderA.__name__: _ACCESS_A})
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_565_feat") is True
        assert options.get("BaseInputData") == (SiblingSel565ReaderA, _ACCESS_A)

    def test_string_key_routes_to_reader_b_no_sibling_collision(self) -> None:
        options = Options({SiblingSel565ReaderB.__name__: _ACCESS_B})
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_565_feat") is True
        assert options.get("BaseInputData") == (SiblingSel565ReaderB, _ACCESS_B)

    def test_unknown_key_matches_no_reader(self) -> None:
        options = Options({"SiblingSel565NoSuchReader": _ACCESS_A})
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_565_feat") is False
        assert "BaseInputData" not in options


class TestSiblingSelectionViaClassKey:
    """Group B: the option key may be the reader class itself."""

    def test_class_as_key_resolves_like_string_form(self) -> None:
        options = Options(cast(dict[str, Any], {SiblingSel565ReaderA: _ACCESS_A}))
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_565_feat") is True
        assert options.get("BaseInputData") == (SiblingSel565ReaderA, _ACCESS_A)

    def test_key_normalization_string_and_class(self) -> None:
        assert BaseInputData.deal_with_base_input_data_name_as_cls_or_str("SiblingSel565ReaderA") == (
            "SiblingSel565ReaderA"
        )
        assert (
            BaseInputData.deal_with_base_input_data_name_as_cls_or_str(SiblingSel565ReaderA) == "SiblingSel565ReaderA"
        )

    def test_non_base_input_data_class_key_raises(self) -> None:
        with pytest.raises(ValueError, match="not a subclass of BaseInputData"):
            BaseInputData.deal_with_base_input_data_name_as_cls_or_str(SiblingSel565NotAReader)

    def test_non_string_key_raises(self) -> None:
        with pytest.raises(ValueError, match="is not a string"):
            BaseInputData.deal_with_base_input_data_name_as_cls_or_str(42)


class TestReservedBaseInputDataKey:
    """Group C: reserved "BaseInputData" options key semantics."""

    def test_add_base_input_data_identical_pair_is_noop_different_pair_raises(self) -> None:
        options = Options()
        BaseInputData.add_base_input_data_to_options(SiblingSel565ReaderA, _ACCESS_A, options)
        BaseInputData.add_base_input_data_to_options(SiblingSel565ReaderA, _ACCESS_A, options)
        assert options.get("BaseInputData") == (SiblingSel565ReaderA, _ACCESS_A)
        with pytest.raises(ValueError, match="BaseInputData already set with different values"):
            BaseInputData.add_base_input_data_to_options(SiblingSel565ReaderB, _ACCESS_B, options)

    def test_init_reader_consumes_stored_tuple(self) -> None:
        options = Options(group={"BaseInputData": (SiblingSel565ReaderA, _ACCESS_A)})
        reader, data_access = SiblingSel565ReaderA().init_reader(options)
        assert isinstance(reader, SiblingSel565ReaderA)
        assert data_access == _ACCESS_A

    def test_init_reader_none_options_raises(self) -> None:
        with pytest.raises(ValueError, match="Options were not set"):
            SiblingSel565ReaderA().init_reader(None)

    def test_init_reader_missing_base_input_data_key_raises(self) -> None:
        with pytest.raises(ValueError, match="'BaseInputData' key is missing"):
            SiblingSel565ReaderA().init_reader(Options())


class TestUrlReaderRecipeEndToEnd:
    """Group D: sanctioned non-file (HTTP-style) reader recipe, end to end."""

    def test_url_reader_is_final_and_suffix_inert(self) -> None:
        assert SiblingSel565UrlReader.is_final_reader() is True
        with pytest.raises(NotImplementedError):
            SiblingSel565UrlReader.suffix()

    def test_url_reader_end_to_end_string_key(self) -> None:
        feature = Feature(
            name=_URL_FEATURE_NAME,
            options={
                SiblingSel565UrlReader.__name__: _URL_ACCESS,
                _URL_MARKER_KEY: True,
            },
        )
        features: list[Feature | str] = [feature]
        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({ReadFileFeature}),
        )
        assert result[0].to_pydict()[_URL_FEATURE_NAME] == _URL_FEATURE_VALUES

    def test_url_reader_end_to_end_class_key(self) -> None:
        feature = Feature(
            name=_URL_FEATURE_NAME,
            options=cast(
                dict[str, Any],
                {
                    SiblingSel565UrlReader: _URL_ACCESS,
                    _URL_MARKER_KEY: True,
                },
            ),
        )
        features: list[Feature | str] = [feature]
        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({ReadFileFeature}),
        )
        assert result[0].to_pydict()[_URL_FEATURE_NAME] == _URL_FEATURE_VALUES
