"""
Pins the input-data reader SELECTION contract (issue #565).

A feature selects a specific reader with an Option whose key equals the reader's
class name (``BaseInputData.data_access_name()``, i.e. ``cls.__name__``). The key
may also be the reader class itself. The matched ``(ReaderClass, data_access)``
pair is written to the matcher's options fork under the reserved ``"BaseInputData"`` key, moved onto
``Feature.input_data_match`` and consumed by ``init_reader`` at load time. For non-file sources (e.g. HTTP), subclassing
a reader and overriding ``match_subclass_data_access`` plus ``load_data`` is
the sanctioned pattern; ``suffix()`` is inert on that path. A reader matches only itself: a root group
returns the concrete reader, and a subclass of it is never consulted.

Isolation: test readers defined here are discovered process-wide via
``get_all_subclasses``. Every reader's ``match_subclass_data_access`` therefore
requires a unique marker value (or marker options key) and returns None otherwise,
so it can never hijack matching in other tests running in the same worker process.
"""

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass, resolve_or_raise
from mloda.provider import FeatureGroup, FeatureSet, PropertySpec, ReadFileFG
from mloda.user import Feature, FeatureName, PluginCollector, mloda
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG


_ACCESS_A = "sibling_sel_565_access_a"
_ACCESS_B = "sibling_sel_565_access_b"


class SiblingSelFamily(BaseInputData):
    """Test-local reader base; it overrides no load_data."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return None


class SiblingSelFG(FeatureGroup):
    """Root group returning reader A; matches only what A accepts (its unique marker access)."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return SiblingSel565ReaderA()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.input_data().load(features)  # type: ignore[union-attr]


class SiblingSel565ReaderA(SiblingSelFamily):
    """Final reader (wholesale load_data override) that only matches its own marker access string."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if data_access == _ACCESS_A:
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_565_a": [1]}


class SiblingSel565ReaderB(SiblingSelFamily):
    """Sibling of SiblingSel565ReaderA; only matches its own marker access string."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if data_access == _ACCESS_B:
            return data_access
        return None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_565_b": [2]}


_URL_MARKER_KEY = "sibling_sel_565_url_marker"
_URL_ACCESS = "fake-http://example/sibling_sel_565/data"
_URL_FEATURE_NAME = "sibling_sel_565_url_value"
_URL_FEATURE_VALUES = [11, 22, 33]


class SiblingSel565UrlReader(SiblingSelFamily):
    """Non-file reader recipe: final, no suffix, gated by a unique marker option key."""

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


class SiblingSel565UrlChild(SiblingSel565UrlReader):
    """Subclass of the URL reader overriding load_data; a root returning the parent must never run it."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pa.table({_URL_FEATURE_NAME: [-1]})


class SiblingSelUrlFG(FeatureGroup):
    """Root group returning the URL reader."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return SiblingSel565UrlReader()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.input_data().load(features)  # type: ignore[union-attr]


class SiblingSel565NotAReader:
    """Has get_class_name but is not a BaseInputData subclass."""

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__


class TestSiblingSelectionViaClassNameStringKey:
    """Group A: a reader scopes itself via its class-name string option key (unit level)."""

    def test_string_key_routes_to_reader_a(self) -> None:
        options = Options({SiblingSel565ReaderA.__name__: _ACCESS_A})
        assert SiblingSel565ReaderA.feature_scope_data_access(options, "sibling_sel_565_feat") is True
        assert options.get("BaseInputData") == (SiblingSel565ReaderA, _ACCESS_A)

    def test_string_key_routes_to_reader_b_no_sibling_collision(self) -> None:
        options = Options({SiblingSel565ReaderB.__name__: _ACCESS_B})
        assert SiblingSel565ReaderB.feature_scope_data_access(options, "sibling_sel_565_feat") is True
        assert options.get("BaseInputData") == (SiblingSel565ReaderB, _ACCESS_B)

    def test_unknown_key_matches_no_reader(self) -> None:
        options = Options({"SiblingSel565NoSuchReader": _ACCESS_A})
        assert SiblingSel565ReaderA.feature_scope_data_access(options, "sibling_sel_565_feat") is False
        assert "BaseInputData" not in options

    def test_a_reader_ignores_the_key_of_a_sibling(self) -> None:
        options = Options({SiblingSel565ReaderA.__name__: _ACCESS_A, SiblingSel565ReaderB.__name__: _ACCESS_B})
        assert SiblingSel565ReaderA.feature_scope_data_access(options, "sibling_sel_565_feat") is True
        assert options.get("BaseInputData") == (SiblingSel565ReaderA, _ACCESS_A)

    def test_base_class_selects_nothing_from_a_reader_key(self) -> None:
        options = Options({SiblingSel565ReaderA.__name__: _ACCESS_A})
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_565_feat") is False
        assert "BaseInputData" not in options


_REPL_ALIAS = "sibling_sel_1777_alias"
_REPL_MARKER = "sibling_sel_1777_repl_marker"


class SiblingSel1777ReplParent(SiblingSelFamily):
    """Final aliased parent accepting only its unique marker."""

    @classmethod
    def data_access_name(cls) -> str:
        return _REPL_ALIAS

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return data_access if data_access == _REPL_MARKER else None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1777_repl_parent": [1]}


class SiblingSel1777ReplChild(SiblingSel1777ReplParent):
    """Final child inheriting the parent's alias and match."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1777_repl_child": [1]}


_VALUE_KEY = "sibling_sel_1777_mode"
_BAD_VALUE = "sibling_sel_1777_bad_mode"
_VALUE_FEATURE = "sibling_sel_1777_value_feat"


class SiblingSel1777ValueFG(FeatureGroup):
    """Root group returning reader A; the required strict mapped key keeps it from matching other tests' features."""

    PROPERTY_MAPPING = {
        _VALUE_KEY: PropertySpec("Mode", allowed_values={"good": "Good mode"}, context=True, strict_validation=True),
    }

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return SiblingSel565ReaderA()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_VALUE_FEATURE}


class TestPinnedReaderSelf1777:
    """A pinned reader matches only itself: a parent, child or sibling key never serves it."""

    def test_parent_key_selects_the_parent_not_a_final_subclass(self) -> None:
        options = Options({_REPL_ALIAS: _REPL_MARKER})
        assert SiblingSel1777ReplParent.feature_scope_data_access(options, "sibling_sel_1777_repl_feat") is True
        assert options.get("BaseInputData") == (SiblingSel1777ReplParent, _REPL_MARKER)

    def test_child_probe_serves_itself_through_the_inherited_key(self) -> None:
        options = Options({_REPL_ALIAS: _REPL_MARKER})
        assert SiblingSel1777ReplChild.feature_scope_data_access(options, "sibling_sel_1777_repl_feat") is True
        assert options.get("BaseInputData") == (SiblingSel1777ReplChild, _REPL_MARKER)

    def test_declining_pin_eliminates_naming_the_reader(self) -> None:
        feature = Feature(
            name="sibling_sel_1777_decline_feat",
            options={SiblingSel565ReaderA.__name__: "sibling_sel_1777_unknown"},
        )
        accessible: FeatureGroupEnvironmentMapping = {SiblingSelFG: {PyArrowTable}}
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible, None, None)
        assert result.identified == {}
        elimination = result.eliminations.get(SiblingSelFG)
        assert elimination is not None
        assert "SiblingSel565ReaderA" in elimination.reason

    def test_root_returning_a_ignores_the_pin_of_a_sibling(self) -> None:
        feature = Feature(
            name="sibling_sel_1777_feat",
            options={SiblingSel565ReaderA.__name__: _ACCESS_A, SiblingSel565ReaderB.__name__: _ACCESS_B},
        )
        accessible: FeatureGroupEnvironmentMapping = {SiblingSelFG: {PyArrowTable}}
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible, None, None)
        assert SiblingSelFG in result.identified
        assert feature.input_data_match == (SiblingSel565ReaderA, _ACCESS_A)

    def test_later_value_failure_is_reported_not_a_reader_decline(self) -> None:
        feature = Feature(
            name=_VALUE_FEATURE,
            options={SiblingSel565ReaderA.__name__: _ACCESS_A, _VALUE_KEY: _BAD_VALUE},
        )
        accessible: FeatureGroupEnvironmentMapping = {SiblingSel1777ValueFG: {PyArrowTable}}
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible, None, None)
        assert result.identified == {}
        elimination = result.eliminations.get(SiblingSel1777ValueFG)
        assert elimination is not None
        assert elimination.stage == "value_rejection"
        assert f"'{_BAD_VALUE}' not found in mapping for '{_VALUE_KEY}'" in elimination.reason
        assert "matched nothing" not in elimination.reason


class TestSiblingSelectionViaClassKey:
    """Group B: the option key may be the reader class itself."""

    def test_class_as_key_resolves_like_string_form(self) -> None:
        options = Options(cast(dict[str, Any], {SiblingSel565ReaderA: _ACCESS_A}))
        assert SiblingSel565ReaderA.feature_scope_data_access(options, "sibling_sel_565_feat") is True
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

    def test_init_reader_consumes_the_pair(self) -> None:
        reader, data_access = SiblingSel565ReaderA().init_reader((SiblingSel565ReaderA, _ACCESS_A))
        assert isinstance(reader, SiblingSel565ReaderA)
        assert data_access == _ACCESS_A


class TestClassKeyNormalization565:
    """Group E: class option keys must be identity-equivalent to their data_access_name() string form.

    Pins the fix for the class-key/string-key identity split: a reader class used as an
    Options key should be normalized to ``cls.data_access_name()`` when the Options object
    is constructed, so both spellings hash, compare, and look up identically.
    """

    def test_options_class_key_equals_string_key(self) -> None:
        class_keyed = Options(cast(dict[str, Any], {SiblingSel565ReaderA: _ACCESS_A}))
        string_keyed = Options({SiblingSel565ReaderA.__name__: _ACCESS_A})
        assert class_keyed == string_keyed
        assert hash(class_keyed) == hash(string_keyed)

    def test_feature_class_key_equals_string_key(self) -> None:
        class_keyed = Feature(
            name="sibling_sel_565_norm_feat",
            options=cast(dict[str, Any], {SiblingSel565ReaderA: _ACCESS_A}),
        )
        string_keyed = Feature(
            name="sibling_sel_565_norm_feat",
            options={SiblingSel565ReaderA.__name__: _ACCESS_A},
        )
        assert class_keyed == string_keyed
        assert hash(class_keyed) == hash(string_keyed)
        assert len({class_keyed, string_keyed}) == 1

    def test_child_options_with_class_key_hashable(self) -> None:
        feature = Feature(name="sibling_sel_565_child_feat")
        feature.child_options = Options(
            cast(dict[str, Any], {SiblingSel565ReaderA: _ACCESS_A, "sibling_sel_565_child_marker": True})
        )
        assert isinstance(hash(feature), int)

    def test_options_get_by_string_after_class_key_construction(self) -> None:
        options = Options(cast(dict[str, Any], {SiblingSel565ReaderA: _ACCESS_A}))
        assert options.get(SiblingSel565ReaderA.__name__) == _ACCESS_A


_AMBIG_MARKER = "sibling_sel_1757_ambiguous_marker.dat"


def _ambiguous_match(data_access: Any) -> Any:
    if isinstance(data_access, DataAccessCollection):
        return data_access if _AMBIG_MARKER in data_access.files.values() else None
    return data_access if data_access == _AMBIG_MARKER else None


class SiblingSel1757ReaderA(SiblingSelFamily):
    """Final reader accepting only the unique ambiguity marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _ambiguous_match(data_access)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_a": [1]}


class SiblingSel1757ReaderB(SiblingSelFamily):
    """Sibling accepting the same unique ambiguity marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _ambiguous_match(data_access)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_b": [2]}


class TestGlobalScopeMatchesOnlyTheProbedReader1757:
    """Readers accepting one data access never compete: each reader matches only itself."""

    def test_each_reader_resolves_itself_from_the_collection(self) -> None:
        collection = DataAccessCollection(files={_AMBIG_MARKER})
        for reader in (SiblingSel1757ReaderA, SiblingSel1757ReaderB):
            matched = reader.match_data_access(["sibling_sel_1757_feat"], collection, options=Options())
            assert matched == (reader, collection)

    def test_a_reader_matches_through_the_collection_without_naming_a_sibling(self) -> None:
        collection = DataAccessCollection(files={_AMBIG_MARKER})
        options = Options()
        assert SiblingSel1757ReaderB().matches("sibling_sel_1757_feat", options, collection) is True
        assert options.get("BaseInputData") == (SiblingSel1757ReaderB, collection)

    def test_pinned_reader_still_resolves_via_feature_scope(self) -> None:
        collection = DataAccessCollection(files={_AMBIG_MARKER})
        options = Options(group={SiblingSel1757ReaderB.__name__: _AMBIG_MARKER})
        assert SiblingSel1757ReaderB().matches("sibling_sel_1757_feat", options, collection) is True
        assert options.get("BaseInputData") == (SiblingSel1757ReaderB, _AMBIG_MARKER)

    def test_pin_of_one_reader_does_not_scope_the_other(self) -> None:
        collection = DataAccessCollection(files={_AMBIG_MARKER})
        options = Options(group={SiblingSel1757ReaderB.__name__: _AMBIG_MARKER})
        assert SiblingSel1757ReaderA().matches("sibling_sel_1757_feat", options, collection) is True
        assert options.get("BaseInputData") == (SiblingSel1757ReaderA, collection)


_SAME_MARKER = "sibling_sel_1757_same_marker.dat"
_DIFF_MARKER = "sibling_sel_1757_diff_marker.dat"
_ALIAS_MARKER = "sibling_sel_1757_alias_marker.dat"


def _accept(data_access: Any, marker: str, result: Any) -> Any:
    if isinstance(data_access, DataAccessCollection):
        return result if marker in data_access.files.values() else None
    return None


class SiblingSel1757SameParent(SiblingSelFamily):
    """Final parent reader accepting only the unique same-access marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _accept(data_access, _SAME_MARKER, _SAME_MARKER)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_same_parent": [1]}


class SiblingSel1757SameChild(SiblingSel1757SameParent):
    """Final child returning the same access as its parent."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_same_child": [1]}


class SiblingSel1757DiffParent(SiblingSelFamily):
    """Final parent reader accepting only the unique different-access marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _accept(data_access, _DIFF_MARKER, "diff_parent_access")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_diff_parent": [1]}


class SiblingSel1757DiffChild(SiblingSel1757DiffParent):
    """Final child returning a different access than its parent."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _accept(data_access, _DIFF_MARKER, "diff_child_access")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_diff_child": [1]}


class SiblingSel1757AliasPlain(SiblingSelFamily):
    """Final reader accepting only the unique alias marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _accept(data_access, _ALIAS_MARKER, _ALIAS_MARKER)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_alias_plain": [1]}


class SiblingSel1757Aliased(SiblingSelFamily):
    """Final reader overriding data_access_name() with an alias."""

    @classmethod
    def data_access_name(cls) -> str:
        return "sibling_sel_1757_alias_name"

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _accept(data_access, _ALIAS_MARKER, _ALIAS_MARKER)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_aliased": [1]}


class TestRootReaderIgnoresSubclasses1757:
    """A reader matches only itself: a load_data-overriding subclass never replaces it."""

    def test_global_scope_returns_the_parent_not_its_child(self) -> None:
        collection = DataAccessCollection(files={_SAME_MARKER})
        matched = SiblingSel1757SameParent.match_data_access(["sibling_sel_1757_same_feat"], collection, Options())
        assert matched == (SiblingSel1757SameParent, _SAME_MARKER)

    def test_global_scope_returns_the_parent_when_the_child_accepts_a_different_access(self) -> None:
        collection = DataAccessCollection(files={_DIFF_MARKER})
        matched = SiblingSel1757DiffParent.match_data_access(["sibling_sel_1757_diff_feat"], collection, Options())
        assert matched == (SiblingSel1757DiffParent, "diff_parent_access")

    def test_pin_naming_the_child_does_not_select_it_for_the_parent(self) -> None:
        options = Options({SiblingSel1757SameChild.__name__: DataAccessCollection(files={_SAME_MARKER})})
        assert SiblingSel1757SameParent.feature_scope_data_access(options, "sibling_sel_1757_same_feat") is False
        assert "BaseInputData" not in options

    def test_pin_naming_the_parent_does_not_hand_the_feature_to_the_child(self) -> None:
        options = Options({SiblingSel1757SameParent.__name__: DataAccessCollection(files={_SAME_MARKER})})
        assert SiblingSel1757SameParent.feature_scope_data_access(options, "sibling_sel_1757_same_feat") is True
        assert options.get("BaseInputData") == (SiblingSel1757SameParent, _SAME_MARKER)

    def test_aliased_reader_scopes_by_its_alias_and_matches_only_itself(self) -> None:
        collection = DataAccessCollection(files={_ALIAS_MARKER})
        for reader in (SiblingSel1757AliasPlain, SiblingSel1757Aliased):
            matched = reader.match_data_access(["sibling_sel_1757_alias_feat"], collection, Options())
            assert matched == (reader, _ALIAS_MARKER)
        options = Options({"sibling_sel_1757_alias_name": DataAccessCollection(files={_ALIAS_MARKER})})
        assert SiblingSel1757Aliased.feature_scope_data_access(options, "sibling_sel_1757_alias_feat") is True
        assert SiblingSel1757AliasPlain.feature_scope_data_access(options, "sibling_sel_1757_alias_feat") is False


_CSVFG_COL = "sibling_sel_csvfg_col"


def _csv_file(tmp_path: Path) -> str:
    path = tmp_path / "sibling_sel_csvfg.csv"
    path.write_text(f"{_CSVFG_COL}\n1\n2\n")
    return str(path)


class TestClassKeyNormalizationOnCsvFG:
    """Groups B and E on the stock CsvFG: a format group class as an option key equals its class-name string."""

    def test_options_class_key_equals_string_key(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        class_keyed = Options(cast(dict[str, Any], {CsvFG: path}))
        string_keyed = Options({"CsvFG": path})
        assert class_keyed == string_keyed
        assert hash(class_keyed) == hash(string_keyed)
        assert class_keyed.get("CsvFG") == path

    def test_feature_class_key_equals_string_key(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        class_keyed = Feature(name=_CSVFG_COL, options=cast(dict[str, Any], {CsvFG: path}))
        string_keyed = Feature(name=_CSVFG_COL, options={"CsvFG": path})
        assert class_keyed == string_keyed
        assert hash(class_keyed) == hash(string_keyed)
        assert len({class_keyed, string_keyed}) == 1

    def test_class_key_resolves_like_the_string_form(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        feature = Feature(name=_CSVFG_COL, options=cast(dict[str, Any], {CsvFG: path}))

        result = IdentifyFeatureGroupClass.evaluate(feature, {CsvFG: {PyArrowTable}}, None)

        assert CsvFG in result.identified
        assert feature.input_data_match is not None
        assert feature.input_data_match[0] is CsvFG
        assert feature.input_data_match[1].source == os.path.abspath(path)


class TestSecondCsvFormatGroupIsAmbiguous:
    """Two format groups owning the same suffix on the same file are ambiguous; no reader-level pick exists."""

    def test_a_gated_second_csv_group_next_to_csvfg_gives_multiple_feature_groups(self, tmp_path: Path) -> None:
        marker = "sibling_sel_second_csv_marker"

        class SiblingSelSecondCsvFG(ReadFileFG):
            """Owns .csv too; gated by a unique marker option so it never claims in other tests."""

            @classmethod
            def suffixes(cls) -> tuple[str, ...]:
                return (".csv",)

            @classmethod
            def column_names(cls, path: str) -> Collection[str]:
                with open(path, encoding="utf-8") as handle:
                    return handle.readline().strip().split(",")

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: DataAccessCollection | None = None,
            ) -> bool:
                if options.get(marker) is None:
                    return False
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        path = _csv_file(tmp_path)
        feature = Feature(_CSVFG_COL, Options(context={marker: True}))
        plugins: FeatureGroupEnvironmentMapping = {CsvFG: {PyArrowTable}, SiblingSelSecondCsvFG: {PyArrowTable}}

        with pytest.raises(ValueError, match="Multiple feature groups found") as excinfo:
            resolve_or_raise(feature, plugins, None, DataAccessCollection(files={path}))

        assert "CsvFG" in str(excinfo.value)
        assert "SiblingSelSecondCsvFG" in str(excinfo.value)


class TestUrlReaderRecipeEndToEnd:
    """A reader pinned by class-name string or class key runs end to end, ignoring its load_data subclass."""

    @pytest.mark.parametrize("by_class", [False, True])
    def test_url_reader_end_to_end(self, by_class: bool) -> None:
        key: Any = SiblingSel565UrlReader if by_class else SiblingSel565UrlReader.__name__
        feature = Feature(name=_URL_FEATURE_NAME, options={key: _URL_ACCESS, _URL_MARKER_KEY: True})
        features: list[Feature | str] = [feature]
        result = mloda.run_all(
            features,
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({SiblingSelUrlFG}),
        )
        assert result[0].to_pydict()[_URL_FEATURE_NAME] == _URL_FEATURE_VALUES
