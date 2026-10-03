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
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.components.utils import is_match_abort
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import FeatureGroup, FeatureSet, PropertySpec
from mloda.user import Feature, FeatureName
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


_REPL_ALIAS = "sibling_sel_1777_alias"
_REPL_MARKER = "sibling_sel_1777_repl_marker"
_TWIN_ALIAS = "sibling_sel_1777_twin_alias"
_TWIN_MARKER = "sibling_sel_1777_twin_marker"


class SiblingSel1777ReplParent(ReadFile):
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


class SiblingSel1777TwinA(ReadFile):
    """Final reader sharing the twin alias."""

    @classmethod
    def data_access_name(cls) -> str:
        return _TWIN_ALIAS

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return data_access if data_access == _TWIN_MARKER else None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1777_twin_a": [1]}


class SiblingSel1777TwinB(ReadFile):
    """Unrelated final reader sharing the twin alias."""

    @classmethod
    def data_access_name(cls) -> str:
        return _TWIN_ALIAS

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return data_access if data_access == _TWIN_MARKER else None

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1777_twin_b": [2]}


_VALUE_KEY = "sibling_sel_1777_mode"
_BAD_VALUE = "sibling_sel_1777_bad_mode"
_VALUE_FEATURE = "sibling_sel_1777_value_feat"


class SiblingSel1777ValueFG(FeatureGroup):
    """Root group fronting ReadFile; the required strict mapped key keeps it from matching other tests' features."""

    PROPERTY_MAPPING = {
        _VALUE_KEY: PropertySpec("Mode", allowed_values={"good": "Good mode"}, context=True, strict_validation=True),
    }

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return ReadFile()

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_VALUE_FEATURE}


class TestPinnedReaderConflict1777:
    """Two accepting pins of one family raise without leaking access values; declining pins are skipped."""

    def test_final_subclass_replaces_parent_sharing_the_pinned_key(self) -> None:
        options = Options({_REPL_ALIAS: _REPL_MARKER})
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_1777_repl_feat") is True
        assert options.get("BaseInputData") == (SiblingSel1777ReplChild, _REPL_MARKER)

    def test_unrelated_readers_sharing_an_alias_are_named_by_qualified_name(self) -> None:
        options = Options({_TWIN_ALIAS: _TWIN_MARKER})
        with pytest.raises(ValueError) as excinfo:
            BaseInputData.feature_scope_data_access(options, "sibling_sel_1777_twin_feat")
        message = str(excinfo.value)
        for reader in (SiblingSel1777TwinA, SiblingSel1777TwinB):
            assert f"{reader.__module__}.{reader.__qualname__}" in message
        assert not is_match_abort(excinfo.value)

    def test_all_declining_pins_eliminate_deterministically_naming_first_pin(self) -> None:
        feature = Feature(
            name="sibling_sel_1777_decline_feat",
            options={
                SiblingSel565ReaderA.__name__: "sibling_sel_1777_unknown",
                SiblingSel565ReaderB.__name__: "sibling_sel_1777_unknown",
            },
        )
        accessible: FeatureGroupEnvironmentMapping = {ReadFileFeature: {PyArrowTable}}
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible, None, None)
        assert result.identified == {}
        elimination = result.eliminations.get(ReadFileFeature)
        assert elimination is not None
        assert "SiblingSel565ReaderA" in elimination.reason

    def test_two_accepting_pins_raise_naming_both_without_writing(self) -> None:
        group = {SiblingSel565ReaderA.__name__: _ACCESS_A, SiblingSel565ReaderB.__name__: _ACCESS_B}
        options = Options(group=dict(group))
        with pytest.raises(ValueError) as excinfo:
            BaseInputData.feature_scope_data_access(options, "sibling_sel_1777_feat")
        message = str(excinfo.value)
        assert "SiblingSel565ReaderA" in message
        assert "SiblingSel565ReaderB" in message
        assert message.index("SiblingSel565ReaderA") < message.index("SiblingSel565ReaderB")
        assert "pin each reader on its own feature" in message
        assert _ACCESS_A not in message
        assert _ACCESS_B not in message
        assert not is_match_abort(excinfo.value)
        assert "BaseInputData" not in options
        assert options.group == group

    @pytest.mark.parametrize(
        "access_a,access_b,winner,access",
        [
            (_ACCESS_A, _ACCESS_A, SiblingSel565ReaderA, _ACCESS_A),
            (_ACCESS_B, _ACCESS_B, SiblingSel565ReaderB, _ACCESS_B),
        ],
        ids=["b_declines", "a_declines"],
    )
    def test_declining_pin_is_skipped_and_accepting_pin_wins(
        self, access_a: str, access_b: str, winner: type, access: str, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options({SiblingSel565ReaderA.__name__: access_a, SiblingSel565ReaderB.__name__: access_b})
        assert BaseInputData.feature_scope_data_access(options, "sibling_sel_1777_feat") is True
        assert options.get("BaseInputData") == (winner, access)
        assert set(rejection_window) == set()

    @pytest.mark.parametrize(
        "access_a,access_b", [(_ACCESS_A, _ACCESS_A), (_ACCESS_B, _ACCESS_B)], ids=["b_declines", "a_declines"]
    )
    def test_later_value_failure_is_reported_not_the_losing_pin(self, access_a: str, access_b: str) -> None:
        feature = Feature(
            name=_VALUE_FEATURE,
            options={
                SiblingSel565ReaderA.__name__: access_a,
                SiblingSel565ReaderB.__name__: access_b,
                _VALUE_KEY: _BAD_VALUE,
            },
        )
        accessible: FeatureGroupEnvironmentMapping = {SiblingSel1777ValueFG: {PyArrowTable}}
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible, None, None)
        assert result.identified == {}
        elimination = result.eliminations.get(SiblingSel1777ValueFG)
        assert elimination is not None
        assert elimination.stage == "value_rejection"
        assert f"'{_BAD_VALUE}' not found in mapping for '{_VALUE_KEY}'" in elimination.reason
        assert "matched nothing" not in elimination.reason

    def test_resolution_contains_conflict_as_matcher_error(self) -> None:
        feature = Feature(
            name="sibling_sel_1777_feat",
            options={SiblingSel565ReaderA.__name__: _ACCESS_A, SiblingSel565ReaderB.__name__: _ACCESS_B},
        )
        accessible: FeatureGroupEnvironmentMapping = {ReadFileFeature: {PyArrowTable}}
        result = IdentifyFeatureGroupClass.evaluate(feature, accessible, None, None)
        assert result.identified == {}
        elimination = result.eliminations.get(ReadFileFeature)
        assert elimination is not None
        assert elimination.stage == "matcher_error"
        assert "SiblingSel565ReaderA" in elimination.reason
        assert "SiblingSel565ReaderB" in elimination.reason


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


class SiblingSel1757ReaderA(ReadFile):
    """Final reader accepting only the unique ambiguity marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _ambiguous_match(data_access)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_a": [1]}


class SiblingSel1757ReaderB(ReadFile):
    """Sibling accepting the same unique ambiguity marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _ambiguous_match(data_access)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_b": [2]}


class TestAmbiguousReaderMatch1757:
    """Group F: several readers accepting one data access is an unmarked ValueError naming the pin remedy."""

    def test_two_acceptors_raise_naming_both_and_pin_hint(self) -> None:
        collection = DataAccessCollection(files={_AMBIG_MARKER})
        with pytest.raises(ValueError) as excinfo:
            ReadFile.match_data_access(["sibling_sel_1757_feat"], collection, options=Options())
        message = str(excinfo.value)
        assert "SiblingSel1757ReaderA" in message
        assert "SiblingSel1757ReaderB" in message
        assert message.index("SiblingSel1757ReaderA") < message.index("SiblingSel1757ReaderB")
        assert "options=" in message
        assert _AMBIG_MARKER not in message
        assert not is_match_abort(excinfo.value)

    def test_pinned_reader_still_resolves_via_feature_scope(self) -> None:
        collection = DataAccessCollection(files={_AMBIG_MARKER})
        options = Options(group={SiblingSel1757ReaderB.__name__: _AMBIG_MARKER})
        assert ReadFile().matches("sibling_sel_1757_feat", options, collection) is True
        assert options.get("BaseInputData") == (SiblingSel1757ReaderB, _AMBIG_MARKER)


_SAME_MARKER = "sibling_sel_1757_same_marker.dat"
_DIFF_MARKER = "sibling_sel_1757_diff_marker.dat"
_ALIAS_MARKER = "sibling_sel_1757_alias_marker.dat"


def _accept(data_access: Any, marker: str, result: Any) -> Any:
    if isinstance(data_access, DataAccessCollection):
        return result if marker in data_access.files.values() else None
    return None


class SiblingSel1757SameParent(ReadFile):
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


class SiblingSel1757DiffParent(ReadFile):
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


class SiblingSel1757AliasPlain(ReadFile):
    """Final reader accepting only the unique alias marker."""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        return _accept(data_access, _ALIAS_MARKER, _ALIAS_MARKER)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return {"sibling_sel_1757_alias_plain": [1]}


class SiblingSel1757Aliased(ReadFile):
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


class TestSubclassPreferenceAndAlias1757:
    """A subclass replaces its ancestor for the same access; aliases drive names and the pin hint."""

    def test_child_wins_over_parent_for_the_same_access(self) -> None:
        collection = DataAccessCollection(files={_SAME_MARKER})
        matched = ReadFile.match_data_access(["sibling_sel_1757_same_feat"], collection, options=Options())
        assert matched == (SiblingSel1757SameChild, _SAME_MARKER)

    def test_child_and_parent_with_different_accesses_raise_naming_both(self) -> None:
        collection = DataAccessCollection(files={_DIFF_MARKER})
        with pytest.raises(ValueError) as excinfo:
            ReadFile.match_data_access(["sibling_sel_1757_diff_feat"], collection, options=Options())
        message = str(excinfo.value)
        assert "SiblingSel1757DiffParent" in message
        assert "SiblingSel1757DiffChild" in message

    def test_aliased_sibling_is_named_and_hinted_by_its_alias(self) -> None:
        collection = DataAccessCollection(files={_ALIAS_MARKER})
        with pytest.raises(ValueError) as excinfo:
            ReadFile.match_data_access(["sibling_sel_1757_alias_feat"], collection, options=Options())
        message = str(excinfo.value)
        assert "sibling_sel_1757_alias_name" in message
        assert "SiblingSel1757Aliased" not in message
        assert "options={'sibling_sel_1757_alias_name'" in message
