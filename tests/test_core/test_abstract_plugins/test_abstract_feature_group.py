import gc
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import pytest
from mloda.user import DataAccessCollection
from mloda.user import DataType
from mloda.user import FeatureName
from mloda.provider import FeatureGroup
from mloda.user import PluginLoader
from mloda.user import Feature
from mloda.user import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.property_spec import property_spec
from mloda.core.abstract_plugins.components.utils import get_all_subclasses, safe_field_with_error
from mloda.core.abstract_plugins.feature_group import feature_group_subclasses
from mloda.core.api.plugin_docs import get_feature_group_docs
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.interfaces.service.catalog import plugin_collector_for
from mloda.interfaces.service.config import Settings
from mloda.interfaces.service.request import caller_refused_option_keys
from tests.helpers.subclass_walks import count_root_walks


class BaseTestFeatureGroup1(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if "BaseTestFeature" in str(feature_name) and "1" in str(feature_name):
            return True
        return False


class BaseTestFeatureGroup2(FeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if "BaseTestFeature" in str(feature_name) and "2" in str(feature_name):
            return True
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """This function should return the input features for the feature group
        if this feature is dependent on other features.
        Else, return None"""
        return {Feature.str_of("BaseTestFeature1")}

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        if "BaseTestFeature" in feature.name and "2" in feature.name:
            return DataType.STRING
        return None


def test_apply_naming_convention_basic_multi_column() -> None:
    """Test that apply_naming_convention handles basic multi-column arrays correctly.

    Given a 2D numpy array with 3 columns and a feature name,
    the method should return a dictionary mapping column names to column data
    using the pattern: feature_name~0, feature_name~1, feature_name~2
    """
    # Arrange: Create a 2D array with 3 columns (simulating sklearn encoder output)
    result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    feature_name = "test_feature"

    # Act: Apply naming convention
    output = FeatureGroup.apply_naming_convention(result, feature_name)

    # Assert: Verify the output structure
    assert isinstance(output, dict), "Output should be a dictionary"
    assert len(output) == 3, "Should have 3 entries for 3 columns"

    # Verify column names follow the pattern
    assert "test_feature~0" in output
    assert "test_feature~1" in output
    assert "test_feature~2" in output

    # Verify column data is correct
    np.testing.assert_array_equal(output["test_feature~0"], np.array([1.0, 4.0, 7.0]))
    np.testing.assert_array_equal(output["test_feature~1"], np.array([2.0, 5.0, 8.0]))
    np.testing.assert_array_equal(output["test_feature~2"], np.array([3.0, 6.0, 9.0]))


@pytest.mark.parametrize(
    "column_name,expected_base_feature",
    [
        ("category~1", "category"),  # Basic case: strip suffix
        ("plain_name", "plain_name"),  # Edge case: no suffix
        ("feature~0", "feature"),  # Zero-indexed suffix
        ("complex_name~99", "complex_name"),  # Multi-digit suffix
        ("x__op~2__next~5", "x__op~2__next"),  # only the LAST ~digits suffix is stripped
        ("x__op~a__next~b", "x__op~a__next~b"),  # no trailing digit suffix -> unchanged (issue repro)
        ("a~1~2", "a~1"),  # embedded ~1 kept, only trailing ~2 stripped
        ("~5", "~5"),  # suffix with empty base -> unchanged
        ("a~", "a~"),  # trailing separator with empty (non-digit) suffix -> unchanged
        ("col_a__pca_2d~dim1", "col_a__pca_2d~dim1"),  # non-digit custom suffix -> unchanged
        ("", ""),  # empty string -> unchanged
    ],
)
def test_get_column_base_feature_strips_suffix(column_name: str, expected_base_feature: str) -> None:
    """Test that get_column_base_feature strips only a trailing all-digit ~N suffix, matched from the last separator."""
    # Act: Extract base feature name
    result = FeatureGroup.get_column_base_feature(column_name)

    # Assert: Verify correct base feature name
    assert result == expected_base_feature, f"Expected {expected_base_feature}, got {result}"


def test_expand_feature_columns_generates_list() -> None:
    """Test that expand_feature_columns generates list of column names with suffixes.

    Given a feature name and the number of columns to generate,
    the method should return a list of column names following the pattern:
    ["feature~0", "feature~1", "feature~2", ...]
    """
    # Arrange: Feature name and number of columns
    feature_name = "category"
    num_columns = 3

    # Act: Generate column names
    result = FeatureGroup.expand_feature_columns(feature_name, num_columns)

    # Assert: Verify the output structure and content
    assert isinstance(result, list), "Output should be a list"
    assert len(result) == 3, "Should have 3 column names"
    assert result == ["category~0", "category~1", "category~2"], "Column names should follow the ~N suffix pattern"


def test_apply_naming_convention_with_custom_suffix_generator() -> None:
    """Test that apply_naming_convention uses custom suffix_generator when provided.

    When a suffix_generator callable is provided, it should be used instead of the
    default ~0, ~1, ~2 pattern. This is useful for custom naming schemes like
    dimensionality reduction where suffixes like ~dim1, ~dim2, ~dim3 are preferred.

    The suffix_generator takes a column index (int) and returns a suffix string.
    """
    # Arrange: Create a 2D array with 3 columns
    result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    feature_name = "feature"

    # Custom suffix generator: produces "dim1", "dim2", "dim3" instead of "0", "1", "2"
    def custom_suffix_generator(i: int) -> str:
        return f"dim{i + 1}"

    # Act: Apply naming convention with custom suffix generator
    output = FeatureGroup.apply_naming_convention(result, feature_name, suffix_generator=custom_suffix_generator)

    # Assert: Verify the output uses custom suffixes
    assert isinstance(output, dict), "Output should be a dictionary"
    assert len(output) == 3, "Should have 3 entries for 3 columns"

    # Verify column names use custom suffix pattern
    assert "feature~dim1" in output, "Should use custom suffix 'dim1'"
    assert "feature~dim2" in output, "Should use custom suffix 'dim2'"
    assert "feature~dim3" in output, "Should use custom suffix 'dim3'"

    # Verify column data is correct
    np.testing.assert_array_equal(output["feature~dim1"], np.array([1.0, 4.0, 7.0]))
    np.testing.assert_array_equal(output["feature~dim2"], np.array([2.0, 5.0, 8.0]))
    np.testing.assert_array_equal(output["feature~dim3"], np.array([3.0, 6.0, 9.0]))


def test_resolve_multi_column_feature_discovers_tilde_columns() -> None:
    """Test that resolve_multi_column_feature discovers multi-column features with ~ pattern.

    Given a feature name and a set of available columns that match the multi-column pattern,
    the method should return all matching columns sorted by their suffix index.

    For example, if "my_feature~0", "my_feature~1", "my_feature~2" exist in the columns,
    resolve_multi_column_feature("my_feature", columns) should return all three columns.
    """
    # Arrange: Set of available columns containing multi-column feature
    feature_name = "my_feature"
    available_columns = {"my_feature~0", "my_feature~1", "my_feature~2"}

    # Act: Resolve multi-column feature
    result = FeatureGroup.resolve_multi_column_feature(feature_name, available_columns)

    # Assert: Should return all matching columns sorted
    assert isinstance(result, list), "Output should be a list"
    assert len(result) == 3, "Should find all 3 columns"
    assert result == ["my_feature~0", "my_feature~1", "my_feature~2"], "Should return columns in sorted order"


_DI_KEY = "declared_inputs_probe_key"


class _DeclaredInputsProbeBase(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__([\w]+)_declared_inputs_probe$"
    PROPERTY_MAPPING = {
        _DI_KEY: property_spec("Probe op.", strict=True, allowed_values={"sum": "Sum"}, context=True),
    }

    @classmethod
    @abstractmethod
    def declared_inputs_probe_hook(cls) -> str: ...


class _DeclaredInputsProbeConcrete(_DeclaredInputsProbeBase):
    @classmethod
    def declared_inputs_probe_hook(cls) -> str:
        return "x"


class _DeclaredInputsProbeInstanceOverride(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.str_of("declared_inputs_probe_src")}


class _DeclaredInputsProbeRoot(FeatureGroup):
    pass


_DI_NAME = FeatureName("declared_inputs_probe_src__sum_declared_inputs_probe")


def _names(features: set[Feature] | None) -> set[str]:
    assert features is not None
    return {f.name for f in features}


def test_declared_input_features_on_abstract_base_matches_concrete() -> None:
    expected = _names(_DeclaredInputsProbeConcrete().input_features(Options(), _DI_NAME))
    assert expected == {"declared_inputs_probe_src"}
    assert _names(_DeclaredInputsProbeBase.declared_input_features(Options(), _DI_NAME)) == expected


def test_declared_input_features_with_instance_method_override() -> None:
    result = _DeclaredInputsProbeInstanceOverride.declared_input_features(
        Options(), FeatureName("declared_inputs_probe_x")
    )
    assert _names(result) == {"declared_inputs_probe_src"}


def test_declared_input_features_leaves_base_untouched() -> None:
    subclasses_before = list(FeatureGroup.__subclasses__())
    base_subclasses_before = list(_DeclaredInputsProbeBase.__subclasses__())
    abstract_before = _DeclaredInputsProbeBase.__abstractmethods__

    _DeclaredInputsProbeBase.declared_input_features(Options(), _DI_NAME)

    assert list(FeatureGroup.__subclasses__()) == subclasses_before
    assert list(_DeclaredInputsProbeBase.__subclasses__()) == base_subclasses_before
    assert _DeclaredInputsProbeBase.__abstractmethods__ == abstract_before
    with pytest.raises(TypeError):
        _DeclaredInputsProbeBase()  # type: ignore[abstract]


def test_declared_input_features_propagates_not_implemented_for_root_group() -> None:
    with pytest.raises(NotImplementedError):
        _DeclaredInputsProbeRoot.declared_input_features(Options(), FeatureName("declared_inputs_probe_root"))


_DI_T = TypeVar("_DI_T")
_DI_FIXED = {"declared_inputs_probe_fixed"}


def _di_fixed_inputs(self: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
    return {Feature.str_of(n) for n in _DI_FIXED}


class _DeclaredInputsProbeAbcMixin(ABC):
    @abstractmethod
    def m(self) -> None: ...


class _DeclaredInputsProbeAbcGroup(_DeclaredInputsProbeAbcMixin, FeatureGroup):
    input_features = _di_fixed_inputs


class _DeclaredInputsProbeGenericGroup(FeatureGroup, Generic[_DI_T]):
    input_features = _di_fixed_inputs


class _DeclaredInputsProbeRecording:
    seen: ClassVar[list[str]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _DeclaredInputsProbeRecording.seen.append(cls.__name__)


class _DeclaredInputsProbeRecordingGroup(_DeclaredInputsProbeRecording, FeatureGroup):
    input_features = _di_fixed_inputs

    @classmethod
    @abstractmethod
    def declared_inputs_probe_rec_hook(cls) -> str: ...


@pytest.mark.parametrize(
    "group",
    [_DeclaredInputsProbeAbcGroup, _DeclaredInputsProbeGenericGroup, _DeclaredInputsProbeRecordingGroup],
    ids=["abc_mixin_with_abstractmethod", "generic", "recording_mixin"],
)
def test_declared_input_features_on_special_abstract_bases(group: Any) -> None:
    result = group.declared_input_features(Options(), FeatureName("declared_inputs_probe_x"))
    assert _names(result) == _DI_FIXED


def test_declared_input_features_does_not_trigger_mixin_init_subclass() -> None:
    before = list(_DeclaredInputsProbeRecording.seen)
    _DeclaredInputsProbeRecordingGroup.declared_input_features(Options(), FeatureName("declared_inputs_probe_x"))
    assert _DeclaredInputsProbeRecording.seen == before


def test_declared_input_features_does_not_grow_mixin_subclasses() -> None:
    name = FeatureName("declared_inputs_probe_x")
    _DeclaredInputsProbeRecordingGroup.declared_input_features(Options(), name)
    after_first = len(_DeclaredInputsProbeRecording.__subclasses__())
    for _ in range(50):
        _DeclaredInputsProbeRecordingGroup.declared_input_features(Options(), name)
    assert len(_DeclaredInputsProbeRecording.__subclasses__()) == after_first


class TestFeatureGroupSubclassRegistry:
    """`feature_group_subclasses()` mirrors the live FeatureGroup tree without walking it."""

    def test_registry_equals_the_walk(self) -> None:
        PluginLoader.all()
        registry = feature_group_subclasses()
        if registry != get_all_subclasses(FeatureGroup):
            gc.collect()
            registry = feature_group_subclasses()

        assert registry == get_all_subclasses(FeatureGroup)
        assert len(registry) > 0
        assert FeatureGroup not in registry

    def test_registry_tracks_a_transient_subclass(self) -> None:
        class _CooperativeMixin:
            def __init_subclass__(cls, **kwargs: Any) -> None:
                super().__init_subclass__(**kwargs)

        class ZzTransientRegistryGroup(_CooperativeMixin, FeatureGroup):
            pass

        present = ZzTransientRegistryGroup in feature_group_subclasses()
        name = ZzTransientRegistryGroup.__name__
        ref = weakref.ref(ZzTransientRegistryGroup)
        del ZzTransientRegistryGroup
        for generation in (0, 1, 2):
            if ref() is None:
                break
            gc.collect(generation)
        collected = ref() is None
        names = {fg.__name__ for fg in feature_group_subclasses()}

        assert present
        assert collected
        assert name not in names

    def test_definition_rejected_by_class_validation_is_never_registered(self) -> None:
        def make() -> type[FeatureGroup]:
            class ZzRejectedRegistryGroup(FeatureGroup):
                PROPERTY_MAPPING = {"key": "not a PropertySpec"}  # type: ignore[dict-item]

            return ZzRejectedRegistryGroup

        _, message = safe_field_with_error(make, FeatureGroup)
        walk_names = {fg.__name__ for fg in get_all_subclasses(FeatureGroup)}
        registry_names = {fg.__name__ for fg in feature_group_subclasses()}
        gc.collect()

        assert message is not None
        assert "ZzRejectedRegistryGroup" in walk_names
        assert "ZzRejectedRegistryGroup" not in registry_names

    def test_group_skipping_registration_is_invisible_to_registry_readers(self) -> None:
        class _SkipsRegistration:
            def __init_subclass__(cls, **kwargs: Any) -> None:
                pass

        def make() -> type[FeatureGroup]:
            class ZzUnregisteredRegistryGroup(_SkipsRegistration, FeatureGroup):
                pass

            return ZzUnregisteredRegistryGroup

        held = [make()]
        name = held[0].__name__
        ref = weakref.ref(held[0])
        walk_sees = held[0] in get_all_subclasses(FeatureGroup)
        registry_sees = held[0] in feature_group_subclasses()
        prefilter_sees = held[0] in PreFilterPlugins.get_featuregroup_subclasses()
        docs_names = {info.name for info in get_feature_group_docs(name=name)}

        del held[:]
        for generation in (0, 1, 2):
            if ref() is None:
                break
            gc.collect(generation)
        collected = ref() is None

        assert walk_sees
        assert not registry_sees
        assert not prefilter_sees
        assert name not in docs_names
        assert collected

    @pytest.mark.parametrize(
        "reader",
        [
            lambda: PreFilterPlugins.get_featuregroup_subclasses(),
            lambda: plugin_collector_for(Settings()),
            lambda: caller_refused_option_keys(),
            lambda: get_feature_group_docs(name="zz_no_such_feature_group"),
        ],
        ids=["prefilter_featuregroup_subclasses", "plugin_collector_for", "caller_refused_option_keys", "docs"],
    )
    def test_readers_of_all_feature_groups_never_walk_the_tree(
        self, monkeypatch: pytest.MonkeyPatch, reader: Callable[[], Any]
    ) -> None:
        root_walks = count_root_walks(monkeypatch, FeatureGroup)
        reader()
        monkeypatch.undo()

        assert root_walks == []
