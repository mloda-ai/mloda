"""The Engine and mlodaAPI intake lock own-key tracking before feature-group matching.

A matcher write (the linked-reader pattern) must never count as the feature's own declaration,
whether the feature is requested directly or declared as an input feature. mlodaAPI's own
pre-Engine intake stamps (strict_type_enforcement, ApiInputData) and any GlobalFilter twin
merged at engine intake must equally never read as own.
"""

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.request import mlodaAPI
from mloda.core.core.engine import Engine
from mloda.core.filter.global_filter import GlobalFilter

from tests.test_core.test_abstract_plugins.test_abstract_compute_framework import BaseTestComputeFramework1


LINKED_NAME = "ownlock_linked_feature"
CONSUMER_NAME = "ownlock_consumer_feature"
LINKED_KEY = "ownlock_linked_reader"
LINKED_VALUE = "written_by_the_ownlock_matcher"
USER_KEY = "ownlock_user_key"


class OwnLockLinkingFG(FeatureGroup):
    """Root group whose matcher links a reader by writing LINKED_KEY into the options."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {LINKED_NAME}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {BaseTestComputeFramework1}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != LINKED_NAME:
            return False
        if LINKED_KEY not in options:
            options.add_to_group(LINKED_KEY, LINKED_VALUE)
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class OwnLockConsumerFG(FeatureGroup):
    """Consumer group declaring the linked feature as its input."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {CONSUMER_NAME}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {BaseTestComputeFramework1}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CONSUMER_NAME

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(LINKED_NAME)}


def _setup_engine(features: Features) -> Engine:
    return Engine(
        features,
        {BaseTestComputeFramework1},
        None,
        plugin_collector=PluginCollector.enabled_feature_groups({OwnLockLinkingFG, OwnLockConsumerFG}),
    )


def _resolved_linked_options(engine: Engine) -> Options:
    (linked,) = engine.feature_group_collection[OwnLockLinkingFG]
    return linked.options


class TestMatcherWriteIsNeverOwn:
    """A key the matcher writes is present on the resolved feature but never own."""

    def test_directly_requested_linked_feature(self) -> None:
        options = _resolved_linked_options(_setup_engine(Features([Feature(LINKED_NAME)])))

        assert options.group.get(LINKED_KEY) == LINKED_VALUE
        assert LINKED_KEY not in options.own_group_keys, (
            f"a matcher write must not count as own on a directly requested feature, got {options.own_group_keys}"
        )

    def test_linked_feature_requested_as_input(self) -> None:
        options = _resolved_linked_options(_setup_engine(Features([Feature(CONSUMER_NAME)])))

        assert options.group.get(LINKED_KEY) == LINKED_VALUE
        assert LINKED_KEY not in options.own_group_keys


class TestUserDeclarationStaysOwn:
    """Locking before matching must not demote what the user declared."""

    def test_directly_requested_user_key_stays_own(self) -> None:
        requested = Feature(LINKED_NAME, options=Options(context={USER_KEY: 1}))

        options = _resolved_linked_options(_setup_engine(Features([requested])))

        assert options.is_own(USER_KEY) is True


def _assert_no_framework_own_keys(options: Options) -> None:
    for key in (DefaultOptionKeys.strict_type_enforcement, "ApiInputData", LINKED_KEY):
        if key in options.group:
            assert key not in options.own_group_keys, (
                f"framework key {key!r} must not count as own, got {options.own_group_keys}"
            )


class TestApiIntakeStampsAreNeverOwn:
    """mlodaAPI._process_features stamps framework keys before the Engine exists; they must not read as own."""

    def test_intake_stamps_stay_unowned_user_key_stays_own(self) -> None:
        requested = Feature(LINKED_NAME, options=Options(context={USER_KEY: 1}), data_type=DataType.INT32)

        api = mlodaAPI(
            [requested],
            compute_frameworks={BaseTestComputeFramework1},
            plugin_collector=PluginCollector.enabled_feature_groups({OwnLockLinkingFG, OwnLockConsumerFG}),
            strict_type_enforcement=True,
            api_data={"OwnLockApiKey": {"ownlock_api_column": [1]}},
        )

        assert api.engine is not None
        options = _resolved_linked_options(api.engine)

        assert DefaultOptionKeys.strict_type_enforcement in options.group
        assert DefaultOptionKeys.strict_type_enforcement not in options.own_group_keys, (
            f"an mlodaAPI-stamped framework key must not count as own, got {options.own_group_keys}"
        )
        assert "ApiInputData" in options.group
        assert "ApiInputData" not in options.own_group_keys

        assert options.is_own(USER_KEY) is True

    def test_global_filter_twin_stamps_stay_unowned_user_key_stays_own(self) -> None:
        requested = Feature(LINKED_NAME, options=Options(context={USER_KEY: 1}), data_type=DataType.INT32)

        gf = GlobalFilter()
        gf.add_filter(LINKED_NAME, "equal", {"value": 1})

        api = mlodaAPI(
            [requested],
            compute_frameworks={BaseTestComputeFramework1},
            plugin_collector=PluginCollector.enabled_feature_groups({OwnLockLinkingFG, OwnLockConsumerFG}),
            strict_type_enforcement=True,
            api_data={"OwnLockApiKey": {"ownlock_api_column": [1]}},
            global_filter=gf,
        )

        assert api.engine is not None
        resolved = api.engine.feature_group_collection[OwnLockLinkingFG]

        requested_seen = False
        for feature in resolved:
            _assert_no_framework_own_keys(feature.options)
            if feature.initial_requested_data:
                requested_seen = True
                assert feature.options.is_own(USER_KEY) is True

        assert requested_seen, "expected the originally requested feature among the resolved features"
