"""The Engine locks own-key tracking before feature-group matching.

A matcher write (the linked-reader pattern) must never count as the feature's own declaration,
whether the feature is requested directly or declared as an input feature. Names carry an ``ownlock`` tag.
"""

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.engine import Engine

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
