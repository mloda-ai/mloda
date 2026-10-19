"""
Integration test: ApiInputDataCollection with Links and Joins.

Three features:
1. ApiInputDataFeature (existing) - receives mloda data
2. CreatorDataFeature (custom) - creates own data via DataCreator
3. JoinedFeature (custom) - depends on both with a join
"""

import logging
from typing import Any

import pytest
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Link, JoinSpec
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.provider import ApiInputDataFeature


# ============================================================================
# Feature B: Data Creator Feature
# ============================================================================
class CreatorDataFeature(FeatureGroup):
    """Creates its own data via DataCreator."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"creator_id", "creator_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"creator_id": [1, 2, 3], "creator_value": ["a", "b", "c"]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"creator_id", "creator_value"}


# ============================================================================
# Feature C: Joined Feature (depends on mloda and Creator)
# ============================================================================
class LeftJoinedFeature(FeatureGroup):
    """Joins mloda data with Creator data using LEFT join."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Define input features with a LEFT join link."""

        # Create the link: LEFT join on api_id = creator_id
        link = Link.left(
            JoinSpec(ApiInputDataFeature, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",)))
        )

        # Return features - attach link to one of them
        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Combine the joined data - output column must match feature name
        data["LeftJoinedFeature"] = data["api_value"].fillna("") + "_" + data["creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class AppendedFeature(FeatureGroup):
    """Appends mloda data with Creator data (stacks vertically)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Define input features with an APPEND link."""

        # Create the link: APPEND stacks data vertically
        link = Link.append(
            JoinSpec(ApiInputDataFeature, Index(("api_id",))), JoinSpec(CreatorDataFeature, Index(("creator_id",)))
        )

        # Return features - attach link to one of them
        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Combine the appended data - output column must match feature name
        data["AppendedFeature"] = data["api_value"].fillna("") + "_" + data["creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


# ============================================================================
# Test Class
# ============================================================================
class TestApiLinkJoin:
    """Integration tests for ApiInputDataCollection with Links."""

    _enabled_left = PluginCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            CreatorDataFeature,
            LeftJoinedFeature,
        }
    )

    _enabled_append = PluginCollector.enabled_feature_groups(
        {
            ApiInputDataFeature,
            CreatorDataFeature,
            AppendedFeature,
        }
    )

    def test_api_creator_left_join(self) -> None:
        """
        Test LEFT join between mloda data and DataCreator data.

        mloda data (4 rows): ids 1, 2, 3, 4
        Creator data (3 rows): ids 1, 2, 3
        Result: 4 rows (LEFT keeps all mloda rows, id 4 has null for creator_value)
        """
        # Request the joined feature
        feature_list: list[Feature | str] = [Feature(name="LeftJoinedFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_left,
            compute_frameworks=[PandasDataFrame],
            api_data={"ApiExample": {"api_id": [1, 2, 3, 4], "api_value": ["w", "x", "y", "z"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert "LeftJoinedFeature" in df.columns
        assert len(df) == 4  # LEFT join keeps all mloda rows

    def test_api_creator_append(self) -> None:
        """
        Test APPEND between mloda data and DataCreator data.

        mloda data (2 rows)
        Creator data (3 rows)
        Result: 5 rows (stacked vertically)
        """
        # Request the appended feature
        feature_list: list[Feature | str] = [Feature(name="AppendedFeature")]

        result = mloda.run_all(
            feature_list,
            plugin_collector=self._enabled_append,
            compute_frameworks=[PandasDataFrame],
            api_data={"ApiExample": {"api_id": [1, 2], "api_value": ["x", "y"]}},
        )

        assert len(result) == 1
        df = result[0]
        assert "AppendedFeature" in df.columns
        assert len(df) == 5  # APPEND stacks: 2 mloda + 3 Creator rows


# ============================================================================
# Planning warning for a declared link that plans no join
# ============================================================================
class UnjoinedCustomers(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ujc_id", "ujc_name"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ujc_id": [1, 2], "ujc_name": ["a", "b"]}


class UnjoinedOrders(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ujo_cust_id", "ujo_amount"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ujo_cust_id": [2, 1], "ujo_amount": [20, 10]}


class UnjoinedNameAmount(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ujc_name"), Feature("ujo_amount")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data["ujc_name"] + data["ujo_amount"].astype(str)}


class UnjoinedSelf(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ujs_id", "ujs_a", "ujs_b"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ujs_id": [1], "ujs_a": [1], "ujs_b": [2]}


class UnjoinedSubCustomers(UnjoinedCustomers):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ujsc_id", "ujsc_name"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ujsc_id": [1, 2], "ujsc_name": ["a", "b"]}


class UnjoinedSubOrders(UnjoinedOrders):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"ujso_cust_id", "ujso_amount"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ujso_cust_id": [2, 1], "ujso_amount": [20, 10]}


class UnjoinedSubNameAmount(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("ujsc_name"), Feature("ujso_amount")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data["ujsc_name"] + data["ujso_amount"].astype(str)}


_UNJOINED_LINK = Link.inner(JoinSpec(UnjoinedCustomers, "ujc_id"), JoinSpec(UnjoinedOrders, "ujo_cust_id"))
_UNJOINED_GROUPS: set[type[FeatureGroup]] = {UnjoinedCustomers, UnjoinedOrders, UnjoinedNameAmount}


def _plan_warnings(
    caplog: pytest.LogCaptureFixture,
    features: list[Feature | str],
    links: set[Link],
    groups: set[type[FeatureGroup]],
) -> list[str]:
    with caplog.at_level(logging.WARNING):
        mloda.prepare(
            features,
            compute_frameworks=[PandasDataFrame],
            links=links,
            plugin_collector=PluginCollector.enabled_feature_groups(groups),
        )
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _assert_unjoined_link_warning(messages: list[str]) -> None:
    assert len(messages) == 1
    assert "UnjoinedCustomers" in messages[0]
    assert "UnjoinedOrders" in messages[0]
    assert "input_features" in messages[0]


def test_unjoined_link_between_requested_groups_warns(caplog: pytest.LogCaptureFixture) -> None:
    messages = _plan_warnings(caplog, ["ujc_name", "ujo_amount"], {_UNJOINED_LINK}, _UNJOINED_GROUPS)
    _assert_unjoined_link_warning(messages)


def test_unjoined_link_pinned_on_a_feature_warns(caplog: pytest.LogCaptureFixture) -> None:
    messages = _plan_warnings(
        caplog,
        [Feature("ujc_name"), Feature("ujo_amount", link=_UNJOINED_LINK)],
        set(),
        _UNJOINED_GROUPS,
    )
    _assert_unjoined_link_warning(messages)


def test_link_joined_by_a_consumer_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    messages = _plan_warnings(caplog, ["UnjoinedNameAmount"], {_UNJOINED_LINK}, _UNJOINED_GROUPS)
    assert messages == []


def test_link_with_one_side_in_the_plan_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    messages = _plan_warnings(caplog, ["ujc_name"], {_UNJOINED_LINK}, _UNJOINED_GROUPS)
    assert messages == []


def test_self_link_without_discriminators_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    link = Link.inner(JoinSpec(UnjoinedSelf, "ujs_id"), JoinSpec(UnjoinedSelf, "ujs_id"))
    messages = _plan_warnings(caplog, ["ujs_a", "ujs_b"], {link}, {UnjoinedSelf})
    assert messages == []


def test_base_link_shadowed_by_a_joined_subclass_link_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    sub_link = Link.inner(JoinSpec(UnjoinedSubCustomers, "ujsc_id"), JoinSpec(UnjoinedSubOrders, "ujso_cust_id"))
    messages = _plan_warnings(
        caplog,
        ["UnjoinedSubNameAmount"],
        {_UNJOINED_LINK, sub_link},
        {UnjoinedSubCustomers, UnjoinedSubOrders, UnjoinedSubNameAmount},
    )
    assert messages == []
