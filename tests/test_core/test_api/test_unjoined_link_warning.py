"""Planning must warn when a declared link plans no join (issue #1998).

A ``Link`` between two feature groups joins nothing unless some requested
feature reads both sides, yet ``docs/docs/in_depth/join_data.md`` presents
``mloda.run_all(features=[...], links=...)`` as if the link alone merged the
frames. Without a consumer the run silently returns the two sides as separate,
unjoined frames and the caller has no join keys left to merge them by, so
planning must log one WARNING per declared link whose two sides both appear in
the plan but which no step joins.

Contract under test:

  * Both sides in the plan, no consumer: exactly one WARNING record naming
    ``Customers`` and ``Orders`` and naming the remedy, a feature group whose
    ``input_features()`` read both sides. The run still returns the two
    separate frames, so the warning cannot start joining as a side effect.
  * The same holds when the link travels on the feature itself
    (``Feature(..., link=link)``, no ``links=`` kwarg): both declaration ways
    must warn, and the results stay two separate frames.
  * Only one side in the plan: no such WARNING, a shared link set reused
    across runs stays quiet.
  * Neither side in the plan: no such WARNING.
  * A consumer whose ``input_features()`` read both sides joins: no such
    WARNING, and ``NameAmount`` still computes ``a10``, ``b20``, ``c30``.

Warning records are recognized by a message carrying BOTH feature-group class
names, never by logger name.

Root feature names carry an ``unjoined_link_`` prefix on purpose: a
``DataCreator`` claim is registry-wide, so generic names like ``name`` or
``id`` would collide with every other test resolving them.
"""

import logging
from typing import Any

import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, JoinSpec, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

CUSTOMERS = "Customers"
ORDERS = "Orders"
REMEDY = "input_features()"
NAME = "unjoined_link_name"
AMOUNT = "unjoined_link_amount"
SKU = "unjoined_link_sku"
CUSTOMER_KEY = "unjoined_link_id"
ORDER_KEY = "unjoined_link_cust_id"


class Customers(FeatureGroup):
    """Left side of the link: root group holding the customer key and name columns."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({CUSTOMER_KEY, NAME})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {CUSTOMER_KEY: [1, 2, 3], NAME: ["a", "b", "c"]}


class Orders(FeatureGroup):
    """Right side of the link: root group holding the order key and amount columns."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({ORDER_KEY, AMOUNT})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {ORDER_KEY: [3, 1, 2], AMOUNT: [30, 10, 20]}


class NameAmount(FeatureGroup):
    """Consumer whose ``input_features()`` read both sides, so the link joins for it."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(NAME), Feature(AMOUNT)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"NameAmount": data[NAME] + data[AMOUNT].astype(str)}


class Products(FeatureGroup):
    """Group unrelated to the link: neither side of it appears in the plan."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({SKU})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {SKU: ["p1", "p2"]}


link = Link.inner(JoinSpec(Customers, CUSTOMER_KEY), JoinSpec(Orders, ORDER_KEY))


def _plugin_collector() -> PluginCollector:
    """Fresh collector per run, restricted to this module's four feature groups."""
    return PluginCollector.enabled_feature_groups({Customers, Orders, NameAmount, Products})


def _unjoined_link_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """WARNING records naming both link sides: the issue #1998 unjoined-link diagnostic."""
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and CUSTOMERS in record.getMessage() and ORDERS in record.getMessage()
    ]


class TestUnjoinedDeclaredLinkWarning:
    """Issue #1998: a declared link whose two sides both appear in the plan but which no step joins warns once."""

    def test_both_sides_in_plan_without_consumer_warns_once_and_keeps_two_frames(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No requested feature reads both sides: one warning naming input_features(), results unchanged."""
        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [NAME, AMOUNT],
                compute_frameworks=[PandasDataFrame],
                links={link},
                plugin_collector=_plugin_collector(),
            )

        records = _unjoined_link_warnings(caplog)
        assert len(records) == 1, (
            f"expected exactly one WARNING naming {CUSTOMERS} and {ORDERS}, got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert CUSTOMERS in message
        assert ORDERS in message
        assert REMEDY in message

        assert len(result) == 2, f"the warning must not join anything, got {len(result)} frames"
        columns = {column for frame in result for column in frame.columns}
        assert columns == {NAME, AMOUNT}

    def test_only_one_side_of_the_link_in_the_plan_stays_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        """Requesting only a Customers-side feature with the link declared logs no such warning."""
        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [NAME],
                compute_frameworks=[PandasDataFrame],
                links={link},
                plugin_collector=_plugin_collector(),
            )

        assert _unjoined_link_warnings(caplog) == []
        assert len(result) == 1

    def test_neither_side_of_the_link_in_the_plan_stays_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        """Requesting only an unrelated group's feature with the link declared logs no such warning."""
        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [SKU],
                compute_frameworks=[PandasDataFrame],
                links={link},
                plugin_collector=_plugin_collector(),
            )

        assert _unjoined_link_warnings(caplog) == []
        assert len(result) == 1

    def test_consumer_joining_both_sides_logs_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A consumer whose input_features() read both sides joins, so planning stays silent and the join computes."""
        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                ["NameAmount"],
                compute_frameworks=[PandasDataFrame],
                links={link},
                plugin_collector=_plugin_collector(),
            )

        assert _unjoined_link_warnings(caplog) == []
        assert len(result) == 1
        assert result[0]["NameAmount"].tolist() == ["a10", "b20", "c30"]

    def test_feature_level_link_declaration_warns_once_and_keeps_two_frames(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Issue #1998: the link declared on the feature itself, with no links= kwarg, warns exactly the same."""
        with caplog.at_level(logging.WARNING):
            result = mloda.run_all(
                [Feature(NAME), Feature(AMOUNT, link=link)],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=_plugin_collector(),
            )

        records = _unjoined_link_warnings(caplog)
        assert len(records) == 1, (
            f"expected exactly one WARNING naming {CUSTOMERS} and {ORDERS}, got {len(records)}: "
            f"{[record.getMessage() for record in records]}"
        )
        message = records[0].getMessage()
        assert CUSTOMERS in message
        assert ORDERS in message
        assert REMEDY in message

        assert len(result) == 2, f"the warning must not join anything, got {len(result)} frames"
        columns = {column for frame in result for column in frame.columns}
        assert columns == {NAME, AMOUNT}
