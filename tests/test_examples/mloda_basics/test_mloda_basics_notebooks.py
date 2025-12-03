from datetime import datetime, timezone
import os


from pathlib import Path
from typing import Any, List

from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from testbook import testbook

from docs.docs.examples.mloda_basics.create_synthetic_data import (
    CategoricalSyntheticDataSet,
    LocationSyntheticDataSet,
    MlLifeCycleDataCreator,
    OrderSyntheticDataSet,
    PaymentSyntheticDataSet,
)
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.api.request import mlodaAPI
from mloda_core.filter.global_filter import GlobalFilter


root_dir = Path(os.path.abspath(os.curdir))

root_dir = root_dir.joinpath("docs").joinpath("docs").joinpath("examples").joinpath("mloda_basics")

lifecycle_path = root_dir.joinpath("1_ml_mloda_intro.ipynb")
lifecycle_path_2 = root_dir.joinpath("2_ml_advantage_process_focus.ipynb")
lifecycle_path_4 = root_dir.joinpath("4_ml_data_producers_user_owner.ipynb")


class TestMlodaBasicsNotebooks:
    @testbook(lifecycle_path, execute=True)  # type: ignore
    def atest_notebook_1_mloda_basics(self, tb: Any) -> None:
        pass

    @testbook(lifecycle_path_2, execute=True)  # type: ignore
    def test_notebook_2_mloda_basics(self, tb: Any) -> None:
        pass

    @testbook(lifecycle_path_4, execute=True)  # type: ignore
    def test_notebook_4_mloda_basics(self, tb: Any) -> None:
        pass

    def test_create_synthetic_data(self) -> None:
        # Order Data
        str_feature_list = ["order_id", "product_id", "quantity", "item_price"]

        # Payment Data
        str_feature_list += ["payment_id", "payment_type", "payment_status"]

        # Location data
        str_feature_list += ["user_location", "merchant_location"]

        # Category data
        str_feature_list += ["user_age_group", "product_category", "transaction_type"]

        # We are setting the MlLifeCycleDataCreator to not get confused with other tests
        example_options = {MlLifeCycleDataCreator.__name__: True, "num_samples": 100}

        feature_list: List[Feature | str] = []
        for feature in str_feature_list:
            feature_list.append(Feature(name=feature, options=example_options))

        # We are adding here the domain information to achieve a unique resolution for the transaction_id.
        feature_list.append(
            Feature(
                name="transaction_id",
                domain="Location",
                options=example_options,
            )
        )
        feature_list.append(
            Feature(
                name="transaction_id",
                domain="Categorical",
                options=example_options,
            )
        )

        global_filter = GlobalFilter()
        global_filter.add_time_and_time_travel_filters(
            event_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            event_to=datetime(2023, 1, 1, tzinfo=timezone.utc),
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2024, 1, 15, tzinfo=timezone.utc),
            time_filter_feature=Feature("created_at", options=example_options),
            time_travel_filter_feature="valid_datetime",
        )

        # Currently, we do not support multiple names for the same filter.
        global_filter.add_time_and_time_travel_filters(
            event_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            event_to=datetime(2023, 1, 1, tzinfo=timezone.utc),
            time_filter_feature=Feature("update_date", options=example_options),
        )

        plugin_collector = PlugInCollector.enabled_feature_groups(
            {OrderSyntheticDataSet, PaymentSyntheticDataSet, LocationSyntheticDataSet, CategoricalSyntheticDataSet}
        )

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks=["PandasDataFrame"],
            global_filter=global_filter,
            plugin_collector=plugin_collector,
        )
        assert 4 == len(result)

        for res in result:
            assert len(res) > 0
