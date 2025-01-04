import sqlite3
import numpy as np

from typing import Any, List, Optional

import pandas as pd

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.domain import Domain
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.api.request import mlodaAPI


class MlLifeCycleDataCreator(DataCreator):
    def matches(
        self, feature_name: str, options: Options, data_access_collection: Optional[DataAccessCollection] = None
    ) -> bool:
        # This match function is only adjusted as it is part of a larger project. Else this DataCreator might be found by other
        # functionalities as well, e.g. in the testing framework. This should not be the case in a normal project.
        if options.get(MlLifeCycleDataCreator.__name__) is None:
            return False
        if feature_name in self.feature_names:
            return True
        return False


class OrderSyntheticDataSet(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return MlLifeCycleDataCreator({"order_id", "product_id", "quantity", "item_price", "created_at"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        num_samples = features.get_options_key("num_samples")

        return pd.DataFrame(
            {
                "order_id": range(1, num_samples + 1),
                "product_id": np.random.randint(100, 500, num_samples),
                "quantity": np.random.randint(1, 10, num_samples),
                "item_price": np.random.uniform(10, 200, num_samples).round(2),
                "created_at": pd.date_range(start="2022-01-01", end="2024-01-01", periods=num_samples, tz="UTC"),
            }
        )


class PaymentSyntheticDataSet(OrderSyntheticDataSet):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return MlLifeCycleDataCreator({"payment_id", "payment_type", "payment_status", "created_at", "valid_datetime"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        num_samples = features.get_options_key("num_samples")

        # Shuffle valid datetime informations
        valid_datetime_dates = pd.date_range(start="2024-01-01", end="2024-02-01", periods=num_samples, tz="UTC")
        np.random.shuffle(valid_datetime_dates.values)

        return pd.DataFrame(
            {
                "payment_id": range(1, num_samples + 1),
                "payment_type": np.random.choice(["credit card", "paypal", "debit card", "store credit"], num_samples),
                "payment_status": np.random.choice(["completed", "failed", "pending", "refunded"], num_samples),
                "created_at": pd.date_range(start="2022-01-01", end="2024-01-01", periods=num_samples, tz="UTC"),
                "valid_datetime": valid_datetime_dates,
            }
        )


class LocationSyntheticDataSet(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return MlLifeCycleDataCreator({"transaction_id", "user_location", "merchant_location", "update_date"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        num_samples = features.get_options_key("num_samples")

        return pd.DataFrame(
            {
                "transaction_id": range(1, num_samples + 1),
                "user_location": np.random.choice(["North", "South", "East", "West"], num_samples),
                "merchant_location": np.random.choice(["North", "South", "East", "West"], num_samples),
                "update_date": pd.date_range(start="2022-01-01", end="2024-01-01", periods=num_samples, tz="UTC"),
            }
        )

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("Location")


class CategoricalSyntheticDataSet(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return MlLifeCycleDataCreator({"transaction_id", "user_age_group", "product_category", "transaction_type"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        num_samples = features.get_options_key("num_samples")

        return pd.DataFrame(
            {
                "transaction_id": range(1, num_samples + 1),
                "user_age_group": np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"], num_samples),
                "product_category": np.random.choice(
                    ["electronics", "clothing", "home", "beauty", "books"], num_samples
                ),
                "transaction_type": np.random.choice(["online", "in-store", "mail-order", "telephone"], num_samples),
            }
        )

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("Categorical")


def create_ml_lifecylce_data() -> None:
    # Order Data
    str_feature_list = ["order_id", "product_id", "quantity", "item_price", "created_at"]

    # Payment Data
    str_feature_list += ["payment_id", "payment_type", "payment_status", "valid_datetime"]

    # Location data
    str_feature_list += ["user_location", "merchant_location", "transaction_id", "update_date"]

    # Category data
    str_feature_list += ["user_age_group", "product_category", "transaction_type", "transaction_id"]

    domain_iterator = ["Location", "Categorical"]

    options = {"MlLifeCycleDataCreator": True, "num_samples": 100}

    feature_list: List[Feature | str] = []
    for feature in str_feature_list:
        # Transaction_id has a special treatment, as we have it twice with different domains.
        if feature == "transaction_id":
            feature_list.append(
                Feature(
                    name=feature,
                    domain=domain_iterator.pop(0),
                    options=options,
                )
            )
        else:
            feature_list.append(Feature(name=feature, options=options))

    results = mlodaAPI.run_all(feature_list, compute_frameworks=["PandasDataframe"])

    results[0].to_csv("base_data/output.csv", index=False)
    results[1].to_parquet("base_data/output.parquet", index=False)
    results[2].to_json("base_data/output.json", orient="records", lines=True)

    with sqlite3.connect("base_data/example.sqlite") as conn:
        results[3].to_sql("example_table", conn, if_exists="replace", index=False)

    for res in results:
        print(res[:2])
