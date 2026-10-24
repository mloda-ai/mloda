import sqlite3
import numpy as np

from typing import Any

import pandas as pd

from mloda.user import mloda
from mloda.provider import FeatureSet, GeneratedFeatureGroup
from mloda.user import DataAccessCollection, Domain, Feature, FeatureName, Options


class MlLifeCycleDataCreator(GeneratedFeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Match only when the MlLifeCycleDataCreator option is set, so other lookups do not find it.
        if options.get(MlLifeCycleDataCreator.__name__) is None:
            return False
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)


class OrderSyntheticDataSet(MlLifeCycleDataCreator):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"order_id", "product_id", "quantity", "item_price", "created_at"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        num_samples = next((f.options.get("num_samples") for f in features.features), None)

        if num_samples is None:
            raise ValueError("No num_samples option set for features.")

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
    def feature_names_supported(cls) -> set[str]:
        return {"payment_id", "payment_type", "payment_status", "created_at", "valid_datetime"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        num_samples = features.get_options_key("num_samples")

        # Shuffle valid datetime information
        valid_datetime_dates = pd.date_range(start="2024-01-01", end="2024-02-01", periods=num_samples, tz="UTC")
        valid_datetime_dates = valid_datetime_dates.to_numpy(copy=True)
        np.random.shuffle(valid_datetime_dates)

        return pd.DataFrame(
            {
                "payment_id": range(1, num_samples + 1),
                "payment_type": np.random.choice(["credit card", "paypal", "debit card", "store credit"], num_samples),
                "payment_status": np.random.choice(["completed", "failed", "pending", "refunded"], num_samples),
                "created_at": pd.date_range(start="2022-01-01", end="2024-01-01", periods=num_samples, tz="UTC"),
                "valid_datetime": valid_datetime_dates,
            }
        )


class LocationSyntheticDataSet(MlLifeCycleDataCreator):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"transaction_id", "user_location", "merchant_location", "update_date"}

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


class CategoricalSyntheticDataSet(MlLifeCycleDataCreator):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"transaction_id", "user_age_group", "product_category", "transaction_type"}

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

    feature_list: list[Feature | str] = []
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

    results = mloda.run_all(feature_list, compute_frameworks=["PandasDataFrame"])

    results[0].to_csv("base_data/output.csv", index=False)
    results[1].to_parquet("base_data/output.parquet", index=False)
    results[2].to_json("base_data/output.json", orient="records", lines=True)

    with sqlite3.connect("base_data/example.sqlite") as conn:
        results[3].to_sql("example_table", conn, if_exists="replace", index=False)

    for res in results:
        print(res[:2])
