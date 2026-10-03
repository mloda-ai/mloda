from datetime import datetime, timezone
import subprocess  # nosec B404
import sys


from pathlib import Path

import pytest

from mloda.user import DataAccessCollection, PluginCollector, PluginLoader
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader

# Registers PandasDataFrame so test_create_synthetic_data resolves it when run in isolation.
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401

from docs.docs.examples.mloda_basics.create_synthetic_data import (
    CategoricalSyntheticDataSet,
    LocationSyntheticDataSet,
    MlLifeCycleDataCreator,
    OrderSyntheticDataSet,
    PaymentSyntheticDataSet,
)
from mloda.user import Feature
from mloda.user import mloda
from mloda.user import GlobalFilter


REPO_ROOT = Path(__file__).resolve().parents[3]

NOTEBOOKS: list[str] = [
    "docs/docs/examples/base_usage.py",
    "docs/docs/examples/sklearn_integration_basic.py",
    "docs/docs/examples/mloda_basics/0_table_of_content.py",
    "docs/docs/examples/mloda_basics/1_ml_mloda_intro.py",
    "docs/docs/examples/mloda_basics/2_ml_advantage_process_focus.py",
    "docs/docs/examples/mloda_basics/3_ml_data_feature_feature_groups.py",
    "docs/docs/examples/mloda_basics/4_ml_data_providers_user_steward.py",
]


class TestMlodaBasicsNotebooks:
    @pytest.mark.notebooks
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "notebook",
        [pytest.param(nb, id=Path(nb).stem) for nb in NOTEBOOKS],
    )
    def test_notebook_runs(self, notebook: str) -> None:
        notebook_path = REPO_ROOT / notebook
        result = subprocess.run(  # nosec B603
            [sys.executable, str(notebook_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Notebook {notebook} failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

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

        feature_list: list[Feature | str] = []
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
            event_time_column=Feature("created_at", options=example_options),
            validity_time_column="valid_datetime",
        )

        # Currently, we do not support multiple names for the same filter.
        global_filter.add_time_and_time_travel_filters(
            event_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            event_to=datetime(2023, 1, 1, tzinfo=timezone.utc),
            event_time_column=Feature("update_date", options=example_options),
        )

        plugin_collector = PluginCollector.enabled_feature_groups(
            {OrderSyntheticDataSet, PaymentSyntheticDataSet, LocationSyntheticDataSet, CategoricalSyntheticDataSet}
        )

        result = mloda.run_all(
            feature_list,
            compute_frameworks=["PandasDataFrame"],
            global_filter=global_filter,
            plugin_collector=plugin_collector,
        )
        assert 4 == len(result)

        for res in result:
            assert len(res) > 0


class TestMlodaBasicsBaseDataSources:
    """The demo base_data folder plus sqlite credentials give one column two sources."""

    @staticmethod
    def _data_access_collection() -> DataAccessCollection:
        base_data = REPO_ROOT / "docs" / "docs" / "examples" / "mloda_basics" / "base_data"
        collection = DataAccessCollection()
        collection.add_folder(str(base_data))
        collection.add_credentials({SQLITEReader.db_path(): str(base_data / "example.sqlite")})
        return collection

    def test_bare_shared_column_is_ambiguous_between_file_and_db(self) -> None:
        PluginLoader.all()

        with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
            mloda.run_all(
                [Feature("transaction_id")],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=self._data_access_collection(),
            )

        message = str(exc_info.value)
        assert "ReadFileFeature" in message
        assert "ReadDBFeature" in message
        assert "BaseInputData already set" not in message

    @pytest.mark.parametrize("scope", ["ReadFileFeature", "ReadDBFeature"])
    def test_scoped_shared_column_loads_from_its_own_source(self, scope: str) -> None:
        PluginLoader.all()

        results = mloda.run_all(
            [Feature("transaction_id", feature_group=scope)],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=self._data_access_collection(),
        )

        assert len(results) == 1
        assert results[0].num_rows == 100
