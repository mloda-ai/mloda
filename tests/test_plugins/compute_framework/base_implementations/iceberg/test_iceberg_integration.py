from mloda_core.abstract_plugins.components.feature_collection import Features
import pytest
from typing import Any, Dict, List, Optional, Set, Type, Union
from unittest.mock import Mock, patch

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.match_data.match_data import MatchData
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.api.request import mlodaAPI
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None
    IcebergTable = None  # type: ignore
    Catalog = None  # type: ignore


@pytest.fixture
def mock_iceberg_catalog() -> Mock:
    """Create a mock Iceberg catalog for testing."""
    mock_catalog = Mock(spec=Catalog)
    mock_catalog.load_table = Mock()
    return mock_catalog


@pytest.fixture
def mock_iceberg_table() -> Mock:
    """Create a mock Iceberg table for testing."""
    mock_table = Mock(spec=IcebergTable)

    # Create mock scan that returns PyArrow data
    mock_scan = Mock()
    arrow_data = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
            "category": ["A", "B", "A", "C", "B"],
            "score": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    mock_scan.to_arrow.return_value = arrow_data
    mock_table.scan.return_value = mock_scan

    # Mock schema
    mock_schema = Mock()
    mock_schema.column_names = ["id", "value", "category", "score"]
    mock_table.schema.return_value = mock_schema

    return mock_table


iceberg_test_dict = {
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
    "category": ["A", "B", "A", "C", "B"],
    "score": [1.5, 2.5, 3.5, 4.5, 5.5],
}


class IcebergTestDataCreator(AbstractFeatureGroup):
    """Test data creator for Iceberg integration tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        """Return a DataCreator with the supported feature names."""
        return DataCreator(set(iceberg_test_dict.keys()))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Create mock Iceberg table with test data."""
        mock_table = Mock(spec=IcebergTable)

        # Create mock scan that returns PyArrow data
        mock_scan = Mock()
        arrow_data = pa.Table.from_pydict(iceberg_test_dict)
        mock_scan.to_arrow.return_value = arrow_data
        mock_table.scan.return_value = mock_scan

        # Mock schema
        mock_schema = Mock()
        mock_schema.column_names = list(iceberg_test_dict.keys())
        mock_table.schema.return_value = mock_schema

        return mock_table

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Return the Iceberg compute framework."""
        return {IcebergFramework}


class ATestIcebergFeatureGroup(AbstractFeatureGroup, MatchData):
    """Base class for Iceberg feature groups."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        """Check for data access collection if any child classes match the data access."""

        if not IcebergFramework.is_available():
            return False

        if feature_name not in cls.feature_names_supported():
            return False

        # For testing, we'll use a mock catalog or table
        if isinstance(framework_connection_object, (Mock, IcebergTable)):
            return framework_connection_object

        if data_access_collection is None:
            return False

        if data_access_collection.initialized_connection_objects is None:
            return False

        if data_access_collection.initialized_connection_objects:
            for conn in data_access_collection.initialized_connection_objects:
                if isinstance(conn, (Mock, IcebergTable)) or hasattr(conn, "load_table"):
                    return conn
        return False

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {IcebergFramework}


class IcebergSimpleTransformFeatureGroup(AbstractFeatureGroup):
    """Simple feature group for testing Iceberg transformations."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Require base features for transformation."""
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str == "doubled_value":
            return {Feature("value")}
        elif feature_name_str == "score_plus_ten":
            return {Feature("score")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Perform simple transformations on the data."""
        # Since Iceberg tables are read-only in this context, we'll work with PyArrow
        if isinstance(data, IcebergTable):
            # Convert Iceberg table to PyArrow for processing
            arrow_data = data.scan().to_arrow()
        else:
            arrow_data = data

        # Perform transformations using PyArrow compute
        import pyarrow.compute as pc

        result_data = arrow_data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "doubled_value":
                # Add doubled_value column
                doubled_values = pc.multiply(arrow_data["value"], 2)
                result_data = result_data.append_column("doubled_value", doubled_values)

            elif feature_name == "score_plus_ten":
                # Add score_plus_ten column
                score_plus_ten = pc.add(arrow_data["score"], 10)
                result_data = result_data.append_column("score_plus_ten", score_plus_ten)

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"doubled_value", "score_plus_ten"}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {IcebergFramework}


class IcebergToArrowFeatureGroup(AbstractFeatureGroup):
    """Feature group that converts Iceberg data to PyArrow format."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("doubled_value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Convert to PyArrow and rename column."""
        # Ensure we're working with PyArrow data
        if isinstance(data, IcebergTable):
            arrow_data = data.scan().to_arrow()
        else:
            arrow_data = data

        # Rename the doubled_value column to arrow_doubled_value
        result_data = arrow_data
        for feat in features.features:
            feature_name = str(feat.name)
            if feature_name == "arrow_doubled_value":
                # Rename doubled_value to arrow_doubled_value
                schema = arrow_data.schema
                new_names = [name if name != "doubled_value" else "arrow_doubled_value" for name in schema.names]
                result_data = arrow_data.rename_columns(new_names)

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"arrow_doubled_value"}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergIntegrationWithMlodaAPI:
    """Integration tests for IcebergFramework with mlodaAPI."""

    @pytest.mark.parametrize(
        "modes",
        [({ParallelizationModes.SYNC})],
    )
    def test_basic_iceberg_feature_calculation(
        self, modes: Set[ParallelizationModes], flight_server: Any, mock_iceberg_catalog: Mock
    ) -> None:
        """Test basic feature calculation with Iceberg framework."""
        # Enable the test feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {IcebergTestDataCreator, IcebergSimpleTransformFeatureGroup}
        )

        # Define features to calculate with catalog connection
        feature_list: Features | List[Feature | str] = [
            Feature(name="doubled_value", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
            Feature(name="score_plus_ten", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
        ]

        data_access_collection = DataAccessCollection(initialized_connection_objects={mock_iceberg_catalog})

        # Run with Iceberg framework
        result = mlodaAPI.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes=modes,
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks={IcebergFramework},
        )

        # The result should be a PyArrow table (converted from Iceberg)
        final_data = result[0]
        assert hasattr(final_data, "column_names")

        # Verify the transformations worked
        assert "doubled_value" in final_data.column_names
        assert "score_plus_ten" in final_data.column_names

        # Check some values
        data_dict = final_data.to_pydict()
        doubled_values = data_dict["doubled_value"]
        assert doubled_values == [20, 40, 60, 80, 100]  # Original values * 2

        score_plus_ten = data_dict["score_plus_ten"]
        assert score_plus_ten == [11.5, 12.5, 13.5, 14.5, 15.5]  # Original scores + 10

    def test_iceberg_to_pyarrow_transformation(self, flight_server: Any, mock_iceberg_catalog: Mock) -> None:
        """Test transformation from Iceberg to PyArrow framework."""
        # Enable feature groups for cross-framework transformation
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {IcebergTestDataCreator, IcebergSimpleTransformFeatureGroup, IcebergToArrowFeatureGroup}
        )

        # Define feature that requires transformation between frameworks
        feature_list: Features | List[Feature | str] = [
            Feature(name="arrow_doubled_value", options={"IcebergTestDataCreator": mock_iceberg_catalog})
        ]

        data_access_collection = DataAccessCollection(initialized_connection_objects={mock_iceberg_catalog})

        # Run with both Iceberg and PyArrow frameworks
        result = mlodaAPI.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes={ParallelizationModes.SYNC},
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks={IcebergFramework, PyarrowTable},
        )

        # Verify results
        final_data = result[0]
        assert "arrow_doubled_value" in final_data.column_names

        # Check that the transformation worked correctly
        data_dict = final_data.to_pydict()
        arrow_doubled_values = data_dict["arrow_doubled_value"]
        assert arrow_doubled_values == [20, 40, 60, 80, 100]  # Original values * 2

    def test_iceberg_framework_availability_check(self) -> None:
        """Test that Iceberg framework availability is correctly detected."""
        # This test verifies that the framework correctly detects PyIceberg availability
        assert IcebergFramework.is_available() is True

    def test_iceberg_data_creator_basic_functionality(self, flight_server: Any, mock_iceberg_catalog: Mock) -> None:
        """Test basic functionality of the Iceberg data creator."""
        # Enable just the data creator
        plugin_collector = PlugInCollector.enabled_feature_groups({IcebergTestDataCreator})

        # Request basic features from the data creator
        feature_list: Features | List[Feature | str] = [
            Feature(name="id", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
            Feature(name="value", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
            Feature(name="category", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
        ]

        data_access_collection = DataAccessCollection(initialized_connection_objects={mock_iceberg_catalog})

        # Run with Iceberg framework
        result = mlodaAPI.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes={ParallelizationModes.SYNC},
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks={IcebergFramework},
        )

        # Verify results
        final_data = result[0]
        assert isinstance(final_data, Mock)  # It's a mock Iceberg table
