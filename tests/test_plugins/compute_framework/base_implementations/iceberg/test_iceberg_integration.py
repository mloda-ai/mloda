from mloda.user import Features
import pytest
from typing import Any
from unittest.mock import Mock

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.provider import DataCreator
from mloda.provider import BaseInputData
from mloda.provider import MatchData
from mloda.provider import ComputeFramework
from mloda.user import mloda
from mloda.user import ParallelizationMode
from mloda.user import DataAccessCollection
from mloda.user import GlobalFilter
from mloda.user import Index
from mloda.user import Link
from mloda.user import JoinSpec
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_core.test_filter.test_feature_group_final_filters import RegularFeatureGroupForFilterTest
from tests.test_core.test_tooling import PARALLELIZATION_MODES_SYNC_THREADING

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType, StructType
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None  # type: ignore[assignment, unused-ignore]
    IcebergTable = None  # type: ignore
    Catalog = None  # type: ignore
    Schema = None  # type: ignore
    LongType = None  # type: ignore
    NestedField = None  # type: ignore
    StringType = None  # type: ignore
    StructType = None  # type: ignore


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


class IcebergTestDataCreator(FeatureGroup):
    """Test data creator for Iceberg integration tests."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
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
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        """Return the Iceberg compute framework."""
        return {IcebergFramework}


class ATestIcebergFeatureGroup(FeatureGroup, MatchData):
    """Base class for Iceberg feature groups."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        """Check for data access collection if any child classes match the data access."""

        if not IcebergFramework.is_available():
            return None

        if feature_name not in cls.feature_names_supported():
            return None

        # For testing, we'll use a mock catalog or table
        if isinstance(framework_connection_object, (Mock, IcebergTable)):
            return framework_connection_object

        if data_access_collection is None:
            return None

        if data_access_collection.connections:
            for conn in data_access_collection.connections.values():
                if isinstance(conn, (Mock, IcebergTable)) or hasattr(conn, "load_table"):
                    return conn
        return None

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}


class IcebergSimpleTransformFeatureGroup(FeatureGroup):
    """Simple feature group for testing Iceberg transformations."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Require base features for transformation."""
        feature_name_str = str(feature_name) if isinstance(feature_name, FeatureName) else str(feature_name)

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
    def feature_names_supported(cls) -> set[str]:
        return {"doubled_value", "score_plus_ten"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}


class IcebergToArrowFeatureGroup(FeatureGroup):
    """Feature group that converts Iceberg data to PyArrow format."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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
    def feature_names_supported(cls) -> set[str]:
        return {"arrow_doubled_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class IcebergPyArrowRegularFeatureGroupForFilterTest(RegularFeatureGroupForFilterTest):
    """Same data as RegularFeatureGroupForFilterTest, run on IcebergFramework via a pa.Table result."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}


class IcebergTableRegularFeatureGroupForFilterTest(RegularFeatureGroupForFilterTest):
    """Same data as RegularFeatureGroupForFilterTest, in a Mock Iceberg Table whose schema() returns a real Schema."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        arrow_data = super().calculate_feature(data, features)

        mock_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_scan.to_arrow.return_value = arrow_data
        mock_table.scan.return_value = mock_scan
        mock_table.schema.return_value = Schema(
            NestedField(1, cls.get_class_name(), LongType()),
            NestedField(2, "status", StringType()),
        )
        return mock_table


class IcebergTableStructFieldFilterTest(FeatureGroup):
    """Requests the nested field 'b.c', which the plain scan drops from the arrow result."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"b.c", "status"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        b = pa.array([{"c": 10}, {"c": 20}, {"c": 30}, {"c": 40}], type=pa.struct([("c", pa.int64())]))
        status = pa.array(["active", "inactive", "active", "inactive"])
        arrow_data = pa.table({"b": b, "status": status})

        mock_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_scan.to_arrow.return_value = arrow_data
        mock_table.scan.return_value = mock_scan
        mock_table.schema.return_value = Schema(
            NestedField(1, "b", StructType(NestedField(2, "c", LongType(), required=False)), required=False),
            NestedField(3, "status", StringType(), required=False),
        )
        return mock_table


IJK_KEY = "ijk_key"
IJK_KEY2 = "ijk_key2"
IJK_PAYLOAD = "ijk_payload"
IJK_STATUS = "ijk_status"
IJK_RIGHT_PAYLOAD = "ijk_right_payload"

# Captured Mock Iceberg table so the test can inspect its scan() call after mloda.run_all finishes.
ijk_captured_tables: list[Mock] = []


class IcebergJoinKeyLeftFG(FeatureGroup):
    """Only the payload is requested by the consumer; the composite index's second column (ijk_key2) is
    never a requested feature and only survives a projected, filtered scan via the link index column stamp."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={IJK_KEY, IJK_KEY2, IJK_PAYLOAD, IJK_STATUS})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index((IJK_KEY, IJK_KEY2))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        arrow_data = pa.table(
            {
                IJK_KEY: [1, 2, 3, 4],
                IJK_KEY2: ["k1", "k2", "k3", "k4"],
                IJK_PAYLOAD: [10, 20, 30, 40],
                IJK_STATUS: ["active", "inactive", "active", "inactive"],
            }
        )

        def _scan(*args: Any, **kwargs: Any) -> Mock:
            selected = kwargs.get("selected_fields", arrow_data.column_names)
            mock_scan = Mock()
            mock_scan.to_arrow.return_value = arrow_data.select(list(selected))
            return mock_scan

        mock_table = Mock(spec=IcebergTable)
        mock_table.scan.side_effect = _scan
        mock_table.schema.return_value = Schema(
            NestedField(1, IJK_KEY, LongType(), required=False),
            NestedField(2, IJK_KEY2, StringType(), required=False),
            NestedField(3, IJK_PAYLOAD, LongType(), required=False),
            NestedField(4, IJK_STATUS, StringType(), required=False),
        )
        ijk_captured_tables.append(mock_table)
        return mock_table


class IcebergJoinKeyRightFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={IJK_KEY, IJK_KEY2, IJK_RIGHT_PAYLOAD})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index((IJK_KEY, IJK_KEY2))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                IJK_KEY: [1, 2, 3],
                IJK_KEY2: ["k1", "k2", "k3"],
                IJK_RIGHT_PAYLOAD: ["r1", "r2", "r3"],
            }
        )


class IcebergJoinKeyConsumerFG(FeatureGroup):
    """Combines both payloads; requesting neither side's join key directly."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=IJK_PAYLOAD), Feature(name=IJK_RIGHT_PAYLOAD)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        payloads = data[IJK_PAYLOAD].to_pylist()
        right_payloads = data[IJK_RIGHT_PAYLOAD].to_pylist()
        combined = [f"{left}|{right}" for left, right in zip(payloads, right_payloads)]
        return data.append_column(cls.get_class_name(), pa.array(combined))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


ANCESTOR_STAMP_DESC_K = "ancestor_stamp_desc_k"
ANCESTOR_STAMP_VALUE = "ancestor_stamp_value"
ANCESTOR_STAMP_STATUS = "ancestor_stamp_status"
ANCESTOR_STAMP_C_KEY = "ancestor_stamp_c_key"
ANCESTOR_STAMP_C_PAYLOAD = "ancestor_stamp_c_payload"


class AncestorStampRootFG(FeatureGroup):
    """Iceberg root: key column desc_k, value, status, filtered by a GlobalFilter on status."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={ANCESTOR_STAMP_DESC_K, ANCESTOR_STAMP_VALUE, ANCESTOR_STAMP_STATUS})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {IcebergFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        arrow_data = pa.table(
            {
                ANCESTOR_STAMP_DESC_K: ["d1", "d2", "d3", "d4"],
                ANCESTOR_STAMP_VALUE: [10, 20, 30, 40],
                ANCESTOR_STAMP_STATUS: ["active", "inactive", "active", "inactive"],
            }
        )

        def _scan(*args: Any, **kwargs: Any) -> Mock:
            selected = kwargs.get("selected_fields", arrow_data.column_names)
            mock_scan = Mock()
            mock_scan.to_arrow.return_value = arrow_data.select(list(selected))
            return mock_scan

        mock_table = Mock(spec=IcebergTable)
        mock_table.scan.side_effect = _scan
        mock_table.schema.return_value = Schema(
            NestedField(1, ANCESTOR_STAMP_DESC_K, StringType(), required=False),
            NestedField(2, ANCESTOR_STAMP_VALUE, LongType(), required=False),
            NestedField(3, ANCESTOR_STAMP_STATUS, StringType(), required=False),
        )
        return mock_table


class AncestorStampMiddleFG(FeatureGroup):
    """Consumes the root's value; desc_k (the join index) is passed through raw from the root's data and
    is never itself a declared input_feature, so it depends entirely on the root step being stamped."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if str(feature_name) == ANCESTOR_STAMP_DESC_K:
            return None
        return {Feature(ANCESTOR_STAMP_VALUE)}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index((ANCESTOR_STAMP_DESC_K,))]

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {ANCESTOR_STAMP_DESC_K}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import pyarrow.compute as pc

        result = data
        for feat in features.features:
            if str(feat.name) == cls.get_class_name():
                doubled = pc.multiply(data[ANCESTOR_STAMP_VALUE], 2)
                result = result.append_column(cls.get_class_name(), doubled)
        return result


class AncestorStampOtherFG(FeatureGroup):
    """Unrelated PyArrow data creator, joined to the middle FG's desc_k on c_key."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={ANCESTOR_STAMP_C_KEY, ANCESTOR_STAMP_C_PAYLOAD})

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index((ANCESTOR_STAMP_C_KEY,))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                ANCESTOR_STAMP_C_KEY: ["d1", "d2", "d3"],
                ANCESTOR_STAMP_C_PAYLOAD: ["p1", "p2", "p3"],
            }
        )


class AncestorStampConsumerFG(FeatureGroup):
    """Combines the middle FG's own feature with the other side's payload."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name=AncestorStampMiddleFG.get_class_name()), Feature(name=ANCESTOR_STAMP_C_PAYLOAD)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        middles = data[AncestorStampMiddleFG.get_class_name()].to_pylist()
        payloads = data[ANCESTOR_STAMP_C_PAYLOAD].to_pylist()
        combined = [f"{middle}|{payload}" for middle, payload in zip(middles, payloads)]
        return data.append_column(cls.get_class_name(), pa.array(combined))

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergIntegrationWithMlodaAPI:
    """Integration tests for IcebergFramework with mloda."""

    @pytest.mark.parametrize(
        "feature_group",
        [
            pytest.param(IcebergPyArrowRegularFeatureGroupForFilterTest, id="pa_table"),
            pytest.param(IcebergTableRegularFeatureGroupForFilterTest, id="iceberg_table"),
        ],
    )
    @PARALLELIZATION_MODES_SYNC_THREADING
    def test_default_feature_group_final_filter_applied(
        self, feature_group: type[FeatureGroup], modes: set[ParallelizationMode], flight_server: Any
    ) -> None:
        """A default FeatureGroup (final_filters() -> None) still gets its rows filtered on Iceberg."""
        feature_name = feature_group.get_class_name()

        plugin_collector = PluginCollector.enabled_feature_groups({feature_group})

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = mloda.run_all(
            [Feature(name=feature_name, initial_requested_data=True)],
            flight_server=flight_server,
            parallelization_modes=modes,
            plugin_collector=plugin_collector,
            compute_frameworks=[IcebergFramework],
            global_filter=global_filter,
        )

        for final_data in result:
            assert final_data[feature_name].to_pylist() == [10, 30]
            assert final_data.column_names == [feature_name]

    def test_nested_struct_field_survives_final_filter_iceberg(self, flight_server: Any) -> None:
        """A default FeatureGroup requesting 'b.c' (nested) keeps it after the final filter step."""
        plugin_collector = PluginCollector.enabled_feature_groups({IcebergTableStructFieldFilterTest})

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        result = mloda.run_all(
            [Feature(name="b.c", initial_requested_data=True)],
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            compute_frameworks=[IcebergFramework],
            global_filter=global_filter,
        )

        for final_data in result:
            assert final_data["b.c"].to_pylist() == [10, 30]
            assert final_data.column_names == ["b.c"]
            assert final_data["b.c"].type == pa.int64()

    def test_nested_struct_field_same_shape_with_and_without_filter_iceberg(self, flight_server: Any) -> None:
        """'b.c' must have the same column name and type whether or not a global filter is active."""
        plugin_collector = PluginCollector.enabled_feature_groups({IcebergTableStructFieldFilterTest})

        global_filter = GlobalFilter()
        global_filter.add_filter("status", "equal", {"value": "active"})

        filtered_result = mloda.run_all(
            [Feature(name="b.c", initial_requested_data=True)],
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            compute_frameworks=[IcebergFramework],
            global_filter=global_filter,
        )
        unfiltered_result = mloda.run_all(
            [Feature(name="b.c", initial_requested_data=True)],
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            compute_frameworks=[IcebergFramework],
        )

        filtered_data = next(iter(filtered_result))
        unfiltered_data = next(iter(unfiltered_result))

        assert filtered_data.column_names == unfiltered_data.column_names == ["b.c"]
        assert filtered_data["b.c"].type == unfiltered_data["b.c"].type == pa.int64()
        assert unfiltered_data["b.c"].to_pylist() == [10, 20, 30, 40]

    def test_join_key_not_a_requested_feature_survives_projected_filter_scan(self, flight_server: Any) -> None:
        """A projected, filtered scan must still keep a join index column that is not itself requested."""
        ijk_captured_tables.clear()
        plugin_collector = PluginCollector.enabled_feature_groups(
            {IcebergJoinKeyLeftFG, IcebergJoinKeyRightFG, IcebergJoinKeyConsumerFG}
        )
        global_filter = GlobalFilter()
        global_filter.add_filter(IJK_STATUS, "equal", {"value": "active"})
        link = Link.inner_on(IcebergJoinKeyLeftFG, IcebergJoinKeyRightFG)

        result = mloda.run_all(
            [Feature(name=IcebergJoinKeyConsumerFG.get_class_name(), initial_requested_data=True)],
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            compute_frameworks=[PyArrowTable, IcebergFramework],
            links={link},
            global_filter=global_filter,
        )

        assert len(ijk_captured_tables) == 1
        call_args = ijk_captured_tables[0].scan.call_args
        assert IJK_KEY in call_args.kwargs["selected_fields"]
        assert IJK_KEY2 in call_args.kwargs["selected_fields"]

        matching = [
            frame for frame in result if IcebergJoinKeyConsumerFG.get_class_name() in getattr(frame, "column_names", [])
        ]
        assert len(matching) == 1
        values = sorted(matching[0][IcebergJoinKeyConsumerFG.get_class_name()].to_pylist())
        assert values == ["10|r1", "30|r3"]

    def test_ancestor_of_join_member_gets_link_index_column_stamped(self, flight_server: Any) -> None:
        """An Iceberg step two hops above a join member (through a pass-through FG) must still get the
        join index column stamped, since the join key reaches the join only as raw passed-through data."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {AncestorStampRootFG, AncestorStampMiddleFG, AncestorStampOtherFG, AncestorStampConsumerFG}
        )
        global_filter = GlobalFilter()
        global_filter.add_filter(ANCESTOR_STAMP_STATUS, "equal", {"value": "active"})
        link = Link.inner(
            JoinSpec(AncestorStampMiddleFG, Index((ANCESTOR_STAMP_DESC_K,))),
            JoinSpec(AncestorStampOtherFG, Index((ANCESTOR_STAMP_C_KEY,))),
        )

        result = mloda.run_all(
            [Feature(name=AncestorStampConsumerFG.get_class_name(), initial_requested_data=True)],
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            compute_frameworks=[PyArrowTable, IcebergFramework],
            links={link},
            global_filter=global_filter,
        )

        matching = [
            frame for frame in result if AncestorStampConsumerFG.get_class_name() in getattr(frame, "column_names", [])
        ]
        assert len(matching) == 1
        values = sorted(matching[0][AncestorStampConsumerFG.get_class_name()].to_pylist())
        assert values == ["20|p1", "60|p3"]

    @pytest.mark.parametrize(
        "modes",
        [({ParallelizationMode.SYNC})],
    )
    def test_basic_iceberg_feature_calculation(
        self, modes: set[ParallelizationMode], flight_server: Any, mock_iceberg_catalog: Mock
    ) -> None:
        """Test basic feature calculation with Iceberg framework."""
        # Enable the test feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {IcebergTestDataCreator, IcebergSimpleTransformFeatureGroup}
        )

        # Define features to calculate with catalog connection
        feature_list: Features | list[Feature | str] = [
            Feature(name="doubled_value", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
            Feature(name="score_plus_ten", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
        ]

        data_access_collection = DataAccessCollection(connections={mock_iceberg_catalog})

        # Run with Iceberg framework
        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes=modes,
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks=[IcebergFramework],
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
        plugin_collector = PluginCollector.enabled_feature_groups(
            {IcebergTestDataCreator, IcebergSimpleTransformFeatureGroup, IcebergToArrowFeatureGroup}
        )

        # Define feature that requires transformation between frameworks
        feature_list: Features | list[Feature | str] = [
            Feature(name="arrow_doubled_value", options={"IcebergTestDataCreator": mock_iceberg_catalog})
        ]

        data_access_collection = DataAccessCollection(connections={mock_iceberg_catalog})

        # Run with both Iceberg and PyArrow frameworks
        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks=[PyArrowTable, IcebergFramework],
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
        plugin_collector = PluginCollector.enabled_feature_groups({IcebergTestDataCreator})

        # Request basic features from the data creator
        feature_list: Features | list[Feature | str] = [
            Feature(name="id", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
            Feature(name="value", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
            Feature(name="category", options={"IcebergTestDataCreator": mock_iceberg_catalog}),
        ]

        data_access_collection = DataAccessCollection(connections={mock_iceberg_catalog})

        # Run with Iceberg framework
        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks=[IcebergFramework],
        )

        # Verify results
        final_data = result[0]
        assert isinstance(final_data, pa.Table)
        assert set(final_data.column_names) == {"id", "value", "category"}
