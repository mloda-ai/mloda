"""The flight server hands transform a pa.Table whatever the source framework is, so transform must accept it."""

from uuid import uuid4

import pandas as pd
import pyarrow as pa

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

import mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer  # noqa: F401
import mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer  # noqa: F401


COLUMN_NAMES = {"flight_data_key", "flight_data_payload"}
KEYS = [1, 2, 3]
PAYLOADS = ["a", "b", "c"]


class FlightDataMockFeatureGroup(FeatureGroup):
    """Placeholder feature group; the transform hop never touches it."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _step(from_framework: type[ComputeFramework], to_framework: type[ComputeFramework]) -> TransformFrameworkStep:
    return TransformFrameworkStep(
        from_framework=from_framework,
        to_framework=to_framework,
        required_uuids={uuid4()},
        from_feature_group=FlightDataMockFeatureGroup,
        to_feature_group=FlightDataMockFeatureGroup,
    )


def _cfw(framework: type[ComputeFramework], mode: ParallelizationMode) -> ComputeFramework:
    return framework(mode, frozenset(), uuid4())


def _flight_table() -> pa.Table:
    return pa.table({"flight_data_key": KEYS, "flight_data_payload": PAYLOADS})


def test_python_dict_to_pandas_accepts_a_flight_table() -> None:
    step = _step(PythonDictFramework, PandasDataFrame)
    cfw = _cfw(PandasDataFrame, ParallelizationMode.MULTIPROCESSING)

    result = step.transform(cfw, _flight_table(), COLUMN_NAMES)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == COLUMN_NAMES
    assert list(result["flight_data_key"]) == KEYS
    assert list(result["flight_data_payload"]) == PAYLOADS


def test_pandas_to_python_dict_accepts_a_flight_table() -> None:
    step = _step(PandasDataFrame, PythonDictFramework)
    cfw = _cfw(PythonDictFramework, ParallelizationMode.MULTIPROCESSING)

    result = step.transform(cfw, _flight_table(), COLUMN_NAMES)

    assert result == {"flight_data_key": KEYS, "flight_data_payload": PAYLOADS}


def test_pandas_to_pyarrow_passes_a_flight_table_through_unchanged() -> None:
    step = _step(PandasDataFrame, PyArrowTable)
    cfw = _cfw(PyArrowTable, ParallelizationMode.MULTIPROCESSING)
    data = _flight_table()

    result = step.transform(cfw, data, COLUMN_NAMES)

    assert result is data


def test_pyarrow_source_still_transforms_a_flight_table() -> None:
    step = _step(PyArrowTable, PandasDataFrame)
    cfw = _cfw(PandasDataFrame, ParallelizationMode.MULTIPROCESSING)

    result = step.transform(cfw, _flight_table(), COLUMN_NAMES)

    assert isinstance(result, pd.DataFrame)
    assert list(result["flight_data_key"]) == KEYS
    assert list(result["flight_data_payload"]) == PAYLOADS


def test_native_pandas_to_pyarrow_still_transforms() -> None:
    step = _step(PandasDataFrame, PyArrowTable)
    cfw = _cfw(PyArrowTable, ParallelizationMode.SYNC)
    frame = pd.DataFrame({"flight_data_key": KEYS, "flight_data_payload": PAYLOADS})

    result = step.transform(cfw, frame, COLUMN_NAMES)

    assert isinstance(result, pa.Table)
    assert result.column("flight_data_key").to_pylist() == KEYS
    assert result.column("flight_data_payload").to_pylist() == PAYLOADS


def test_equal_expected_frameworks_return_native_data_unchanged() -> None:
    step = _step(PandasDataFrame, PandasDataFrame)
    cfw = _cfw(PandasDataFrame, ParallelizationMode.SYNC)
    frame = pd.DataFrame({"flight_data_key": KEYS, "flight_data_payload": PAYLOADS})

    result = step.transform(cfw, frame, COLUMN_NAMES)

    assert result is frame
