"""Missing value family zero-row contract: every imputation method yields a typed result column with zero rows."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.user import Feature, Options
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import ZeroRowResultTypeTestMixin

IMPUTATION_METHODS = sorted(MissingValueFeatureGroup.IMPUTATION_METHODS)

ZERO_ROWS = pa.table({"income": pa.array([], type=pa.int64()), "temperature": pa.array([], type=pa.float64())})

# Only constant imputation reads constant_value; the other methods ignore it.
CONSTANT_VALUES: dict[str, Any] = {"income": 0, "temperature": 0.0}


class MissingValueZeroRowTestMixin(ZeroRowResultTypeTestMixin):
    """Zero-row int64 and float64 sources, for every imputation method."""

    @pytest.mark.parametrize("source", ["income", "temperature"])
    @pytest.mark.parametrize("imputation_method", IMPUTATION_METHODS)
    def test_zero_rows_result_is_typed(self, imputation_method: str, source: str) -> None:
        feature_name = f"{source}__{imputation_method}_imputed"
        feature = Feature(feature_name, options=Options(context={"constant_value": CONSTANT_VALUES[source]}))
        self.assert_typed(self.calculate(ZERO_ROWS, feature), feature_name, 0)
