"""Hardening tests for the shared positive-integer predicate (issue #773).

Python bools satisfy ``isinstance(value, int)``, so hand-written positive-int predicates
accepted ``horizon=True`` and ``window_size=True``. The shipped plugins each carried their
own copy with inconsistent bool and numpy-integer behavior; they now share one predicate,
and these tests pin that contract for the predicate itself and for WINDOW_SIZE, HORIZON
and K_VALUE.
"""

from typing import Any

import pytest

from mloda.provider import is_positive_int
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
    DimensionalityReductionFeatureGroup,
)
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

np = pytest.importorskip("numpy")


ACCEPTED: list[Any] = [1, 5, 10**9, "1", "42"]
REJECTED: list[Any] = [
    True,  # bool is the whole point: isinstance(True, int) is True
    False,
    0,
    -1,
    -5,
    2.5,
    "0",
    "-3",
    "abc",
    "",
    "²",  # str.isdigit() accepts it, but int() raises on it
    "1.5",
    None,
    [],
]


class TestIsPositiveInt:
    @pytest.mark.parametrize("value", ACCEPTED)
    def test_accepts_positive_integers(self, value: Any) -> None:
        assert is_positive_int(value) is True

    @pytest.mark.parametrize("value", REJECTED)
    def test_rejects_everything_else(self, value: Any) -> None:
        assert is_positive_int(value) is False

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint32"])
    def test_accepts_numpy_integers(self, dtype: str) -> None:
        assert is_positive_int(np.dtype(dtype).type(7)) is True

    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64"])
    def test_rejects_non_positive_numpy_integers(self, dtype: str) -> None:
        assert is_positive_int(np.dtype(dtype).type(0)) is False

    def test_rejects_numpy_bool(self) -> None:
        """np.bool_ is not numbers.Integral, so it must not slip through as an integer."""
        assert is_positive_int(np.bool_(True)) is False

    @pytest.mark.parametrize("value", ["²", "³", "½"])
    def test_rejects_digit_like_strings_that_int_cannot_parse(self, value: str) -> None:
        """These satisfy str.isdigit()/isnumeric() but raise in int(), so isdecimal() is the right gate."""
        assert is_positive_int(value) is False

    def test_accepts_non_ascii_decimal_digits(self) -> None:
        """'٣' is a genuine decimal digit that int() parses to 3, so it is accepted."""
        assert int("٣") == 3
        assert is_positive_int("٣") is True


class TestPluginKeysShareThePredicate:
    """WINDOW_SIZE, HORIZON and K_VALUE must agree on what a positive integer is."""

    CASES = [
        ("WINDOW_SIZE", TimeWindowFeatureGroup, TimeWindowFeatureGroup.WINDOW_SIZE),
        ("HORIZON", ForecastingFeatureGroup, ForecastingFeatureGroup.HORIZON),
        ("K_VALUE", ClusteringFeatureGroup, ClusteringFeatureGroup.K_VALUE),
        ("DIMENSION", DimensionalityReductionFeatureGroup, DimensionalityReductionFeatureGroup.DIMENSION),
    ]

    @staticmethod
    def _validator(group: Any, key: str) -> Any:
        validator = group.PROPERTY_MAPPING[key].element_validator
        assert validator is not None, f"{key} should declare an element_validator"
        return validator

    @pytest.mark.parametrize("label,group,key", CASES)
    def test_rejects_bool(self, label: str, group: Any, key: str) -> None:
        """The #773 defect: horizon=True and window_size=True were accepted."""
        assert self._validator(group, key)(True) is False
        assert self._validator(group, key)(False) is False

    @pytest.mark.parametrize("label,group,key", CASES)
    def test_accepts_python_and_numpy_integers(self, label: str, group: Any, key: str) -> None:
        validator = self._validator(group, key)
        assert validator(3) is True
        assert validator(np.int64(3)) is True

    @pytest.mark.parametrize("label,group,key", CASES)
    def test_rejects_zero_and_negative(self, label: str, group: Any, key: str) -> None:
        validator = self._validator(group, key)
        assert validator(0) is False
        assert validator(-2) is False

    def test_k_value_still_accepts_auto(self) -> None:
        """Clustering's extra 'auto' value must survive the migration."""
        validator = self._validator(ClusteringFeatureGroup, ClusteringFeatureGroup.K_VALUE)
        assert validator("auto") is True
