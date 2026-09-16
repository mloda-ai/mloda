"""Tests for _resolve_pinned_file suffix handling on base vs concrete classes."""

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.utils import escalate_match_abort, is_match_abort
from mloda.user import DataAccessCollection


class ConcreteWithSuffix(BaseInputData):
    """Concrete subclass that implements suffix()."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".csv", ".CSV")

    @classmethod
    def validate_columns(cls, file_path: str, feature_names: list[str]) -> bool:
        return True


class ConcreteNoSuffix(BaseInputData):
    """Subclass where suffix() raises NotImplementedError (like the base)."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise NotImplementedError

    @classmethod
    def validate_columns(cls, file_path: str, feature_names: list[str]) -> bool:
        return True


class ConcreteMarkedAbortSuffix(BaseInputData):
    """Subclass whose suffix() raises a marked abort instead of declaring one."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        raise escalate_match_abort(NotImplementedError("marked abort"))

    @classmethod
    def validate_columns(cls, file_path: str, feature_names: list[str]) -> bool:
        return True


class TestHasSuffix:
    def test_concrete_with_suffix_returns_true(self) -> None:
        assert ConcreteWithSuffix._has_suffix() is True

    def test_base_without_suffix_returns_false(self) -> None:
        assert ConcreteNoSuffix._has_suffix() is False

    def test_marked_abort_suffix_reraises(self) -> None:
        with pytest.raises(NotImplementedError) as excinfo:
            ConcreteMarkedAbortSuffix._has_suffix()

        assert is_match_abort(excinfo.value)


class TestMatchesSuffix:
    def test_concrete_matches_csv(self) -> None:
        assert ConcreteWithSuffix._matches_suffix("/data/file.csv") is True

    def test_concrete_rejects_parquet(self) -> None:
        assert ConcreteWithSuffix._matches_suffix("/data/file.parquet") is False

    def test_base_class_matches_anything(self) -> None:
        assert ConcreteNoSuffix._matches_suffix("/data/file.csv") is True
        assert ConcreteNoSuffix._matches_suffix("/data/file.parquet") is True
        assert ConcreteNoSuffix._matches_suffix("/data/file.json") is True


class TestResolvePinnedFile:
    def test_base_class_resolves_pinned_csv(self) -> None:
        """Base class (no suffix) should resolve pinned files without suffix check."""
        dac = DataAccessCollection(
            files={"/data/customers.csv"},
            column_to_file={"customer_id": "/data/customers.csv"},
        )
        result = ConcreteNoSuffix._resolve_pinned_file(dac, ["customer_id"])
        assert result == "/data/customers.csv"

    def test_concrete_resolves_matching_suffix(self) -> None:
        dac = DataAccessCollection(
            files={"/data/customers.csv"},
            column_to_file={"customer_id": "/data/customers.csv"},
        )
        result = ConcreteWithSuffix._resolve_pinned_file(dac, ["customer_id"])
        assert result == "/data/customers.csv"

    def test_concrete_rejects_wrong_suffix(self) -> None:
        dac = DataAccessCollection(
            files={"/data/customers.parquet"},
            column_to_file={"customer_id": "/data/customers.parquet"},
        )
        result = ConcreteWithSuffix._resolve_pinned_file(dac, ["customer_id"])
        assert result is None

    def test_no_pinned_columns_returns_none(self) -> None:
        dac = DataAccessCollection(
            files={"/data/customers.csv"},
            column_to_file={"other_col": "/data/customers.csv"},
        )
        result = ConcreteWithSuffix._resolve_pinned_file(dac, ["customer_id"])
        assert result is None

    def test_marked_abort_suffix_reraises(self) -> None:
        dac = DataAccessCollection(
            files={"/data/customers.csv"},
            column_to_file={"customer_id": "/data/customers.csv"},
        )
        with pytest.raises(NotImplementedError) as excinfo:
            ConcreteMarkedAbortSuffix._resolve_pinned_file(dac, ["customer_id"])

        assert is_match_abort(excinfo.value)
