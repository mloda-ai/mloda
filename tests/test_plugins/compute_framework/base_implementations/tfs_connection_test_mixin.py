from abc import abstractmethod
from typing import Any

import pytest

from mloda.user import DataAccessCollection


class TfsConnectionInitMixin:
    """Unit-level coverage for pick_connection_from_dac on a SQL ComputeFramework.

    Subclasses provide the framework class and a fixture yielding a valid native connection.
    """

    @pytest.fixture
    @abstractmethod
    def framework_class(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def valid_connection(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def second_valid_connection(self) -> Any:
        """A second, distinct connection of the same framework type.

        Used to exercise the multi-match disambiguation policy. Subclasses must
        ensure this is *not* the same object as `valid_connection` so both end
        up in the DAC set.
        """
        raise NotImplementedError

    def wrong_type_connection(self) -> Any:
        return object()

    def test_returns_none_with_none(self, framework_class: Any) -> None:
        assert framework_class.pick_connection_from_dac(None) is None

    def test_returns_none_with_empty_collection(self, framework_class: Any) -> None:
        dac = DataAccessCollection(connections={})
        assert framework_class.pick_connection_from_dac(dac) is None

    def test_returns_none_with_wrong_type(self, framework_class: Any) -> None:
        dac = DataAccessCollection(connections={"wrong": self.wrong_type_connection()})
        assert framework_class.pick_connection_from_dac(dac) is None

    def test_returns_matching_connection(self, framework_class: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(connections={"primary": valid_connection})
        assert framework_class.pick_connection_from_dac(dac) is valid_connection

    def test_classmethod_is_pure(self, framework_class: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(connections={"primary": valid_connection})
        first = framework_class.pick_connection_from_dac(dac)
        second = framework_class.pick_connection_from_dac(dac)
        assert first is valid_connection
        assert second is valid_connection

    def test_raises_on_multiple_matches(
        self, framework_class: Any, valid_connection: Any, second_valid_connection: Any
    ) -> None:
        assert valid_connection is not second_valid_connection
        dac = DataAccessCollection(connections={"primary": valid_connection, "secondary": second_valid_connection})
        with pytest.raises(ValueError, match="Ambiguous resolve"):
            framework_class.pick_connection_from_dac(dac)

    def test_hint_resolves_ambiguity(
        self, framework_class: Any, valid_connection: Any, second_valid_connection: Any
    ) -> None:
        """A ``data_access_handle`` hint on Options disambiguates between
        multiple framework-matching connections in the DAC.
        """
        from mloda.user import Options

        assert valid_connection is not second_valid_connection
        dac = DataAccessCollection(
            connections={"primary": valid_connection, "secondary": second_valid_connection},
        )
        options = Options(context={"data_access_handle": "secondary"})
        resolved = framework_class.pick_connection_from_dac(dac, options=options)
        assert resolved is second_valid_connection
