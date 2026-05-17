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

    def wrong_type_connection(self) -> Any:
        return object()

    def test_returns_none_with_none(self, framework_class: Any) -> None:
        assert framework_class.pick_connection_from_dac(None) is None

    def test_returns_none_with_empty_collection(self, framework_class: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects=set())
        assert framework_class.pick_connection_from_dac(dac) is None

    def test_returns_none_with_wrong_type(self, framework_class: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects={self.wrong_type_connection()})
        assert framework_class.pick_connection_from_dac(dac) is None

    def test_returns_matching_connection(self, framework_class: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects={valid_connection})
        assert framework_class.pick_connection_from_dac(dac) is valid_connection

    def test_classmethod_is_pure(self, framework_class: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects={valid_connection})
        first = framework_class.pick_connection_from_dac(dac)
        second = framework_class.pick_connection_from_dac(dac)
        assert first is valid_connection
        assert second is valid_connection
