from abc import abstractmethod
from typing import Any

import pytest

from mloda.user import DataAccessCollection, ParallelizationMode


class TfsConnectionInitMixin:
    @pytest.fixture
    @abstractmethod
    def framework_instance(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def valid_connection(self) -> Any:
        raise NotImplementedError

    def wrong_type_connection(self) -> Any:
        return object()

    def test_no_op_with_none(self, framework_instance: Any) -> None:
        framework_instance.init_connection_from_data_access(None)
        assert framework_instance.framework_connection_object is None

    def test_no_op_with_empty_collection(self, framework_instance: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects=set())
        framework_instance.init_connection_from_data_access(dac)
        assert framework_instance.framework_connection_object is None

    def test_no_op_with_wrong_type(self, framework_instance: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects={self.wrong_type_connection()})
        framework_instance.init_connection_from_data_access(dac)
        assert framework_instance.framework_connection_object is None

    def test_sets_correct_type(self, framework_instance: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects={valid_connection})
        framework_instance.init_connection_from_data_access(dac)
        assert framework_instance.framework_connection_object is valid_connection

    def test_idempotent_same_connection(self, framework_instance: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(initialized_connection_objects={valid_connection})
        framework_instance.init_connection_from_data_access(dac)
        framework_instance.init_connection_from_data_access(dac)  # must not raise
        assert framework_instance.framework_connection_object is valid_connection
