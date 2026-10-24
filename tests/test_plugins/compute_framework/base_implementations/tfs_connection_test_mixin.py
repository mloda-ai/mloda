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
        dac = DataAccessCollection(connections={self.wrong_type_connection()})
        assert framework_class.pick_connection_from_dac(dac) is None

    def test_returns_matching_connection(self, framework_class: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(connections={valid_connection})
        assert framework_class.pick_connection_from_dac(dac) is valid_connection

    def test_classmethod_is_pure(self, framework_class: Any, valid_connection: Any) -> None:
        dac = DataAccessCollection(connections={valid_connection})
        first = framework_class.pick_connection_from_dac(dac)
        second = framework_class.pick_connection_from_dac(dac)
        assert first is valid_connection
        assert second is valid_connection

    def test_raises_on_multiple_matches(
        self, framework_class: Any, valid_connection: Any, second_valid_connection: Any
    ) -> None:
        assert valid_connection is not second_valid_connection
        dac = DataAccessCollection(connections={valid_connection, second_valid_connection})
        with pytest.raises(ValueError, match="Ambiguous resolve") as exc:
            framework_class.pick_connection_from_dac(dac)
        assert f"{framework_class.get_class_name()}: Ambiguous resolve" in str(exc.value)

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

    def test_connection_handle_picks_the_named_connection(
        self, framework_class: Any, valid_connection: Any, second_valid_connection: Any
    ) -> None:
        from mloda.user import Options

        dac = DataAccessCollection(connections={"a": valid_connection, "b": second_valid_connection})
        options = Options(group={"connection_handle": "b"})
        assert framework_class.pick_connection_from_dac(dac, options=options) is second_valid_connection

    # Hooks for the connection-binding tests below. Frameworks without relation-bound
    # connections override the three tests and skip them with a reason.
    missing_connection_prefix: str = "connection object is required"

    def data_on_connection(self, connection: Any) -> Any:
        """Framework-native data (e.g. a relation) living on the given connection."""
        raise NotImplementedError

    def test_adopt_connection_rejects_data_on_another_connection_than_the_bound_one(
        self, framework_class: Any, valid_connection: Any, second_valid_connection: Any
    ) -> None:
        from mloda.user import ParallelizationMode

        fw = framework_class(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.set_framework_connection_object(valid_connection)
        foreign = self.data_on_connection(second_valid_connection)
        with pytest.raises(ValueError, match=f"{framework_class.__name__}.*connection_handle"):
            fw._adopt_connection(foreign)

    def test_adopt_connection_accepts_the_bound_connection_and_adopts_when_unbound(
        self, framework_class: Any, valid_connection: Any
    ) -> None:
        from mloda.user import ParallelizationMode

        fw = framework_class(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw.fallback_connection_object = object()
        data = self.data_on_connection(valid_connection)
        fw._adopt_connection(data)
        assert fw.framework_connection_object is valid_connection
        fw._adopt_connection(data)

    def test_missing_connection_error_names_the_dac_and_connection_handle(self, framework_class: Any) -> None:
        from mloda.user import ParallelizationMode

        fw = framework_class(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        with pytest.raises(
            ValueError, match=f"{self.missing_connection_prefix}.*DataAccessCollection.*connection_handle"
        ):
            fw.set_framework_connection_object(None)
