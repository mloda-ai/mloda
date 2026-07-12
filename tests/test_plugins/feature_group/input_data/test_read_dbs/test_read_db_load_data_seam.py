"""Tests for the ReadDB.load_data template-method lifecycle seam.

Contract pinned here:

    ReadDB gains
      - classmethod prepare_credentials(data_access, features): overridable, default pass-through.
      - classmethod produce_rows(connection, features): the per-backend row hook; the base
        default raises NotImplementedError.

    load_data becomes a template method that
      1. probes the DYNAMIC anchor first: it resolves cls.final_reader_anchor() and checks the
         anchor's _final_reader_requires() hooks via _is_overridden against THAT anchor (never
         hardcoded against ReadDB); when the hook set is incomplete it raises NotImplementedError
         BEFORE any credential or connection work,
      2. credentials = prepare_credentials(data_access, features)  (default: pass-through),
      3. connection = get_connection(credentials)  (-> connect()),
      4. try: return produce_rows(connection, features)
         finally: close_connection(connection)  (close even when produce_rows raises).

    ReadDB._final_reader_requires() returns ("produce_rows", "connect"): a backend is final iff
    it overrides BOTH hooks relative to its anchor, or overrides load_data wholesale.

    Consistency contract: for any class, is_final_reader() False implies the template raises
    NotImplementedError before connection work.

    The matcher exception contract is documented: the docstrings of is_valid_credentials and
    check_feature_in_data_access mention that only NotImplementedError is treated as a soft
    no-match by the matcher and any other exception propagates.

Test isolation note:
All ReadDB subclasses used by these tests are defined at MODULE scope, never inside test
methods. Function-local subclasses linger in ``ReadDB.__subclasses__()`` (the global plugin
registry) until GC runs, leaking into sibling tests' plugin discovery, and they are
unpicklable, which breaks multiprocessing runners. Module-scope classes are picklable and
stable. Additionally, every seam reader's ``is_valid_credentials`` returns ``False`` so that,
even if discovered, they can never match a real ``DataAccessCollection`` in a sibling test.
"""

from typing import Any, ClassVar

import pytest

from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader


class FakeConnection:
    """In-test stand-in for a DB connection; records ``close()`` calls."""

    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


# --------------------------------------------------------------------------------------
# Module-scope readers for the load_data lifecycle seam tests.
#
# Defined at MODULE scope (never inside a test) so they are picklable and stable in the
# plugin registry. ``is_valid_credentials`` returns False on every reader so a leaked
# class can never match a real DataAccessCollection in a sibling test file. The tests
# below drive these readers directly and record interactions via ClassVar lists.
# --------------------------------------------------------------------------------------


class _SeamIntermediateDB(ReadDB):
    """Implements connect but NOT produce_rows; connect must run only after the probe."""

    connect_calls: ClassVar[list[Any]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.connect_calls.append(credentials)
        return FakeConnection()

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamRowHookNoConnectDB(ReadDB):
    """Overrides produce_rows but NOT connect: cannot actually connect, so it is not final.

    prepare_credentials records calls: if the probe wrongly passes (e.g. a hardcoded
    produce_rows-only check), the template would run prepare_credentials before failing on
    the abstract connect, and the recorder would expose it.
    """

    prepare_calls: ClassVar[list[Any]] = []

    @classmethod
    def prepare_credentials(cls, data_access: Any, features: Any) -> Any:
        cls.prepare_calls.append(data_access)
        return data_access

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return {"rows": True}

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamHappyPathDB(ReadDB):
    """Final reader whose produce_rows returns the supplied connection/features sentinel."""

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        return FakeConnection()

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return {"connection": connection, "features": features}

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamBoomDB(ReadDB):
    """Final reader whose produce_rows raises; close_connection must still run."""

    created: ClassVar[list[FakeConnection]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        connection = FakeConnection()
        cls.created.append(connection)
        return connection

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        raise RuntimeError("boom")

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamConnectRaisesDB(ReadDB):
    """connect() raises before any connection exists: close_connection must never run."""

    closes: ClassVar[list[Any]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        raise RuntimeError("no connect")

    @classmethod
    def close_connection(cls, connection: Any) -> None:
        cls.closes.append(connection)

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        raise AssertionError("produce_rows must not run when connect() raises")

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamTransformDB(ReadDB):
    """Overrides prepare_credentials; the transformed credentials must reach connect()."""

    received: ClassVar[list[Any]] = []

    @classmethod
    def prepare_credentials(cls, data_access: Any, features: Any) -> Any:
        return {"wrapped": data_access}

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.received.append(credentials)
        return FakeConnection()

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return "rows"

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamPassthroughDB(ReadDB):
    """Uses the default prepare_credentials; data_access must pass through unchanged."""

    received: ClassVar[list[Any]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.received.append(credentials)
        return FakeConnection()

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return "rows"

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamStaticRowHookDB(ReadDB):
    """produce_rows is a @staticmethod: the probe must unwrap __func__ uniformly."""

    sentinel: ClassVar[dict[str, bool]] = {"static_rows": True}
    created: ClassVar[list[FakeConnection]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        connection = FakeConnection()
        cls.created.append(connection)
        return connection

    @staticmethod
    def produce_rows(connection: Any, features: Any) -> Any:
        return _SeamStaticRowHookDB.sentinel

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SeamWholesaleDB(ReadDB):
    """Legacy-style backend: overrides load_data wholesale; the template must stay out of the way."""

    connect_calls: ClassVar[list[Any]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.connect_calls.append(credentials)
        return FakeConnection()

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return {"wholesale": True}

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


# --------------------------------------------------------------------------------------
# Third-party sub-family re-anchoring _final_reader_requires below ReadDB.
#
# _SubFamilyDB shares produce_rows for the whole sub-family and redeclares the requires
# tuple, becoming the anchor for its children. Relative to ReadDB its produce_rows IS
# overridden, so a probe hardcoded against ReadDB would wrongly let _SubFamilyDB.load_data
# proceed into credential/connection work. The dynamic probe resolves the anchor
# (_SubFamilyDB itself) and must raise NotImplementedError first.
# --------------------------------------------------------------------------------------


class _SubFamilyDB(ReadDB):
    """Sub-family base: redeclares the requires tuple and provides a shared produce_rows."""

    connect_calls: ClassVar[list[Any]] = []
    prepare_calls: ClassVar[list[Any]] = []

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        return ("produce_rows", "connect")

    @classmethod
    def prepare_credentials(cls, data_access: Any, features: Any) -> Any:
        cls.prepare_calls.append(data_access)
        return data_access

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return {"family": "shared", "connection": connection}

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.connect_calls.append(credentials)
        return FakeConnection()

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _SubFamilyChildDB(_SubFamilyDB):
    """Concrete child overriding both hooks relative to _SubFamilyDB: final, runs the template."""

    created: ClassVar[list[FakeConnection]] = []

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return {"child": True, "connection": connection, "features": features}

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        connection = FakeConnection()
        cls.created.append(connection)
        return connection


class TestReadDBSeamHooks:
    def test_produce_rows_base_default_raises_not_implemented(self) -> None:
        """The base row hook exists on ReadDB and raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ReadDB.produce_rows(None, None)  # type: ignore[arg-type]

    def test_prepare_credentials_default_is_pass_through(self) -> None:
        """The default prepare_credentials returns data_access unchanged (identity)."""
        sentinel = {"host": "db"}
        assert ReadDB.prepare_credentials(sentinel, None) is sentinel  # type: ignore[arg-type]

    def test_read_db_requires_produce_rows_and_connect(self) -> None:
        """ReadDB._final_reader_requires() names the phase-2 hook set."""
        assert ReadDB._final_reader_requires() == ("produce_rows", "connect")


class TestReadDBTemplate:
    def test_seam_happy_path_closes_exactly_once(self) -> None:
        """A hook-complete backend loads via the inherited template and closes once."""
        sentinel_features = ["a", "b"]
        result = _SeamHappyPathDB.load_data({"k": "v"}, sentinel_features)  # type: ignore[arg-type]

        assert result["features"] is sentinel_features
        connection = result["connection"]
        assert isinstance(connection, FakeConnection)
        assert connection.close_count == 1, "close_connection must run exactly once on success"

    def test_close_connection_runs_when_produce_rows_raises(self) -> None:
        """If produce_rows raises, the exception propagates AND the connection is closed."""
        _SeamBoomDB.created.clear()

        with pytest.raises(RuntimeError, match="boom"):
            _SeamBoomDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert _SeamBoomDB.created, "connect() should have been invoked"
        assert _SeamBoomDB.created[-1].close_count == 1, "close_connection must run even when produce_rows raises"

    def test_connect_raising_skips_close_connection(self) -> None:
        """If connect() raises before producing a connection, close_connection must not run."""
        _SeamConnectRaisesDB.closes.clear()

        with pytest.raises(RuntimeError, match="no connect"):
            _SeamConnectRaisesDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert _SeamConnectRaisesDB.closes == [], "close_connection must not run when connect() fails"

    def test_prepare_credentials_override_is_honored(self) -> None:
        """An overridden prepare_credentials transforms data_access before connect()."""
        _SeamTransformDB.received.clear()

        raw = {"host": "db"}
        result = _SeamTransformDB.load_data(raw, None)  # type: ignore[arg-type]

        assert result == "rows"
        assert _SeamTransformDB.received[-1] == {"wrapped": raw}, "transformed credentials must reach connect()"

    def test_default_prepare_credentials_passes_through_unchanged(self) -> None:
        """Without an override, prepare_credentials forwards data_access to connect() unchanged."""
        _SeamPassthroughDB.received.clear()

        raw = {"host": "db2"}
        _SeamPassthroughDB.load_data(raw, None)  # type: ignore[arg-type]

        assert _SeamPassthroughDB.received[-1] is raw, "default prepare_credentials must pass data_access through"

    def test_staticmethod_produce_rows_runs_seam_end_to_end(self) -> None:
        """A @staticmethod row hook is probed via __func__ unwrapping and runs the template."""
        _SeamStaticRowHookDB.created.clear()

        result = _SeamStaticRowHookDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert result is _SeamStaticRowHookDB.sentinel
        assert _SeamStaticRowHookDB.created, "connect() should have been invoked"
        assert _SeamStaticRowHookDB.created[-1].close_count == 1, "close_connection must run exactly once"


class TestReadDBProbeIsDynamic:
    """The template probe resolves the anchor dynamically; it is never hardcoded against ReadDB."""

    def test_sub_family_re_anchors_classification(self) -> None:
        """Redeclaring _final_reader_requires moves the anchor to the sub-family base."""
        assert _SubFamilyDB.final_reader_anchor() is _SubFamilyDB
        assert _SubFamilyChildDB.final_reader_anchor() is _SubFamilyDB

    def test_sub_family_base_is_not_final_and_raises_before_connecting(self) -> None:
        """The sub-family base shares produce_rows but is not final; load_data must probe
        against the re-anchored hook set and raise BEFORE prepare/connect run.

        A probe hardcoded against ReadDB would see produce_rows overridden and connect.
        """
        _SubFamilyDB.connect_calls.clear()
        _SubFamilyDB.prepare_calls.clear()

        assert _SubFamilyDB.is_final_reader() is False

        with pytest.raises(NotImplementedError):
            _SubFamilyDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert _SubFamilyDB.prepare_calls == [], "prepare_credentials must not run before the dynamic probe"
        assert _SubFamilyDB.connect_calls == [], "connect() must not run before the dynamic probe"

    def test_sub_family_child_is_final_and_runs_the_template(self) -> None:
        """A child overriding both hooks relative to the sub-family anchor loads via the template."""
        _SubFamilyChildDB.created.clear()
        _SubFamilyDB.prepare_calls.clear()

        assert _SubFamilyChildDB.is_final_reader() is True

        sentinel_features = ["x"]
        raw = {"k": "v"}
        result = _SubFamilyChildDB.load_data(raw, sentinel_features)  # type: ignore[arg-type]

        assert result["child"] is True
        assert result["features"] is sentinel_features
        assert _SubFamilyDB.prepare_calls == [raw], "the template must run the inherited prepare_credentials"
        assert _SubFamilyChildDB.created, "connect() should have been invoked"
        assert _SubFamilyChildDB.created[-1].close_count == 1, "close_connection must run exactly once"

    @pytest.mark.parametrize(
        "non_final",
        [ReadDB, _SeamIntermediateDB, _SeamRowHookNoConnectDB, _SubFamilyDB],
    )
    def test_non_final_implies_template_raises_not_implemented(self, non_final: type[ReadDB]) -> None:
        """Consistency contract: is_final_reader() False implies load_data raises NotImplementedError."""
        assert non_final.is_final_reader() is False

        with pytest.raises(NotImplementedError):
            non_final.load_data(None, None)  # type: ignore[arg-type]

    def test_intermediate_base_probe_raises_without_connecting(self) -> None:
        """A base implementing connect but NOT produce_rows must fail the probe before connecting."""
        _SeamIntermediateDB.connect_calls.clear()

        with pytest.raises(NotImplementedError):
            _SeamIntermediateDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert _SeamIntermediateDB.connect_calls == [], "connect() must not run when the probe fails"

    def test_row_hook_without_connect_probe_raises_before_credential_work(self) -> None:
        """A produce_rows-only class fails the probe before prepare_credentials runs."""
        _SeamRowHookNoConnectDB.prepare_calls.clear()

        with pytest.raises(NotImplementedError):
            _SeamRowHookNoConnectDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert _SeamRowHookNoConnectDB.prepare_calls == [], "prepare_credentials must not run when the probe fails"


class TestReadDBClassification:
    def test_read_db_itself_is_not_final(self) -> None:
        assert ReadDB.is_final_reader() is False

    def test_hook_complete_backend_is_final_without_load_data_override(self) -> None:
        """Overriding produce_rows + connect is enough; no wholesale load_data needed."""
        assert "load_data" not in _SeamHappyPathDB.__dict__
        assert _SeamHappyPathDB.is_final_reader() is True

    def test_staticmethod_row_hook_backend_is_final(self) -> None:
        assert _SeamStaticRowHookDB.is_final_reader() is True

    def test_intermediate_base_with_connect_only_is_not_final(self) -> None:
        assert _SeamIntermediateDB.is_final_reader() is False

    def test_row_hook_without_connect_is_not_final(self) -> None:
        """connect is in the requires tuple precisely to screen out this shape."""
        assert _SeamRowHookNoConnectDB.is_final_reader() is False

    def test_wholesale_load_data_override_is_final_and_untouched(self) -> None:
        """A wholesale override stays final and runs without the template lifecycle."""
        _SeamWholesaleDB.connect_calls.clear()

        assert _SeamWholesaleDB.is_final_reader() is True
        assert _SeamWholesaleDB.load_data({"k": "v"}, None) == {"wholesale": True}
        assert _SeamWholesaleDB.connect_calls == [], "a wholesale load_data must not enter the template"

    def test_sqlite_reader_stays_final_via_wholesale_override(self) -> None:
        assert "load_data" in SQLITEReader.__dict__
        assert SQLITEReader.is_final_reader() is True


class TestMatcherExceptionContractIsDocumented:
    """Only NotImplementedError is a soft no-match in the matcher; other exceptions propagate.

    The contract must be stated in the docstrings of the two matcher-facing hooks.
    """

    def test_is_valid_credentials_docstring_mentions_not_implemented_error(self) -> None:
        assert "NotImplementedError" in (ReadDB.is_valid_credentials.__doc__ or "")

    def test_check_feature_in_data_access_docstring_mentions_not_implemented_error(self) -> None:
        assert "NotImplementedError" in (ReadDB.check_feature_in_data_access.__doc__ or "")
