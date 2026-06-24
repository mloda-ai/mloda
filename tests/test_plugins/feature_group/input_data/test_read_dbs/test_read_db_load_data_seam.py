"""Tests for the ReadDB.load_data template-method lifecycle seam (issue #535).

These pin the planned contract:

    load_data is a template method that
      1. probes the per-backend row hook ``produce_rows``: if it is still the
         ReadDB base default, raise NotImplementedError IMMEDIATELY (before any
         prepare/connect step),
      2. credentials = prepare_credentials(data_access, features)  (default: pass-through),
      3. connection = get_connection(credentials)  (-> connect()),
      4. try: return produce_rows(connection, features)
         finally: close_connection(connection).

Downstream connector packages should only need to override ``produce_rows``
(plus ``connect``/credential helpers), not re-implement ``load_data``.

Test isolation note (issue #535):
All ReadDB subclasses used by these tests are defined at MODULE scope, never
inside test methods. Function-local subclasses linger in ``ReadDB.__subclasses__()``
(the global plugin registry) until GC runs, leaking into sibling tests' plugin
discovery, and they are unpicklable -- which deadlocks the multiprocessing reader
tests. Module-scope classes are picklable and stable. Additionally, every seam
reader's ``is_valid_credentials`` returns ``False`` so that, even if discovered,
they can never *match* a real ``DataAccessCollection`` in a sibling test. The seam
tests call ``load_data`` / ``supports_scoped_data_access`` / ``prepare_credentials``
directly, so a ``False`` credential check keeps every assertion valid.

The lone exception is ``_RowHookNoConnectDB``, which intentionally returns ``True``
from ``is_valid_credentials`` -- but only for an impossible sentinel dict that no
real ``DataAccessCollection`` produces, and the class is classified non-final
(``supports_scoped_data_access() is False``) so discovery never even reaches it.
"""

from typing import Any, ClassVar

import pytest

from mloda_plugins.feature_group.input_data.read_db import ReadDB
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader


class FakeConnection:
    """In-test stand-in for a DB connection; records ``close()`` calls."""

    def __init__(self, label: str = "conn") -> None:
        self.label = label
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


class TestReadDBLoadDataSeam:
    def test_abstract_read_db_probe_stays_false(self) -> None:
        """Abstract ReadDB has no row hook: load_data raises and probe is False."""
        with pytest.raises(NotImplementedError):
            ReadDB.load_data(None, None)  # type: ignore[arg-type]

        assert ReadDB.supports_scoped_data_access() is False

    def test_intermediate_base_probe_false_without_connecting(self) -> None:
        """A base that implements connect but NOT produce_rows is not a final class.

        The row hook must be probed *before* connecting, so connect() must never
        run when produce_rows is still the base default.
        """
        _SeamIntermediateDB.connect_calls.clear()

        assert _SeamIntermediateDB.supports_scoped_data_access() is False

        with pytest.raises(NotImplementedError):
            _SeamIntermediateDB.load_data(None, None)  # type: ignore[arg-type]

        assert _SeamIntermediateDB.connect_calls == [], "connect() must not run before the row hook is probed"

    def test_seam_happy_path(self) -> None:
        """A reader overriding produce_rows is final: load_data runs the seam and closes."""
        assert _SeamHappyPathDB.supports_scoped_data_access() is True

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

        assert _SeamPassthroughDB.received[-1] is raw, (
            "default prepare_credentials must pass data_access through unchanged"
        )


# --------------------------------------------------------------------------------------
# Module-scope readers for the "probe must not run the lifecycle" contract (issue #535).
#
# These are defined at MODULE scope on purpose so they are created exactly once and are
# stable across the test session (no per-test subclass churn). The structural
# supports_scoped_data_access() that ReadDB will override must classify these WITHOUT
# executing connect / produce_rows / prepare_credentials / get_connection, so a leaked
# subclass can never trigger real I/O or raise during plugin discovery.
#
# is_valid_credentials returns False here so that, once classified as final readers,
# they never *match* a real DataAccessCollection during discovery in sibling test files
# (e.g. test_read_db_multiprocessing.py) -- keeping the combined run pollution-free.
# --------------------------------------------------------------------------------------


class _ProbeReader(ReadDB):
    """Final reader (overrides produce_rows). Records whether the lifecycle ran."""

    connect_calls: ClassVar[list[Any]] = []
    produce_rows_calls: ClassVar[list[Any]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.connect_calls.append(credentials)
        return FakeConnection()

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        cls.produce_rows_calls.append((connection, features))
        return {"rows": True}

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _BoomProbeReader(ReadDB):
    """Final reader whose row hook explodes; the probe must never invoke it."""

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        return FakeConnection()

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        raise RuntimeError("boom")

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _IntermediateProbeDB(ReadDB):
    """Not a final reader: implements connect + is_valid_credentials but NOT produce_rows."""

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        return FakeConnection()

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class TestSupportsScopedDataAccessIsSideEffectFree:
    """supports_scoped_data_access() must classify readers structurally, not by execution."""

    def test_probe_does_not_run_connect_or_produce_rows(self) -> None:
        """A seam reader is final, but probing it must NOT execute the lifecycle.

        Fails today: the base probe calls load_data(None, None), which runs
        connect() + produce_rows(), incrementing the recorders.
        """
        _ProbeReader.connect_calls.clear()
        _ProbeReader.produce_rows_calls.clear()

        is_final = _ProbeReader.supports_scoped_data_access()

        assert is_final is True
        assert _ProbeReader.connect_calls == [], "probe must not call connect()"
        assert _ProbeReader.produce_rows_calls == [], "probe must not call produce_rows()"

    def test_probe_does_not_raise_when_produce_rows_would_raise(self) -> None:
        """Probing a reader whose row hook raises must still return True, not propagate.

        Fails today: the base probe executes produce_rows(), so RuntimeError("boom")
        escapes supports_scoped_data_access() and crashes discovery.
        """
        is_final = _BoomProbeReader.supports_scoped_data_access()

        assert is_final is True

    def test_abstract_and_intermediate_bases_stay_false(self) -> None:
        """Regression guard: classes without a row hook are not final readers."""
        assert ReadDB.supports_scoped_data_access() is False
        assert _IntermediateProbeDB.supports_scoped_data_access() is False

    def test_wholesale_load_data_override_is_final(self) -> None:
        """Regression guard: overriding load_data wholesale still counts as a final reader."""
        assert SQLITEReader.supports_scoped_data_access() is True


# --------------------------------------------------------------------------------------
# Module-scope readers pinning the review-gap fixes (issue #535 follow-up).
#
# These exercise override-identity comparisons that must unwrap ``__func__`` uniformly
# (via the planned ReadDB._underlying helper) so the probe works whether produce_rows is
# a @classmethod, @staticmethod, or plain def, and pin that a row hook alone (without a
# connect override) is NOT enough to be a final scoped reader.
# --------------------------------------------------------------------------------------


class _StaticRowHookReader(ReadDB):
    """produce_rows is a @staticmethod: ``cls.produce_rows.__func__`` raises AttributeError today.

    The override-identity probe must unwrap via getattr(member, "__func__", member) so a
    staticmethod row hook is recognised without crashing supports_scoped_data_access()/load_data.
    """

    sentinel: ClassVar[dict[str, bool]] = {"static_rows": True}
    created: ClassVar[list[FakeConnection]] = []

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        connection = FakeConnection()
        cls.created.append(connection)
        return connection

    @staticmethod
    def produce_rows(connection: Any, features: Any) -> Any:
        return _StaticRowHookReader.sentinel

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return False


class _RowHookNoConnectDB(ReadDB):
    """Overrides produce_rows but NOT connect: cannot actually connect, so it is not final.

    A reader is "final via the seam" only if BOTH produce_rows AND connect are overridden.
    is_valid_credentials returns True ONLY for an impossible sentinel dict so a leaked copy
    can never match a real DataAccessCollection (and it is classified non-final anyway).
    """

    @classmethod
    def produce_rows(cls, connection: Any, features: Any) -> Any:
        return {"rows": True}

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        return credentials == {"_rowhooknoconnect_probe": True}


class _ConnectRaisesDB(ReadDB):
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


class TestReadDBSeamReviewGaps:
    """Pins the two-reviewer findings on the ReadDB.load_data seam (issue #535)."""

    def test_staticmethod_produce_rows_probe_does_not_crash(self) -> None:
        """A @staticmethod produce_rows override is a final reader and must not crash the probe.

        Fails today: ``cls.produce_rows.__func__`` raises AttributeError because a
        staticmethod accessed via the class is a plain function with no ``__func__``.
        """
        assert _StaticRowHookReader.supports_scoped_data_access() is True

    def test_staticmethod_produce_rows_runs_seam_end_to_end(self) -> None:
        """load_data probes the staticmethod row hook, runs the seam, and closes once.

        Fails today for the same AttributeError reason as the probe above (the load_data
        row-hook probe also does ``cls.produce_rows.__func__``).
        """
        _StaticRowHookReader.created.clear()

        result = _StaticRowHookReader.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert result is _StaticRowHookReader.sentinel
        assert _StaticRowHookReader.created, "connect() should have been invoked"
        assert _StaticRowHookReader.created[-1].close_count == 1, "close_connection must run exactly once"

    def test_row_hook_without_connect_is_not_final(self) -> None:
        """A row hook without a connect override cannot connect, so it must be screened out.

        Fails today: the structural check returns True from the produce_rows override
        identity alone, ignoring that connect() is still the abstract base default.
        """
        assert _RowHookNoConnectDB.supports_scoped_data_access() is False

    def test_connect_raising_skips_close_connection(self) -> None:
        """If connect() raises before producing a connection, close_connection must not run.

        The try/finally guards only produce_rows; a failed connect cannot reach close.
        """
        _ConnectRaisesDB.closes.clear()

        with pytest.raises(RuntimeError, match="no connect"):
            _ConnectRaisesDB.load_data({"k": "v"}, None)  # type: ignore[arg-type]

        assert _ConnectRaisesDB.closes == [], "close_connection must not run when connect() fails"
