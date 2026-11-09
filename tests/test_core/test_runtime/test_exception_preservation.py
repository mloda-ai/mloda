"""Failing tests for GitHub issue #563: preserve original exception type and cause.

``mlodaAPI.run_all`` currently wraps any worker-step exception in a bare
``Exception(exc_info, msg)`` (see ``mloda/core/runtime/run.py`` ->
``ExecutionOrchestrator._check_for_error``). This drops the original exception
TYPE and the ``__cause__`` chain, so a caller cannot write
``except ImportError:`` / ``except ValueError:`` around ``run_all``.

These tests define the intended behavior:

* The specific stdlib / custom exception type raised inside ``calculate_feature``
  must survive out of ``run_all`` (SYNC and THREADING).
* The ``__cause__`` chain must be preserved.
* A typed fallback ``MlodaRunError`` must exist and be raised when no original
  exception object was captured (e.g. the internal ``error_out`` critical path).

They FAIL today because ``run_all`` surfaces a bare ``Exception`` (wrong type)
and because ``MlodaRunError`` does not yet exist.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import errno
import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.validators.missing_input_validator import MissingInputFeatureError
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import (
    BaseArtifact,
    CredentialFieldMissing,
    CredentialReadRefused,
    DataCreatorFeatureGroup,
    FeatureSet,
    SourceMatch,
    credential_handle_spec,
    read_credential,
    report_failed_rows,
)
from mloda.steward import GateBypassError
from mloda.user import Credential, Feature, ParallelizationMode, PluginCollector, mloda

from tests.helpers.credential_handle_stubs import HANDLE_KEY, ChsSource, ChsTwoHandleFG
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    TOY_TOKEN,
    ToySourceFG,
    toy_dac,
    toy_token_request,
)

from tests.helpers.lost_input_feature_groups import (
    SiblingDictAggregate,
    SiblingDictLostInputConsumer,
    SiblingDictRoot,
)

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.credential_grant import CredentialLookup

# Importing the framework registers it as a ComputeFramework subclass so
# ``compute_frameworks=["PythonDictFramework"]`` resolves during ``run_all``.
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (  # noqa: F401
    PythonDictFramework,
)

# --------------------------------------------------------------------------- #
# Minimal root FeatureGroups whose calculate_feature raises a chosen error type
# --------------------------------------------------------------------------- #


class ImportErrorFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` raises a stdlib ``ImportError``."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ImportError("optional backend 'bm25s' missing")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_import_error_col"}


class MyDomainError(ValueError):
    """Custom domain error that is also a ``ValueError`` subclass."""


class DomainErrorFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` raises a ``ValueError`` subclass."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise MyDomainError("domain rule violated")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_domain_error_col"}


class CauseChainFeatureGroup(DataCreatorFeatureGroup):
    """Root FG that raises ``ValueError`` from a ``KeyError`` (``__cause__`` chain)."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError("boom") from KeyError("root")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_cause_chain_col"}


_ENABLED_IMPORT_ERROR = PluginCollector.enabled_feature_groups({ImportErrorFeatureGroup})
_ENABLED_DOMAIN_ERROR = PluginCollector.enabled_feature_groups({DomainErrorFeatureGroup})
_ENABLED_CAUSE_CHAIN = PluginCollector.enabled_feature_groups({CauseChainFeatureGroup})

_RUNNER_LOGGER = "mloda.core.runtime.run"


def _assert_scrubbed_debug_traceback(caplog: pytest.LogCaptureFixture, marker: str) -> None:
    records = [r for r in caplog.records if r.name == _RUNNER_LOGGER and "Traceback" in r.getMessage()]
    assert records
    assert all(r.levelno == logging.DEBUG for r in records)
    assert not any(r.name == _RUNNER_LOGGER and r.levelno >= logging.ERROR for r in caplog.records)
    assert marker not in caplog.text


# --------------------------------------------------------------------------- #
# Credential scrubbing: failure logs and MlodaRunError must never carry secrets,
# while the exception raised to the caller keeps its raw, unscrubbed message.
#
# The secret value itself, not "secret"/"leak", is the only string the assertions
# below search for, so it must never also appear in an identifier, comment or
# source line: a raw traceback embeds the source line of the raise statement,
# and an identifier match there would be a false positive, not a real leak.
# --------------------------------------------------------------------------- #

_LEAK_MARKER = "hunter2z9"
_LEAK_PRESIGNED_URL = f"https://bucket.s3.amazonaws.com/key?X-Amz-Signature={_LEAK_MARKER}"
_LEAK_USERINFO_URL = f"postgres://user:{_LEAK_MARKER}@host:5432/db"
_LEAK_DSN = f"host=h password={_LEAK_MARKER} dbname=d"
_LEAK_MESSAGE = f"failed for {_LEAK_PRESIGNED_URL} and {_LEAK_USERINFO_URL} and {_LEAK_DSN}"


class SecretLeakFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` raises an ``OSError`` carrying three credential shapes."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise OSError(_LEAK_MESSAGE)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_secret_leak_col"}


class SecretLeakChainedFeatureGroup(DataCreatorFeatureGroup):
    """Root FG that raises ``RuntimeError`` from an ``OSError`` carrying the same credential shapes."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("load failed") from OSError(_LEAK_MESSAGE)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_secret_leak_chained_col"}


class UnpicklableSecretLeakError(RuntimeError):
    """A RuntimeError with a non-picklable payload whose message also carries a secret."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.lock = threading.Lock()  # not picklable


class UnpicklableSecretLeakFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` raises a non-picklable, secret-bearing exception."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise UnpicklableSecretLeakError(_LEAK_MESSAGE)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_unpicklable_secret_col"}


_ENABLED_SECRET_LEAK = PluginCollector.enabled_feature_groups({SecretLeakFeatureGroup})
_ENABLED_SECRET_LEAK_CHAINED = PluginCollector.enabled_feature_groups({SecretLeakChainedFeatureGroup})
_ENABLED_UNPICKLABLE_SECRET_LEAK = PluginCollector.enabled_feature_groups({UnpicklableSecretLeakFeatureGroup})


_ROW_LEAK_MARKER = "Jane Doe, DOB 1990"


class RowLeakError(ValueError):
    """A specific error type for testing row leakage."""


class RowLeakFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` raises an error carrying row values."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RowLeakError(f"Failed parsing row: {_ROW_LEAK_MARKER}")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_row_leak_col"}


class RowLeakChainedFeatureGroup(DataCreatorFeatureGroup):
    """Root FG that raises ``RuntimeError`` from a ``RowLeakError``."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("Calculation failed") from RowLeakError(f"Inner row error: {_ROW_LEAK_MARKER}")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_row_leak_chained_col"}


_ENABLED_ROW_LEAK = PluginCollector.enabled_feature_groups({RowLeakFeatureGroup})
_ENABLED_ROW_LEAK_CHAINED = PluginCollector.enabled_feature_groups({RowLeakChainedFeatureGroup})


def test_sync_secret_leak_direct_raw_to_caller_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The caller keeps the raw secret-bearing message; the DEBUG traceback is scrubbed."""
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(OSError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert _LEAK_MARKER in str(excinfo.value)
    _assert_scrubbed_debug_traceback(caplog, _LEAK_MARKER)


def test_sync_secret_leak_chained_raw_to_caller_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The chained cause keeps its raw secret-bearing message; the DEBUG traceback is scrubbed."""
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(RuntimeError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_chained_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK_CHAINED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert isinstance(excinfo.value.__cause__, OSError)
    assert _LEAK_MARKER in str(excinfo.value.__cause__)
    _assert_scrubbed_debug_traceback(caplog, _LEAK_MARKER)


def test_threading_secret_leak_scrubbed_in_logs(flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(OSError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.THREADING},
            flight_server=flight_server,
        )

    assert _LEAK_MARKER in str(excinfo.value)
    _assert_scrubbed_debug_traceback(caplog, _LEAK_MARKER)


def test_multiprocessing_secret_leak_scrubbed_in_logs(flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(OSError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

    assert _LEAK_MARKER in str(excinfo.value)
    _assert_scrubbed_debug_traceback(caplog, _LEAK_MARKER)


@pytest.mark.timeout(15)
def test_multiprocessing_unpicklable_secret_leak_message_is_scrubbed(
    flight_server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The MlodaRunError fallback message for a non-picklable exception must not carry the secret."""
    from mloda.core.abstract_plugins.components.error_utils import MlodaRunError

    with pytest.raises(MlodaRunError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_unpicklable_secret_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_UNPICKLABLE_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

    assert _LEAK_MARKER not in str(excinfo.value)
    assert "UnpicklableSecretLeakError" in str(excinfo.value)
    error_records = [r for r in caplog.records if r.name == _RUNNER_LOGGER and r.levelno == logging.ERROR]
    assert error_records
    assert all(_LEAK_MARKER not in r.getMessage() for r in error_records)


# --------------------------------------------------------------------------- #
# 1. SYNC mode type preservation
# --------------------------------------------------------------------------- #


def test_sync_preserves_import_error_type(caplog: pytest.LogCaptureFixture) -> None:
    """SYNC preserves ImportError and logs its traceback once, at DEBUG on the runner logger."""
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(ImportError, match="bm25s"):
        mloda.run_all(
            [Feature(name="exc_import_error_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_IMPORT_ERROR,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert [r.name for r in caplog.records if r.name == "root"] == []
    traceback_records = [r for r in caplog.records if "Traceback" in r.getMessage()]
    assert [(r.name, r.levelno) for r in traceback_records] == [(_RUNNER_LOGGER, logging.DEBUG)]


# --------------------------------------------------------------------------- #
# 2. THREADING mode type preservation
# --------------------------------------------------------------------------- #


def test_threading_preserves_import_error_type(flight_server: Any) -> None:
    """THREADING ``run_all`` must surface the original ``ImportError`` type.

    FAILS today: the worker error is re-wrapped as a bare ``Exception``.
    """
    with pytest.raises(ImportError, match="bm25s"):
        mloda.run_all(
            [Feature(name="exc_import_error_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_IMPORT_ERROR,
            parallelization_modes={ParallelizationMode.THREADING},
            flight_server=flight_server,
        )


# --------------------------------------------------------------------------- #
# 3. Custom ValueError subclass -> the core "except ImportError works" repro
# --------------------------------------------------------------------------- #


def test_sync_preserves_valueerror_subclass() -> None:
    """A ``ValueError`` subclass must survive so ``except ValueError`` also catches it.

    FAILS today: a bare ``Exception`` is raised, which is not a ``MyDomainError``
    nor a ``ValueError``.
    """
    with pytest.raises(MyDomainError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_domain_error_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_DOMAIN_ERROR,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    # The whole point of the issue: broad `except ValueError` must catch it too.
    assert isinstance(excinfo.value, ValueError)


# --------------------------------------------------------------------------- #
# 4. __cause__ chain preservation
# --------------------------------------------------------------------------- #


def test_sync_preserves_cause_chain() -> None:
    """The surfaced exception must keep its ``__cause__`` (a ``KeyError`` here).

    FAILS today: the original exception object (with its ``__cause__``) is
    discarded; a bare ``Exception`` built from a traceback string is raised.
    """
    with pytest.raises(ValueError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_cause_chain_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_CAUSE_CHAIN,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert isinstance(excinfo.value.__cause__, KeyError)


# --------------------------------------------------------------------------- #
# 5. MlodaRunError exists and is the typed fallback
# --------------------------------------------------------------------------- #


def test_mloda_run_error_exists_and_is_exception_subclass() -> None:
    """``MlodaRunError`` must exist and be an ``Exception`` subclass.

    FAILS today with ImportError: the class does not exist yet.
    """
    from mloda.core.abstract_plugins.components.error_utils import MlodaRunError

    assert issubclass(MlodaRunError, Exception)


def test_check_for_error_raises_mloda_run_error_when_no_exception_captured() -> None:
    """``_check_for_error`` must raise the typed ``MlodaRunError`` fallback.

    When a run reports an error but no original exception object was captured
    (``take_error_exception()`` returns ``None`` -- e.g. the internal critical
    ``error_out`` path), the loop must raise ``MlodaRunError`` rather than a bare
    ``Exception``.

    FAILS today with ImportError: ``MlodaRunError`` does not exist. Once it does,
    this pins that the fallback branch raises the typed error.
    """
    from unittest.mock import MagicMock, Mock

    from mloda.core.abstract_plugins.components.error_utils import MlodaRunError
    from mloda.core.prepare.execution_plan import ExecutionPlan
    from mloda.core.runtime.run import ExecutionOrchestrator

    orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))

    cfw_register = MagicMock()
    cfw_register.get_error.return_value = True
    cfw_register.take_error_exception.return_value = None
    cfw_register.get_error_msg.return_value = "critical error_out"
    cfw_register.get_error_exc_info.return_value = "critical error_out"
    orchestrator.cfw_register = cfw_register

    with pytest.raises(MlodaRunError):
        orchestrator._check_for_error()


# --------------------------------------------------------------------------- #
# 6. MULTIPROCESSING: picklable vs non-picklable worker exceptions (issue #563
#    regression guard). In MULTIPROCESSING mode ``cfw_register`` is a
#    ``multiprocessing.managers`` proxy, so ``set_error(exception=e)`` PICKLES
#    the exception. A picklable exception type must survive (already works). A
#    NON-picklable exception must NOT hang the orchestrator loop: the worker's
#    except block must still send STOP so ``run_all`` raises promptly instead of
#    freezing.
# --------------------------------------------------------------------------- #


class UnpicklableError(RuntimeError):
    """A RuntimeError whose instance carries a non-picklable payload.

    ``threading.Lock()`` cannot be pickled, so ``set_error(exception=self)`` on
    the multiprocessing manager proxy fails to serialize this instance.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.lock = threading.Lock()  # not picklable


class UnpicklableErrorFeatureGroup(DataCreatorFeatureGroup):
    """Root FG whose ``calculate_feature`` raises a non-picklable exception."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise UnpicklableError("boom")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_unpicklable_col"}


_ENABLED_UNPICKLABLE = PluginCollector.enabled_feature_groups({UnpicklableErrorFeatureGroup})


def test_multiprocessing_preserves_picklable_exception_type(flight_server: Any) -> None:
    """MULTIPROCESSING ``run_all`` must surface the original ``ImportError`` type.

    Regression guard: a PICKLABLE worker exception round-trips through the
    ``multiprocessing.managers`` proxy in ``set_error(exception=e)`` and is
    re-raised with its original type. This already passes today; it pins that
    the forthcoming pickle-failure guard does not degrade the happy path.
    """
    with pytest.raises(ImportError, match="bm25s"):
        mloda.run_all(
            [Feature(name="exc_import_error_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_IMPORT_ERROR,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )


def test_multiprocessing_preserves_missing_input_feature_error(flight_server: Any) -> None:
    """A declared input lost at runtime surfaces as MissingInputFeatureError with its fields intact."""
    with pytest.raises(MissingInputFeatureError) as excinfo:
        mloda.run_all(
            [Feature(name="SiblingDictLostInputConsumer")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {SiblingDictRoot, SiblingDictAggregate, SiblingDictLostInputConsumer}
            ),
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )
    assert excinfo.value.feature == "SiblingDictLostInputConsumer"
    assert excinfo.value.missing_inputs == ("sd_a",)
    assert excinfo.value.available_columns == ("SiblingDictAggregate",)
    assert "SiblingDictLostInputConsumer" in excinfo.value.feature_group


@pytest.mark.timeout(15)
def test_multiprocessing_unpicklable_exception_does_not_hang(flight_server: Any) -> None:
    """A NON-picklable worker exception must not hang the orchestrator loop.

    FAILS today: in MULTIPROCESSING mode ``cfw_register`` is a manager proxy, so
    ``set_error(exception=e)`` tries to PICKLE the ``UnpicklableError`` instance
    (it holds a ``threading.Lock``). Pickling raises inside the worker's except
    block BEFORE STOP is sent, so the orchestrator loop never terminates and the
    run HANGS. The ``@pytest.mark.timeout(15)`` turns that hang into a red
    ``Failed`` state instead of freezing the suite.

    The requirement: ``run_all`` must RAISE (any Exception) PROMPTLY. Once the
    production guard degrades ``set_error(exception=e)`` to ``set_error(msg,
    exc_info)`` on pickle failure, this surfaces as ``MlodaRunError`` and the
    test passes without timing out.
    """
    with pytest.raises(Exception):
        mloda.run_all(
            [Feature(name="exc_unpicklable_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_UNPICKLABLE,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )


def test_sync_row_leak_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Row values remain in the original exception but are absent from the DEBUG traceback."""
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(RowLeakError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_row_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ROW_LEAK,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert _ROW_LEAK_MARKER in str(excinfo.value)
    _assert_scrubbed_debug_traceback(caplog, _ROW_LEAK_MARKER)


def test_multiprocessing_row_leak_scrubbed_in_logs(flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Multiprocessing preserves raw row errors while scrubbing the DEBUG traceback."""
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(RowLeakError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_row_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ROW_LEAK,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

    assert _ROW_LEAK_MARKER in str(excinfo.value)
    _assert_scrubbed_debug_traceback(caplog, _ROW_LEAK_MARKER)


def test_sync_row_leak_chained_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Chained row errors remain available to the caller and are scrubbed from the DEBUG traceback."""
    caplog.set_level(logging.DEBUG, logger=_RUNNER_LOGGER)
    with pytest.raises(RuntimeError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_row_leak_chained_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ROW_LEAK_CHAINED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert isinstance(excinfo.value.__cause__, RowLeakError)
    assert _ROW_LEAK_MARKER in str(excinfo.value.__cause__)
    _assert_scrubbed_debug_traceback(caplog, _ROW_LEAK_MARKER)


# --------------------------------------------------------------------------- #
# Registered collection credentials: a bare token under a credential-shaped key
# is masked in raised error text and logs, while type and chain survive.
# --------------------------------------------------------------------------- #

_TOKEN_COLUMN = "toyfmt_tok_col"  # nosec B105


@dataclass(frozen=True)
class FrozenTokenError(Exception):
    """Frozen-dataclass exception whose only field carries the token."""

    detail: str


class TokenValueErrorFG(ToySourceFG):
    """Load raises a ValueError from a KeyError, token in the message."""

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        raise ValueError(f"username={TOY_TOKEN}") from KeyError("inner")


class TokenOSErrorFG(ToySourceFG):
    """Load raises an OSError whose strerror and filename carry the token."""

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        raise OSError(errno.ENOENT, f"cannot open {TOY_TOKEN}", f"/x/{TOY_TOKEN}")


class TokenFrozenFG(ToySourceFG):
    """Load raises a frozen-dataclass exception carrying the token."""

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        raise FrozenTokenError(f"seen {TOY_TOKEN}")


_TOKEN_GROUPS = [TokenValueErrorFG, TokenOSErrorFG, TokenFrozenFG]


def _assert_token_gone(
    exc: BaseException,
    group: type[Any],
    caplog: pytest.LogCaptureFixture,
    secrets: tuple[str, ...] = (TOY_TOKEN,),
) -> None:
    for secret in secrets:
        assert secret not in str(exc)
        assert secret not in repr(exc.args)
        assert secret not in caplog.text
    if group is TokenValueErrorFG:
        assert type(exc) is ValueError
        assert type(exc.__cause__) is KeyError
    if group is TokenOSErrorFG:
        assert type(exc) is FileNotFoundError
        assert TOY_TOKEN not in str(exc.filename)
        assert TOY_TOKEN not in str(exc.strerror)
    if group is TokenFrozenFG:
        assert type(exc) is FrozenTokenError


@pytest.mark.parametrize("group", _TOKEN_GROUPS)
def test_sync_load_error_masks_registered_token(group: type[ToySourceFG], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as excinfo:
            mloda.run_all([Feature(name=_TOKEN_COLUMN)], **toy_token_request(group, _TOKEN_COLUMN))

    _assert_token_gone(excinfo.value, group, caplog)


@pytest.mark.parametrize("group", _TOKEN_GROUPS)
def test_threading_load_error_masks_registered_token(
    group: type[ToySourceFG], flight_server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as excinfo:
            mloda.run_all(
                [Feature(name=_TOKEN_COLUMN)],
                parallelization_modes={ParallelizationMode.THREADING},
                flight_server=flight_server,
                **toy_token_request(group, _TOKEN_COLUMN),
            )

    _assert_token_gone(excinfo.value, group, caplog)


_TOKEN_KEYS = ["key", "hmac", "toyfmt_custom_field"]


@pytest.mark.parametrize("token_key", _TOKEN_KEYS)
def test_sync_load_error_masks_registered_token_under_any_key(token_key: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as excinfo:
            mloda.run_all(
                [Feature(name=_TOKEN_COLUMN)],
                **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN, token_key=token_key),
            )

    _assert_token_gone(excinfo.value, TokenValueErrorFG, caplog)


@pytest.mark.parametrize("token_key", _TOKEN_KEYS)
def test_threading_load_error_masks_registered_token_under_any_key(
    token_key: str, flight_server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as excinfo:
            mloda.run_all(
                [Feature(name=_TOKEN_COLUMN)],
                parallelization_modes={ParallelizationMode.THREADING},
                flight_server=flight_server,
                **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN, token_key=token_key),
            )

    _assert_token_gone(excinfo.value, TokenValueErrorFG, caplog)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("token_key", _TOKEN_KEYS)
def test_multiprocessing_load_error_masks_registered_token_under_any_key(
    token_key: str, flight_server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as excinfo:
            mloda.run_all(
                [Feature(name=_TOKEN_COLUMN)],
                parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                flight_server=flight_server,
                **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN, token_key=token_key),
            )

    exc = excinfo.value
    assert type(exc) is ValueError
    assert TOY_TOKEN not in str(exc)
    assert TOY_TOKEN not in repr(exc.args)
    assert TOY_TOKEN not in caplog.text


def test_stream_all_load_error_masks_registered_token(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            list(mloda.stream_all([Feature(name=_TOKEN_COLUMN)], **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN)))

    _assert_token_gone(excinfo.value, TokenValueErrorFG, caplog)


def test_stream_run_load_error_masks_registered_token(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            session = mloda.prepare(
                [Feature(name=_TOKEN_COLUMN)], **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN)
            )
            list(session.stream_run())

    _assert_token_gone(excinfo.value, TokenValueErrorFG, caplog)


def test_registered_token_scope_does_not_leak_after_a_failed_run() -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets

    with pytest.raises(ValueError):
        mloda.run_all([Feature(name=_TOKEN_COLUMN)], **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN))

    assert current_registered_secrets() == frozenset()


class TokenIdentityFG(ToySourceFG):
    """data_access_identity raises with the registered token in its message."""

    @classmethod
    def data_access_identity(cls, match: SourceMatch) -> str:
        raise ValueError(f"username={TOY_TOKEN}")


@pytest.mark.parametrize("entry", ["run_all", "explain", "resolved_plan"])
def test_plan_building_error_masks_registered_token(entry: str, caplog: pytest.LogCaptureFixture) -> None:
    request = toy_token_request(TokenIdentityFG, _TOKEN_COLUMN)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            if entry == "run_all":
                mloda.run_all([Feature(name=_TOKEN_COLUMN)], **request)
            elif entry == "explain":
                mloda.explain([Feature(name=_TOKEN_COLUMN)], **request)
            else:
                mloda.prepare([Feature(name=_TOKEN_COLUMN)], **request).resolved_plan()

    assert TOY_TOKEN not in str(excinfo.value)
    assert TOY_TOKEN not in repr(excinfo.value.args)
    assert TOY_TOKEN not in caplog.text


def test_early_stream_close_finalizes_inside_the_redaction_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets
    from mloda.core.runtime.run import ExecutionOrchestrator

    seen: list[frozenset[str]] = []
    original = ExecutionOrchestrator._finalize

    def spy(self: Any, cancelled: bool = False) -> None:
        seen.append(current_registered_secrets())
        original(self, cancelled)

    monkeypatch.setattr(ExecutionOrchestrator, "_finalize", spy)
    session = mloda.prepare([Feature(name=_TOKEN_COLUMN)], **toy_token_request(ToySourceFG, _TOKEN_COLUMN))
    stream = session.stream_run()
    next(stream)
    stream.close()

    assert seen
    assert all(TOY_TOKEN in literals for literals in seen)


def test_stream_start_enters_the_redaction_scope_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from mloda.core.api import request as request_module

    depth = 0
    deepest = 0
    original_enter = request_module._CredentialRedaction.__enter__
    original_exit = request_module._CredentialRedaction.__exit__

    def counting_enter(self: Any) -> None:
        nonlocal depth, deepest
        depth += 1
        deepest = max(deepest, depth)
        original_enter(self)

    def counting_exit(self: Any, *exc_info: Any) -> None:
        nonlocal depth
        depth -= 1
        original_exit(self, *exc_info)

    monkeypatch.setattr(request_module._CredentialRedaction, "__enter__", counting_enter)
    monkeypatch.setattr(request_module._CredentialRedaction, "__exit__", counting_exit)
    session = mloda.prepare([Feature(name=_TOKEN_COLUMN)], **toy_token_request(ToySourceFG, _TOKEN_COLUMN))

    stream = session.stream_run()
    stream.close()

    assert deepest == 1


class _RecordingExtender(Extender):
    """Records the text of every exception the wrapped call raises, as an OTel span would."""

    def __init__(self, hook: ExtenderHook) -> None:
        self.hook = hook
        self.seen: list[str] = []

    def wraps(self) -> set[ExtenderHook]:
        return {self.hook}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self.seen.append(f"{exc} {exc.args!r} {exc.__cause__!r}")
            raise


@pytest.mark.parametrize("hook", [ExtenderHook.INPUT_DATA_LOAD, ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE])
def test_extender_never_sees_the_registered_token(hook: ExtenderHook) -> None:
    recorder = _RecordingExtender(hook)
    with pytest.raises(ValueError):
        mloda.run_all(
            [Feature(name=_TOKEN_COLUMN)],
            function_extender={recorder},
            **toy_token_request(TokenValueErrorFG, _TOKEN_COLUMN),
        )

    assert recorder.seen
    assert all(TOY_TOKEN not in text for text in recorder.seen)


# --------------------------------------------------------------------------- #
# Declared credential grants reach every worker mode, are masked and are dropped.
# --------------------------------------------------------------------------- #

_E2E_HANDLE = "vault-main"
_E2E_HOST = "e2e-vault.internal.example"
_E2E_HMAC = "e2e-hmac-Gp8Tn5Rz4"  # nosec B105
_E2E_KEY = "e2e-key-Mq3Wc7Vh2"  # nosec B105
_GRANT_MODES = [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING]


def _vault_values() -> dict[str, Any]:
    return {"host": _E2E_HOST, "hmac": _E2E_HMAC, "key": _E2E_KEY}


def _vault_dac(credentials: dict[str, dict[str, Any]] | None = None) -> Any:
    from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection

    if credentials is None:
        credentials = {_E2E_HANDLE: _vault_values()}
    return DataAccessCollection(credentials=credentials)


def _grant_run(
    group: Any,
    name: str,
    mode: ParallelizationMode,
    flight_server: Any,
    function_extender: set[Extender] | None = None,
) -> Any:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
    from tests.helpers.credential_handle_stubs import HANDLE_KEY

    groups = set(group) if isinstance(group, tuple) else {group}
    return mloda.run_all(
        [Feature(name, options={HANDLE_KEY: _E2E_HANDLE})],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={mode},
        flight_server=flight_server,
        data_access_collection=_vault_dac(),
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
        function_extender=function_extender,
    )


class _FileRecordingExtender(Extender):
    """Appends the text of every exception the wrapped call raises to a file, so a spawned worker's view is readable."""

    def __init__(self, path: str, record_success: bool = False) -> None:
        self.path = path
        self.record_success = record_success

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = func(*args, **kwargs)
            if self.record_success:
                context = HookContext.current()
                assert context is not None
                with open(self.path, "a") as handle:
                    handle.write(json.dumps([list(context.credential_handles), list(context.credentials_read)]) + "\n")
            return result
        except Exception as exc:
            with open(self.path, "a") as handle:
                handle.write(f"{exc} {exc.args!r}\n")
            raise


def _granted(handle: str) -> Any:
    from mloda.core.abstract_plugins.components.credential_grant import _read_granted

    return _read_granted(handle)


def _secret_field(value: Mapping[str, Any]) -> str:
    return str(value["hmac"] if "hmac" in value else value["token"])


def _secret_length_via_helper(handle: str) -> int:
    return len(_secret_field(_granted(handle)))


class ChsReadingFG(ChsSource):
    """Reads its declared handle through the private reader, in calculate_feature and in a helper."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_read_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        handle = features.get_options_key(HANDLE_KEY)
        host = _granted(handle)["host"]
        row = f"{host}|{_secret_length_via_helper(handle)}"
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


class ChsReadSourceFG(ChsSource):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_read_src_a"}


class ChsReadingConsumerFG(ChsReadingFG):
    """Reads its declared handle over a plain input feature that never receives the handle."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_read_consumer_a"}

    def input_features(self, options: Any, feature_name: Any) -> set[Feature] | None:
        return {Feature("chs_read_src_a")}


class ChsFailingReadFG(ChsSource):
    """Reads its declared handle, then fails with an error that echoes the host and the secret."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_failing_read_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        value = _granted(features.get_options_key(HANDLE_KEY))
        raise ValueError(f"read failed host={value['host']} value={dict(value)}")


class ChsReadingArtifact(BaseArtifact):
    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        return {"saved_host": _granted(features.get_options_key(HANDLE_KEY))["host"]}


class ChsArtifactFG(ChsSource):
    """Its artifact saver reads the declared handle, so the grant must span save_artifact."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_artifact_a"}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return ChsReadingArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = "chs-artifact"
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


GRANT_ONLY_SECRET = "grant-only-Zk3Fw8Qd2"  # nosec B105


class GrantOnlyKind:
    """Test grant kind: every declared handle maps to a value only this class knows."""

    def __init__(self, values: Mapping[str, Any], policy: Any = None) -> None:
        self.handles = tuple(values)

    def open(self) -> "CredentialLookup":
        from mloda.core.abstract_plugins.components.credential_grant import CredentialLookup

        return CredentialLookup({h: {"host": "grant-only.example", "token": GRANT_ONLY_SECRET} for h in self.handles})


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
@pytest.mark.parametrize(
    "grant_only, expected",
    [
        pytest.param(False, f"{_E2E_HOST}|{len(_E2E_HMAC)}", id="declared_value"),
        pytest.param(True, f"grant-only.example|{len(GRANT_ONLY_SECRET)}", id="grant_only_value"),
    ],
)
@pytest.mark.parametrize(
    "groups, name",
    [
        pytest.param(ChsReadingFG, "chs_read_a", id="direct"),
        pytest.param((ChsReadingConsumerFG, ChsReadSourceFG), "chs_read_consumer_a", id="over_input"),
    ],
)
def test_declared_credential_is_readable_in_calculate_feature_and_a_helper_in_every_mode(
    mode: ParallelizationMode,
    grant_only: bool,
    expected: str,
    groups: Any,
    name: str,
    flight_server: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor

    if grant_only:
        monkeypatch.setattr(ComputeFrameworkExecutor, "_grant_kind", GrantOnlyKind)

    result = _grant_run(groups, name, mode, flight_server)

    assert list(result[0][name]) == [expected]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
@pytest.mark.parametrize(
    "grant_only, host, secrets, with_extender",
    [
        pytest.param(False, _E2E_HOST, (_E2E_HMAC, _E2E_KEY), True, id="declared_values"),
        pytest.param(True, "grant-only.example", (GRANT_ONLY_SECRET,), True, id="grant_only_value"),
    ],
)
def test_error_echoing_credential_secrets_is_masked_in_every_mode_but_host_stays(
    mode: ParallelizationMode,
    grant_only: bool,
    host: str,
    secrets: tuple[str, ...],
    with_extender: bool,
    flight_server: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Any,
) -> None:
    from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor

    if grant_only:
        monkeypatch.setattr(ComputeFrameworkExecutor, "_grant_kind", GrantOnlyKind)
    seen = tmp_path / "seen.txt"
    extenders: set[Extender] | None = {_FileRecordingExtender(str(seen))} if with_extender else None
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            _grant_run(ChsFailingReadFG, "chs_failing_read_a", mode, flight_server, extenders)

    exc = excinfo.value
    assert host in str(exc)
    _assert_token_gone(exc, ChsFailingReadFG, caplog, secrets)
    if with_extender:
        for secret in secrets:
            assert secret not in seen.read_text()


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_the_grant_spans_save_artifact_in_every_mode(mode: ParallelizationMode, flight_server: Any) -> None:
    from mloda.user import mloda as mloda_api
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
    from tests.helpers.credential_handle_stubs import HANDLE_KEY

    api = mloda_api(
        [Feature("chs_artifact_a", options={HANDLE_KEY: _E2E_HANDLE})],
        [PandasDataFrame],
        data_access_collection=_vault_dac(),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsArtifactFG}),
    )
    results = api.run(parallelization_modes={mode}, flight_server=flight_server)

    assert results.artifacts == {"chs_artifact_a": {"saved_host": _E2E_HOST}}


# --------------------------------------------------------------------------- #
# Runtime credential accessor: read_credential
# --------------------------------------------------------------------------- #

_VAULT_A = "vault-a"
_VAULT_B = "vault-b"
_HOST_A = "vault-a.internal.example"
_HOST_B = "vault-b.internal.example"
_WINDOW_EVENTS: list[tuple[str, str]] = []
_SAVER_CONTEXTS: list[contextvars.Context] = []


def _read_host(handle: str) -> str:
    return str(read_credential(handle)["host"])


def _read_secret_length_via_helper(handle: str) -> int:
    return len(str(read_credential(handle)["hmac"]))


class ChsRcReadingFG(ChsSource):
    """Reads its declared handle with read_credential in calculate_feature and in a helper."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_a", "chs_rc_b"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        handle = features.get_options_key(HANDLE_KEY)
        row = f"{_read_host(handle)}|{_read_secret_length_via_helper(handle)}"
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


class ChsRcFailingFG(ChsSource):
    """Reads via read_credential, then fails with an error echoing host, hmac and key."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_failing_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        value = read_credential(features.get_options_key(HANDLE_KEY))
        raise ValueError(f"read failed host={value['host']} hmac={value['hmac']} key={value['key']}")


class ChsRcMissingFieldFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_missing_field_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return read_credential(features.get_options_key(HANDLE_KEY))["no_such_field"]


class ChsRcUndeclaredFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_undeclared_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        read_credential("not-declared")
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


class ChsRcReadingArtifact(BaseArtifact):
    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        _SAVER_CONTEXTS.append(contextvars.copy_context())
        return {"saved_host": _read_host(features.get_options_key(HANDLE_KEY))}


class ChsRcArtifactFG(ChsSource):
    """Its artifact saver reads the declared handle with read_credential."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_artifact_a"}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return ChsRcReadingArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = "chs-rc-artifact"
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


def _window_load(data_access: Any, features: FeatureSet) -> str:
    host = _read_host(features.get_options_key(HANDLE_KEY))
    _WINDOW_EVENTS.append(("load", host))
    return host


async def _async_host(handle: str) -> str:
    return _read_host(handle)


async def _awaited_async_host(handle: str) -> str:
    return await asyncio.create_task(_async_host(handle))


class ChsWindowFG(ChsSource):
    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_win_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        from mloda.core.abstract_plugins.components.input_data.input_data_load import dispatch_input_data_load

        handle = features.get_options_key(HANDLE_KEY)
        _WINDOW_EVENTS.append(("calculate", _read_host(handle)))
        _WINDOW_EVENTS.append(("async", asyncio.run(_awaited_async_host(handle))))
        dispatch_input_data_load(
            cls,
            cls,
            _window_load,
            "access",
            features,
            identity=lambda: "chs-window-source",
            data_access_format=lambda: "chs",
        )
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


class ChsWinConsumerFG(ChsWindowFG):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_win_consumer_a"}

    def input_features(self, options: Any, feature_name: Any) -> set[Feature] | None:
        return {Feature("chs_read_src_a")}


def _try_read(key: str) -> None:
    """Record whether the outer handle is readable or refused under key."""
    try:
        read_credential(_E2E_HANDLE)
        _WINDOW_EVENTS.append((key, "readable"))
    except CredentialReadRefused as refused:
        assert _E2E_HANDLE in str(refused)
        _WINDOW_EVENTS.append((key, "refused"))


class _ReadProbeExtender(Extender):
    """Tries read_credential around each wrapped call and records the outcome and the HookContext."""

    def __init__(self, hooks: set[ExtenderHook]) -> None:
        self.hooks = hooks
        self.contexts: list[tuple[str, str | None, str]] = []
        self.kept: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return set(self.hooks)

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        self.kept.append(context)
        key = f"ext:{context.hook.value}"
        self.contexts.append((context.hook.value, context.feature_group_class, repr(context)))
        _try_read(key)
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            _WINDOW_EVENTS.append((key, f"raised:{type(exc).__name__}"))
            raise


def _events(key: str) -> list[str]:
    return [outcome for name, outcome in _WINDOW_EVENTS if name == key]


_TWO_VAULTS = {
    _VAULT_A: {"host": _HOST_A, "hmac": "a-hmac-Gp8Tn5Rz4", "key": "a-key-Mq3Wc7Vh2"},
    _VAULT_B: {"host": _HOST_B, "hmac": "b-hmac-Xs2Kd9Lf6x", "key": "b-key-Pt7Nb4Yc1"},
}


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_read_credential_works_in_calculate_feature_and_a_helper_in_every_mode(
    mode: ParallelizationMode, flight_server: Any
) -> None:
    result = _grant_run(ChsRcReadingFG, "chs_rc_a", mode, flight_server)

    assert list(result[0]["chs_rc_a"]) == [f"{_E2E_HOST}|{len(_E2E_HMAC)}"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_two_features_with_different_handles_each_read_their_own_credential(
    mode: ParallelizationMode, flight_server: Any
) -> None:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    result = mloda.run_all(
        [
            Feature("chs_rc_a", options={HANDLE_KEY: _VAULT_A}),
            Feature("chs_rc_b", options={HANDLE_KEY: _VAULT_B}),
        ],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={mode},
        flight_server=flight_server,
        data_access_collection=_vault_dac(_TWO_VAULTS),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcReadingFG}),
    )

    rows = {column: list(frame[column]) for frame in result for column in frame.columns}
    assert rows["chs_rc_a"] == [f"{_HOST_A}|{len('a-hmac-Gp8Tn5Rz4')}"]
    assert rows["chs_rc_b"] == [f"{_HOST_B}|{len('b-hmac-Xs2Kd9Lf6x')}"]


class ChsRcOneOfTwoFG(ChsTwoHandleFG):
    """Declares two handles and reads only the first."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_one_of_two"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        row = _read_host(features.get_options_key("chs_two_a_handle"))
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_a_step_declaring_two_handles_and_reading_one_reports_one_read_in_every_mode(
    mode: ParallelizationMode, flight_server: Any, tmp_path: Any
) -> None:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    seen = tmp_path / "seen.txt"
    mloda.run_all(
        [Feature("chs_rc_one_of_two", options={"chs_two_a_handle": _VAULT_A, "chs_two_z_handle": _VAULT_B})],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={mode},
        flight_server=flight_server,
        data_access_collection=_vault_dac(_TWO_VAULTS),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcOneOfTwoFG}),
        function_extender={_FileRecordingExtender(str(seen), record_success=True)},
    )

    text = seen.read_text()
    assert [json.loads(line) for line in text.splitlines()] == [[[_VAULT_A, _VAULT_B], [_VAULT_A]]]
    for vault in _TWO_VAULTS.values():
        for value in (vault["hmac"], vault["key"]):
            assert value not in text


class ChsRcIsolationFG(ChsSource):
    """Reports whether its own and the other credential's secret literals are registered in the worker."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_iso_a", "chs_rc_iso_b"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets

        handle = features.get_options_key(HANDLE_KEY)
        other = _VAULT_B if handle == _VAULT_A else _VAULT_A
        own_secret = str(read_credential(handle)["hmac"])
        registered = current_registered_secrets()
        row = f"own={own_secret in registered}|other={_TWO_VAULTS[other]['hmac'] in registered}"
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


@pytest.mark.timeout(15)
def test_a_multiprocessing_step_registers_only_its_own_credentials_secrets(flight_server: Any) -> None:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    result = mloda.run_all(
        [
            Feature("chs_rc_iso_a", options={HANDLE_KEY: _VAULT_A}),
            Feature("chs_rc_iso_b", options={HANDLE_KEY: _VAULT_B}),
        ],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
        data_access_collection=_vault_dac(_TWO_VAULTS),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcIsolationFG}),
    )

    rows = {column: list(frame[column]) for frame in result for column in frame.columns}
    assert rows["chs_rc_iso_a"] == ["own=True|other=False"]
    assert rows["chs_rc_iso_b"] == ["own=True|other=False"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_error_echoing_values_read_with_read_credential_is_masked_in_every_mode(
    mode: ParallelizationMode,
    flight_server: Any,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Any,
) -> None:
    seen = tmp_path / "seen.txt"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            _grant_run(ChsRcFailingFG, "chs_rc_failing_a", mode, flight_server, {_FileRecordingExtender(str(seen))})

    exc = excinfo.value
    assert _E2E_HOST in str(exc)
    _assert_token_gone(exc, ChsRcFailingFG, caplog, (_E2E_HMAC, _E2E_KEY))
    for secret in (_E2E_HMAC, _E2E_KEY):
        assert secret not in seen.read_text()


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_read_credential_works_in_custom_saver_in_every_mode(mode: ParallelizationMode, flight_server: Any) -> None:
    from mloda.user import mloda as mloda_api
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    api = mloda_api(
        [Feature("chs_rc_artifact_a", options={HANDLE_KEY: _E2E_HANDLE})],
        [PandasDataFrame],
        data_access_collection=_vault_dac(),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcArtifactFG}),
    )
    results = api.run(parallelization_modes={mode}, flight_server=flight_server)

    assert results.artifacts == {"chs_rc_artifact_a": {"saved_host": _E2E_HOST}}


class ChsRcSplitArtifact(BaseArtifact):
    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        return {"saved_host": _read_host(features.get_options_key("chs_two_z_handle"))}


class ChsRcCalcAndSaverFG(ChsTwoHandleFG):
    """Calculate reads the first handle, the artifact saver reads the second."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_split_a"}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return ChsRcSplitArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        _read_host(features.get_options_key("chs_two_a_handle"))
        if features.artifact_to_save:
            features.save_artifact = "chs-rc-split"
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
@pytest.mark.parametrize(
    ("group", "name", "options", "expected"),
    [
        pytest.param(ChsRcArtifactFG, "chs_rc_artifact_a", {HANDLE_KEY: _E2E_HANDLE}, (_E2E_HANDLE,), id="saver_only"),
        pytest.param(
            ChsRcCalcAndSaverFG,
            "chs_rc_split_a",
            {"chs_two_a_handle": _VAULT_A, "chs_two_z_handle": _VAULT_B},
            (_VAULT_A, _VAULT_B),
            id="calculate_and_saver",
        ),
    ],
)
def test_a_saver_read_shows_on_the_calculate_context_after_the_run(
    mode: ParallelizationMode,
    flight_server: Any,
    group: type[ChsSource],
    name: str,
    options: dict[str, Any],
    expected: tuple[str, ...],
) -> None:
    from mloda.user import mloda as mloda_api
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    probe = _ReadProbeExtender({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE})
    api = mloda_api(
        [Feature(name, options=options)],
        [PandasDataFrame],
        data_access_collection=_vault_dac({**_TWO_VAULTS, _E2E_HANDLE: _vault_values()}),
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
        function_extender={probe},
    )
    api.run(parallelization_modes={mode}, flight_server=flight_server)

    assert [context.credentials_read for context in probe.kept] == [expected]


class ChsRcRaisingArtifact(BaseArtifact):
    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        _read_host(features.get_options_key("chs_two_z_handle"))
        raise RuntimeError("saver failed")


class ChsRcRaisingSaverFG(ChsRcCalcAndSaverFG):
    """Calculate reads the first handle, the saver reads the second and then raises."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_rc_raising_saver_a"}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return ChsRcRaisingArtifact


@pytest.mark.timeout(15)
def test_a_saver_that_reads_then_raises_still_shows_the_read_on_the_calculate_context(flight_server: Any) -> None:
    from mloda.user import mloda as mloda_api
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    probe = _ReadProbeExtender({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE})
    api = mloda_api(
        [Feature("chs_rc_raising_saver_a", options={"chs_two_a_handle": _VAULT_A, "chs_two_z_handle": _VAULT_B})],
        [PandasDataFrame],
        data_access_collection=_vault_dac(_TWO_VAULTS),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcRaisingSaverFG}),
        function_extender={probe},
    )
    with pytest.raises(RuntimeError, match="saver failed"):
        api.run(parallelization_modes={ParallelizationMode.SYNC}, flight_server=flight_server)

    assert [context.credentials_read for context in probe.kept] == [(_VAULT_A, _VAULT_B)]


@pytest.mark.timeout(15)
def test_read_credential_raises_after_save_artifact_returns(flight_server: Any) -> None:
    from mloda.user import mloda as mloda_api
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    _SAVER_CONTEXTS.clear()
    api = mloda_api(
        [Feature("chs_rc_artifact_a", options={HANDLE_KEY: _E2E_HANDLE})],
        [PandasDataFrame],
        data_access_collection=_vault_dac(),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcArtifactFG}),
    )
    api.run(parallelization_modes={ParallelizationMode.SYNC}, flight_server=flight_server)

    assert len(_SAVER_CONTEXTS) == 1
    with pytest.raises(CredentialReadRefused, match=_E2E_HANDLE):
        read_credential(_E2E_HANDLE)
    with pytest.raises(CredentialReadRefused, match=_E2E_HANDLE):
        _SAVER_CONTEXTS[0].run(read_credential, _E2E_HANDLE)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_a_missing_field_raises_the_typed_error_not_the_missing_column_message(
    mode: ParallelizationMode, flight_server: Any
) -> None:
    with pytest.raises(CredentialFieldMissing) as excinfo:
        _grant_run(ChsRcMissingFieldFG, "chs_rc_missing_field_a", mode, flight_server)

    text = str(excinfo.value)
    assert _E2E_HANDLE in text
    assert "no_such_field" in text
    assert "input_features" not in text
    assert _E2E_HMAC not in text


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_an_undeclared_handle_is_refused_naming_the_declared_ones_in_every_mode(
    mode: ParallelizationMode, flight_server: Any
) -> None:
    with pytest.raises(CredentialReadRefused) as excinfo:
        _grant_run(ChsRcUndeclaredFG, "chs_rc_undeclared_a", mode, flight_server)

    assert "not-declared" in str(excinfo.value)
    assert _E2E_HANDLE in str(excinfo.value)
    assert _E2E_HMAC not in str(excinfo.value)


@pytest.mark.timeout(15)
def test_an_extender_wrapping_the_step_sees_the_refusal_as_the_error(flight_server: Any) -> None:
    _WINDOW_EVENTS.clear()
    probe = _ReadProbeExtender({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE})

    with pytest.raises(CredentialReadRefused):
        _grant_run(ChsRcUndeclaredFG, "chs_rc_undeclared_a", ParallelizationMode.SYNC, flight_server, {probe})

    assert _events("ext:feature_group_calculate_feature") == ["refused", "raised:CredentialReadRefused"]
    assert [context.credentials_read for context in probe.kept] == [()]


_WINDOW_SHAPES = [
    pytest.param(set(), id="no_extender"),
    pytest.param({ExtenderHook.INPUT_DATA_LOAD}, id="fetch_extender_only"),
    pytest.param({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}, id="calculate_extender"),
    pytest.param(
        {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}, id="calculate_and_fetch_extender"
    ),
]


def _closed_window_load(data_access: Any, features: FeatureSet) -> str:
    try:
        _read_host(features.get_options_key(HANDLE_KEY))
        _WINDOW_EVENTS.append(("closed_load", "readable"))
    except CredentialReadRefused:
        _WINDOW_EVENTS.append(("closed_load", "refused"))
    return "loaded"


class ChsClosedWindowFG(ChsSource):
    """Dispatches a load while the readable window is closed (grant held, no readable scope)."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_closed_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        from mloda.core.abstract_plugins.components.credential_access import hidden_credentials
        from mloda.core.abstract_plugins.components.input_data.input_data_load import dispatch_input_data_load

        with hidden_credentials():
            dispatch_input_data_load(
                cls,
                cls,
                _closed_window_load,
                "access",
                features,
                identity=lambda: "chs-closed-source",
                data_access_format=lambda: "chs",
            )
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


@pytest.mark.timeout(15)
def test_a_load_dispatched_while_the_window_is_closed_keeps_its_reads_refused(flight_server: Any) -> None:
    _WINDOW_EVENTS.clear()
    probe = _ReadProbeExtender({ExtenderHook.INPUT_DATA_LOAD})

    _grant_run(ChsClosedWindowFG, "chs_closed_a", ParallelizationMode.SYNC, flight_server, {probe})

    assert _events("closed_load") == ["refused"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("hooks", _WINDOW_SHAPES)
def test_read_credential_is_open_in_calculate_an_awaited_task_and_the_load_but_not_in_extenders(
    hooks: set[ExtenderHook], flight_server: Any
) -> None:
    _WINDOW_EVENTS.clear()
    probe = _ReadProbeExtender(hooks)

    _grant_run(ChsWindowFG, "chs_win_a", ParallelizationMode.SYNC, flight_server, {probe} if hooks else None)

    assert _events("calculate") == [_E2E_HOST]
    assert _events("async") == [_E2E_HOST]
    assert _events("load") == [_E2E_HOST]
    for hook in hooks:
        assert _events(f"ext:{hook.value}") == ["refused"]
    assert {name for name, _ in _WINDOW_EVENTS if name.startswith("ext:")} == {f"ext:{h.value}" for h in hooks}
    assert {context.hook for context in probe.kept} == hooks
    for context in probe.kept:
        assert context.credentials_read == (_E2E_HANDLE,)


@pytest.mark.timeout(15)
def test_validate_hooks_cannot_read_while_the_grant_is_held(flight_server: Any) -> None:
    _WINDOW_EVENTS.clear()
    probe = _ReadProbeExtender({ExtenderHook.VALIDATE_INPUT_FEATURE, ExtenderHook.VALIDATE_OUTPUT_FEATURE})

    _grant_run(
        (ChsWinConsumerFG, ChsReadSourceFG), "chs_win_consumer_a", ParallelizationMode.SYNC, flight_server, {probe}
    )

    assert "refused" in _events("ext:validate_input_feature")
    assert "refused" in _events("ext:validate_output_feature")
    assert "readable" not in _events("ext:validate_input_feature") + _events("ext:validate_output_feature")
    assert probe.kept
    assert all(context.credentials_read == () for context in probe.kept)


_NESTED_INNER: list[Any] = []
_NESTED_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.VALIDATE_OUTPUT_FEATURE}


class _NestedProbe(Extender):
    """Inner-run probe: tries the outer handle in wrapped calls and lifecycle hooks, recording under inner: keys."""

    def __init__(self) -> None:
        self.hooks = set(_NESTED_HOOKS)

    def wraps(self) -> set[ExtenderHook]:
        return set(self.hooks)

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        _try_read(f"inner:{context.hook.value}")
        return func(*args, **kwargs)

    def on_plan_start(self, plan: Any) -> None:
        _try_read("inner:on_plan_start")

    def on_plan_complete(self, plan: Any, outcome: Any) -> None:
        _try_read("inner:on_plan_complete")

    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        _try_read("inner:on_run_start")

    def on_run_complete(self, run: Any, outcome: Any) -> None:
        _try_read("inner:on_run_complete")


class ChsNestedOuterFG(ChsSource):
    """Reads its handle, runs a nested mloda run in the same thread, then reads its handle again."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_nested_outer_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        handle = features.get_options_key(HANDLE_KEY)
        _WINDOW_EVENTS.append(("outer_before", _read_host(handle)))
        _NESTED_INNER[0]()
        _WINDOW_EVENTS.append(("outer_after", _read_host(handle)))
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


class ChsNestedBareFG(ChsSource):
    """Declares no handle and tries to read the outer one."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_nested_bare_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        read_credential(_E2E_HANDLE)
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


def _inner_run(
    group: Any,
    name: str,
    entry: str,
    extenders: set[Extender],
    options: dict[str, Any],
    frameworks: list[Any] | None = None,
    dac: Any = None,
) -> list[str]:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    kwargs: dict[str, Any] = {
        "compute_frameworks": frameworks or [PandasDataFrame],
        "data_access_collection": dac or _vault_dac({_VAULT_B: _TWO_VAULTS[_VAULT_B]}),
        "plugin_collector": PluginCollector.enabled_feature_groups({group}),
        "function_extender": extenders,
    }
    features = [Feature(name, options=options)]
    result = (
        mloda.prepare(features, **kwargs).run() if entry == "prepare" else getattr(mloda, entry)(features, **kwargs)
    )
    return [str(row) for frame in result for row in frame[name]]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
@pytest.mark.parametrize("entry", ["run_all", "stream_all", "prepare"])
def test_a_nested_run_starts_with_the_credential_window_closed(
    entry: str, mode: ParallelizationMode, flight_server: Any
) -> None:
    _WINDOW_EVENTS.clear()
    _NESTED_INNER.clear()
    rows: list[str] = []
    _NESTED_INNER.append(
        lambda: rows.extend(_inner_run(ChsRcReadingFG, "chs_rc_a", entry, {_NestedProbe()}, {HANDLE_KEY: _VAULT_B}))
    )
    outer = _ReadProbeExtender({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE})

    _grant_run(ChsNestedOuterFG, "chs_nested_outer_a", mode, flight_server, {outer})

    assert rows == [f"{_HOST_B}|{len(_TWO_VAULTS[_VAULT_B]['hmac'])}"]
    inner_keys = [(name, outcome) for name, outcome in _WINDOW_EVENTS if name.startswith("inner:")]
    assert {name for name, _ in inner_keys} >= {
        "inner:feature_group_calculate_feature",
        "inner:validate_output_feature",
        "inner:on_plan_start",
        "inner:on_plan_complete",
        "inner:on_run_start",
        "inner:on_run_complete",
    }
    assert {outcome for _, outcome in inner_keys} == {"refused"}
    assert _events("outer_before") == [_E2E_HOST]
    assert _events("outer_after") == [_E2E_HOST]
    assert [context.credentials_read for context in outer.kept] == [(_E2E_HANDLE,)]


@pytest.mark.timeout(15)
def test_the_outer_window_reopens_after_a_nested_run_raises(flight_server: Any) -> None:
    """The outer step reads again after a nested run raises."""
    _WINDOW_EVENTS.clear()
    _NESTED_INNER.clear()
    failures: list[BaseException] = []

    def inner() -> None:
        try:
            _inner_run(ChsNestedBareFG, "chs_nested_bare_a", "run_all", set(), {})
        except Exception as exc:
            failures.append(exc)

    _NESTED_INNER.append(inner)

    _grant_run(ChsNestedOuterFG, "chs_nested_outer_a", ParallelizationMode.SYNC, flight_server)

    assert len(failures) == 1
    assert _E2E_HANDLE in str(failures[0])
    assert _events("outer_after") == [_E2E_HOST]


_NESTED_SEEN: dict[str, Any] = {}
_NESTED_GATE_LOADS: dict[str, list[str]] = {"outer": [], "inner": []}


class _NestedLoadGate(Extender):
    """INPUT_DATA_LOAD extender recording the loading group under a label; a gate when never_fall_back."""

    def __init__(self, label: str, gate: bool) -> None:
        self.label = label
        self.never_fall_back = gate

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        assert context is not None
        _NESTED_GATE_LOADS[self.label].append(str(context.feature_group_class))
        return func(*args, **kwargs)


class ChsNestedPublishOuterFG(ChsNestedOuterFG):
    """Publishes an attribute and reports failed rows, runs a nested run, then publishes again."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_nested_publish_outer_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        HookContext.publish_result_attributes({"outer_before": 1})
        report_failed_rows("chs_nested_publish_outer_a", [0], "outer_reason")
        before = HookContext.current()
        _NESTED_INNER[0]()
        _NESTED_SEEN["outer_same_context"] = HookContext.current() is before
        _NESTED_SEEN["outer_framework_after"] = ComputeFramework.current()
        HookContext.publish_result_attributes({"outer_after": 1})
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


class ChsNestedInnerPublishFG(ChsSource):
    """Inner group: publishes an attribute, reports failed rows and records the current contexts."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_nested_inner_publish_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        _NESTED_SEEN["inner_hook_context"] = HookContext.current()
        _NESTED_SEEN["inner_framework"] = ComputeFramework.current()
        HookContext.publish_result_attributes({"inner_attr": 1})
        report_failed_rows("chs_nested_inner_publish_a", [1], "inner_reason")
        return pd.DataFrame({name: [1] for name in features.get_all_names()})


def _run_publish_outer(mode: ParallelizationMode, flight_server: Any) -> _ReadProbeExtender:
    _WINDOW_EVENTS.clear()
    keep = _ReadProbeExtender({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE})
    _grant_run(ChsNestedPublishOuterFG, "chs_nested_publish_outer_a", mode, flight_server, {keep})
    return keep


def _outer_context(keep: _ReadProbeExtender) -> HookContext:
    outer = [c for c in keep.kept if c.feature_names == ("chs_nested_publish_outer_a",)]
    assert len(outer) == 1
    return outer[0]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
@pytest.mark.parametrize("entry", ["run_all", "stream_all", "prepare"])
def test_a_nested_run_does_not_see_or_write_the_outer_hook_context_or_framework(
    entry: str, mode: ParallelizationMode, flight_server: Any
) -> None:
    _NESTED_SEEN.clear()
    _NESTED_INNER.clear()
    _NESTED_INNER.append(lambda: _inner_run(ChsNestedInnerPublishFG, "chs_nested_inner_publish_a", entry, set(), {}))

    keep = _run_publish_outer(mode, flight_server)

    assert _NESTED_SEEN["inner_hook_context"] is None
    assert _NESTED_SEEN["inner_framework"] is None
    attributes = _outer_context(keep).result_attributes
    assert attributes is not None
    assert set(attributes) == {"outer_before", "outer_after"}


@pytest.mark.timeout(15)
def test_a_nested_run_does_not_add_failed_rows_to_the_outer_context(flight_server: Any) -> None:
    """Regression pin: every step opens its own collector, so this already holds."""
    _NESTED_SEEN.clear()
    _NESTED_INNER.clear()
    _NESTED_INNER.append(
        lambda: _inner_run(ChsNestedInnerPublishFG, "chs_nested_inner_publish_a", "run_all", set(), {})
    )

    keep = _run_publish_outer(ParallelizationMode.SYNC, flight_server)

    assert {rows.reason for rows in _outer_context(keep).failed_rows} == {"outer_reason"}


class ChsNestedLoadOuterFG(ChsNestedOuterFG):
    """Outer group of the gate cases: only runs the nested call."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_nested_load_outer_a"}


def _nested_toy_load(extenders: set[Extender]) -> None:
    _inner_run(
        ToySourceFG,
        "toyfmt_nested_col",
        "run_all",
        extenders,
        {},
        frameworks=[PythonDictFramework],
        dac=toy_dac(h1={"toyfmt_nested_col": [1]}),
    )


def _run_gated_outer() -> None:
    _grant_run(
        ChsNestedLoadOuterFG,
        "chs_nested_load_outer_a",
        ParallelizationMode.SYNC,
        None,
        {_NestedLoadGate("outer", gate=True)},
    )


@pytest.mark.timeout(15)
@pytest.mark.parametrize("inner_extenders", ["none", "non_gating"])
def test_a_nested_load_under_an_outer_gate_without_its_own_gate_fails_closed(inner_extenders: str) -> None:
    _WINDOW_EVENTS.clear()
    _NESTED_GATE_LOADS.update(outer=[], inner=[])
    failures: list[BaseException] = []
    extenders: set[Extender] = set() if inner_extenders == "none" else {_NestedLoadGate("inner", gate=False)}

    def inner() -> None:
        with pytest.raises(GateBypassError) as caught:
            _nested_toy_load(extenders)
        failures.append(caught.value)

    _NESTED_INNER.clear()
    _NESTED_INNER.append(inner)

    _run_gated_outer()

    assert len(failures) == 1
    assert _NESTED_GATE_LOADS["outer"] == []
    assert _NESTED_GATE_LOADS["inner"] == []


@pytest.mark.timeout(15)
def test_a_nested_load_under_an_outer_gate_goes_through_the_inner_runs_own_gate() -> None:
    _WINDOW_EVENTS.clear()
    _NESTED_GATE_LOADS.update(outer=[], inner=[])
    _NESTED_INNER.clear()
    _NESTED_INNER.append(lambda: _nested_toy_load({_NestedLoadGate("inner", gate=True)}))

    _run_gated_outer()

    assert _NESTED_GATE_LOADS["outer"] == []
    assert [name.rsplit(".", 1)[-1] for name in _NESTED_GATE_LOADS["inner"]] == ["ToySourceFG"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("inner_raises", [False, True])
def test_the_outer_hook_context_and_framework_are_back_after_a_nested_run(
    inner_raises: bool, flight_server: Any
) -> None:
    _NESTED_SEEN.clear()
    _NESTED_INNER.clear()
    group, name = (
        (ChsNestedBareFG, "chs_nested_bare_a")
        if inner_raises
        else (
            ChsNestedInnerPublishFG,
            "chs_nested_inner_publish_a",
        )
    )

    def inner() -> None:
        with pytest.raises(Exception) if inner_raises else contextlib.nullcontext():
            _inner_run(group, name, "run_all", set(), {})

    _NESTED_INNER.append(inner)

    keep = _run_publish_outer(ParallelizationMode.SYNC, flight_server)

    assert _NESTED_SEEN["outer_same_context"] is True
    assert _NESTED_SEEN["outer_framework_after"] is not None
    attributes = _outer_context(keep).result_attributes
    assert attributes is not None
    assert "outer_after" in attributes


@pytest.mark.timeout(15)
def test_no_hook_context_carries_a_credential_value(flight_server: Any) -> None:
    _WINDOW_EVENTS.clear()
    probe = _ReadProbeExtender(
        {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }
    )

    _grant_run(
        (ChsWinConsumerFG, ChsReadSourceFG), "chs_win_consumer_a", ParallelizationMode.SYNC, flight_server, {probe}
    )

    assert {hook for hook, _, _ in probe.contexts} >= {"feature_group_calculate_feature", "input_data_load"}
    for _, _, rendered in probe.contexts:
        assert _E2E_HMAC not in rendered
        assert _E2E_KEY not in rendered
    for context in probe.kept:
        assert _E2E_HMAC not in repr(context)
        assert _E2E_KEY not in repr(context)
        if (
            context.hook is ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
            and context.feature_group_class == f"{ChsWinConsumerFG.__module__}.{ChsWinConsumerFG.__qualname__}"
        ):
            assert context.credentials_read == (_E2E_HANDLE,)
        if context.hook in {ExtenderHook.VALIDATE_INPUT_FEATURE, ExtenderHook.VALIDATE_OUTPUT_FEATURE}:
            assert context.credentials_read == ()
    load_reads = [c.credentials_read for c in probe.kept if c.hook is ExtenderHook.INPUT_DATA_LOAD]
    assert load_reads
    assert all(reads in ((), (_E2E_HANDLE,)) for reads in load_reads)
    assert (_E2E_HANDLE,) in load_reads


@pytest.mark.timeout(15)
def test_the_hook_context_feature_group_is_the_same_with_and_without_a_calculate_extender(flight_server: Any) -> None:
    def calculate_groups(hooks: set[ExtenderHook]) -> set[str | None]:
        _WINDOW_EVENTS.clear()
        probe = _ReadProbeExtender(hooks)
        _grant_run(ChsWindowFG, "chs_win_a", ParallelizationMode.SYNC, flight_server, {probe})
        return {group for hook, group, _ in probe.contexts if hook in {"feature_group_calculate_feature"}}

    with_calculate = calculate_groups({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE})
    with_both = calculate_groups({ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD})

    assert with_calculate == with_both
    assert len(with_calculate) == 1
    assert "ChsWindowFG" in str(next(iter(with_calculate)))


# --------------------------------------------------------------------------- #
# Run-scoped log masking: a plugin's own log lines never carry a registered literal.
# --------------------------------------------------------------------------- #


class _LineCapture(logging.Handler):
    """Collects the formatted message of every record, so a spawned worker's view is returnable as a value."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _capture_plugin_lines(emit: Any) -> str:
    plugin_logger = logging.getLogger(f"mloda_test.plugin_log.{threading.get_ident()}")
    plugin_logger.setLevel(logging.DEBUG)
    plugin_logger.propagate = False
    handler = _LineCapture()
    plugin_logger.addHandler(handler)
    try:
        emit(plugin_logger)
    finally:
        plugin_logger.removeHandler(handler)
    return "|".join(handler.lines)


class ChsLoggingFG(ChsSource):
    """Logs the granted secrets through %s args and a pre-formatted message."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_log_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        value = read_credential(features.get_options_key(HANDLE_KEY))

        def emit(plugin_logger: logging.Logger) -> None:
            plugin_logger.info("connect %s", value["hmac"])
            plugin_logger.info(f"key is {value['key']}")
            plugin_logger.info("host %s", value["host"])

        row = _capture_plugin_lines(emit)
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


class ChsThreadLoggingFG(ChsSource):
    """Logs the granted secret from a plain threading.Thread, which runs outside the run scope."""

    PROPERTY_MAPPING = {HANDLE_KEY: credential_handle_spec("Vault credential handle.")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chs_log_thread_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        secret = str(read_credential(features.get_options_key(HANDLE_KEY))["hmac"])

        def emit(plugin_logger: logging.Logger) -> None:
            worker = threading.Thread(target=lambda: plugin_logger.info("from thread %s", secret))
            worker.start()
            worker.join(5)

        row = _capture_plugin_lines(emit)
        return pd.DataFrame({name: [row] for name in features.get_all_names()})


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_plugin_log_lines_are_masked_in_every_mode_but_the_host_stays(
    mode: ParallelizationMode, flight_server: Any
) -> None:
    result = _grant_run(ChsLoggingFG, "chs_log_a", mode, flight_server)

    assert list(result[0]["chs_log_a"]) == [f"connect ***|key is ***|host {_E2E_HOST}"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
@pytest.mark.parametrize("enabled", [False, True], ids=["not_enabled", "enabled"])
def test_a_plugin_thread_outside_the_run_scope_logs_unmasked_unless_process_wide_masking_is_on(
    mode: ParallelizationMode, enabled: bool, flight_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thread the plugin starts has no scope: raw by default, masked against the union when enabled."""
    from mloda.core.abstract_plugins.components import log_masking

    monkeypatch.setattr(log_masking, "_process_wide", enabled)
    result = _grant_run(ChsThreadLoggingFG, "chs_log_thread_a", mode, flight_server)

    expected = "from thread ***" if enabled else f"from thread {_E2E_HMAC}"
    assert list(result[0]["chs_log_thread_a"]) == [expected]


# --------------------------------------------------------------------------- #
# Fresh credential values per run() on a prepared session
# --------------------------------------------------------------------------- #

_FRESH_1 = {"host": "fresh-one.internal.example", "hmac": "fresh1-hmac-Zq8Lw3Nd", "key": "fresh1-key-Vb6Hc2Rt"}  # nosec B105
_FRESH_2 = {"host": "fresh-two.internal.example", "hmac": "fresh2-hmac-Yp7Km4Se9x", "key": "fresh2-key-Uc5Jd8Wq"}  # nosec B105


def _row(value: Mapping[str, Any]) -> str:
    return f"{value['host']}|{len(str(value['hmac']))}"


def _prepare_vault(group: Any, name: str, mode: ParallelizationMode, extenders: set[Extender] | None = None) -> Any:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    return mloda.prepare(
        [Feature(name, options={HANDLE_KEY: _E2E_HANDLE})],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={mode},
        data_access_collection=_vault_dac(),
        plugin_collector=PluginCollector.enabled_feature_groups({group}),
        function_extender=extenders,
    )


def _stream_rows(stream: Any, column: str) -> list[str]:
    return [str(row) for frame in stream for row in frame[column]]


def _run_rows(session: Any, column: str, **kwargs: Any) -> list[str]:
    return [str(row) for frame in session.run(**kwargs) for row in frame[column]]


def _stream_run_rows(session: Any, column: str, **kwargs: Any) -> list[str]:
    return _stream_rows(session.stream_run(**kwargs), column)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
@pytest.mark.parametrize("rows_of", [_run_rows, _stream_run_rows], ids=["run", "stream_run"])
def test_each_run_on_a_prepared_session_reads_its_own_fresh_credential(
    rows_of: Any, mode: ParallelizationMode, flight_server: Any
) -> None:
    session = _prepare_vault(ChsRcReadingFG, "chs_rc_a", mode)
    kwargs: dict[str, Any] = {"parallelization_modes": {mode}, "flight_server": flight_server}

    first = rows_of(session, "chs_rc_a", credentials={_E2E_HANDLE: _FRESH_1}, **kwargs)
    second = rows_of(session, "chs_rc_a", credentials={_E2E_HANDLE: _FRESH_2}, **kwargs)
    third = rows_of(session, "chs_rc_a", **kwargs)

    assert first == [_row(_FRESH_1)]
    assert second == [_row(_FRESH_2)]
    assert third == [f"{_E2E_HOST}|{len(_E2E_HMAC)}"]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_a_handle_not_passed_keeps_its_prepared_value_while_the_passed_one_is_fresh(
    mode: ParallelizationMode, flight_server: Any
) -> None:
    from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

    session = mloda.prepare(
        [
            Feature("chs_rc_a", options={HANDLE_KEY: _VAULT_A}),
            Feature("chs_rc_b", options={HANDLE_KEY: _VAULT_B}),
        ],
        compute_frameworks=[PandasDataFrame],
        parallelization_modes={mode},
        data_access_collection=_vault_dac(_TWO_VAULTS),
        plugin_collector=PluginCollector.enabled_feature_groups({ChsRcReadingFG}),
    )

    result = session.run(credentials={_VAULT_A: _FRESH_1}, parallelization_modes={mode}, flight_server=flight_server)

    columns = {column: list(frame[column]) for frame in result for column in frame.columns}
    assert columns["chs_rc_a"] == [_row(_FRESH_1)]
    assert columns["chs_rc_b"] == [_row(_TWO_VAULTS[_VAULT_B])]


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mode", _GRANT_MODES)
def test_error_echoing_a_fresh_credential_is_masked_in_every_mode(
    mode: ParallelizationMode,
    flight_server: Any,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Any,
) -> None:
    seen = tmp_path / "seen.txt"
    session = _prepare_vault(ChsRcFailingFG, "chs_rc_failing_a", mode, {_FileRecordingExtender(str(seen))})

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            session.run(credentials={_E2E_HANDLE: _FRESH_1}, parallelization_modes={mode}, flight_server=flight_server)

    exc = excinfo.value
    secrets = (str(_FRESH_1["hmac"]), str(_FRESH_1["key"]))
    assert str(_FRESH_1["host"]) in str(exc)
    _assert_token_gone(exc, ChsRcFailingFG, caplog, secrets)
    for secret in secrets:
        assert secret not in seen.read_text()


@pytest.mark.timeout(15)
def test_two_interleaved_stream_runs_on_one_session_each_read_their_own_value() -> None:
    session = _prepare_vault(ChsRcReadingFG, "chs_rc_a", ParallelizationMode.SYNC)

    stream_1 = session.stream_run(credentials={_E2E_HANDLE: Credential(**_FRESH_1)})
    stream_2 = session.stream_run(credentials={_E2E_HANDLE: _FRESH_2})
    rows_2 = _stream_rows(stream_2, "chs_rc_a")
    rows_1 = _stream_rows(stream_1, "chs_rc_a")

    assert rows_1 == [_row(_FRESH_1)]
    assert rows_2 == [_row(_FRESH_2)]
