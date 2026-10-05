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

import logging
import threading
from typing import Any

import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda

# Importing the framework registers it as a ComputeFramework subclass so
# ``compute_frameworks=["PythonDictFramework"]`` resolves during ``run_all``.
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (  # noqa: F401
    PythonDictFramework,
)

# --------------------------------------------------------------------------- #
# Minimal root FeatureGroups whose calculate_feature raises a chosen error type
# --------------------------------------------------------------------------- #


class ImportErrorFeatureGroup(FeatureGroup):
    """Root FG whose ``calculate_feature`` raises a stdlib ``ImportError``."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_import_error_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ImportError("optional backend 'bm25s' missing")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_import_error_col"}


class MyDomainError(ValueError):
    """Custom domain error that is also a ``ValueError`` subclass."""


class DomainErrorFeatureGroup(FeatureGroup):
    """Root FG whose ``calculate_feature`` raises a ``ValueError`` subclass."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_domain_error_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise MyDomainError("domain rule violated")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_domain_error_col"}


class CauseChainFeatureGroup(FeatureGroup):
    """Root FG that raises ``ValueError`` from a ``KeyError`` (``__cause__`` chain)."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_cause_chain_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError("boom") from KeyError("root")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_cause_chain_col"}


_ENABLED_IMPORT_ERROR = PluginCollector.enabled_feature_groups({ImportErrorFeatureGroup})
_ENABLED_DOMAIN_ERROR = PluginCollector.enabled_feature_groups({DomainErrorFeatureGroup})
_ENABLED_CAUSE_CHAIN = PluginCollector.enabled_feature_groups({CauseChainFeatureGroup})


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


class SecretLeakFeatureGroup(FeatureGroup):
    """Root FG whose ``calculate_feature`` raises an ``OSError`` carrying three credential shapes."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_secret_leak_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise OSError(_LEAK_MESSAGE)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_secret_leak_col"}


class SecretLeakChainedFeatureGroup(FeatureGroup):
    """Root FG that raises ``RuntimeError`` from an ``OSError`` carrying the same credential shapes."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_secret_leak_chained_col"})

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


class UnpicklableSecretLeakFeatureGroup(FeatureGroup):
    """Root FG whose ``calculate_feature`` raises a non-picklable, secret-bearing exception."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_unpicklable_secret_col"})

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


class RowLeakFeatureGroup(FeatureGroup):
    """Root FG whose ``calculate_feature`` raises an error carrying row values."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_row_leak_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RowLeakError(f"Failed parsing row: {_ROW_LEAK_MARKER}")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_row_leak_col"}


class RowLeakChainedFeatureGroup(FeatureGroup):
    """Root FG that raises ``RuntimeError`` from a ``RowLeakError``."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_row_leak_chained_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("Calculation failed") from RowLeakError(f"Inner row error: {_ROW_LEAK_MARKER}")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"exc_row_leak_chained_col"}


_ENABLED_ROW_LEAK = PluginCollector.enabled_feature_groups({RowLeakFeatureGroup})
_ENABLED_ROW_LEAK_CHAINED = PluginCollector.enabled_feature_groups({RowLeakChainedFeatureGroup})


def test_sync_secret_leak_direct_raw_to_caller_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The caller keeps the raw secret-bearing message; ERROR log records never carry the secret."""
    with pytest.raises(OSError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert _LEAK_MARKER in str(excinfo.value)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_LEAK_MARKER not in r.getMessage() for r in error_records)


def test_sync_secret_leak_chained_raw_to_caller_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The chained cause keeps its raw secret-bearing message; ERROR log records never carry the secret."""
    with pytest.raises(RuntimeError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_chained_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK_CHAINED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert isinstance(excinfo.value.__cause__, OSError)
    assert _LEAK_MARKER in str(excinfo.value.__cause__)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_LEAK_MARKER not in r.getMessage() for r in error_records)


def test_threading_secret_leak_scrubbed_in_logs(flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(OSError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.THREADING},
            flight_server=flight_server,
        )

    assert _LEAK_MARKER in str(excinfo.value)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_LEAK_MARKER not in r.getMessage() for r in error_records)


def test_multiprocessing_secret_leak_scrubbed_in_logs(flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(OSError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_secret_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_SECRET_LEAK,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

    assert _LEAK_MARKER in str(excinfo.value)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_LEAK_MARKER not in r.getMessage() for r in error_records)


@pytest.mark.timeout(15)
def test_multiprocessing_unpicklable_secret_leak_message_is_scrubbed(flight_server: Any) -> None:
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


# --------------------------------------------------------------------------- #
# 1. SYNC mode type preservation
# --------------------------------------------------------------------------- #


def test_sync_preserves_import_error_type(caplog: pytest.LogCaptureFixture) -> None:
    """SYNC ``run_all`` must surface the original ``ImportError`` type, not a bare Exception.

    FAILS today: ``_check_for_error`` raises a bare ``Exception`` so
    ``pytest.raises(ImportError)`` does not match and the Exception propagates.
    It also logs the traceback exactly once, on an mloda logger, never on the root logger.
    """
    with pytest.raises(ImportError, match="bm25s"):
        mloda.run_all(
            [Feature(name="exc_import_error_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_IMPORT_ERROR,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert [r.name for r in caplog.records if r.name == "root"] == []
    traceback_records = [r for r in caplog.records if "Traceback" in r.getMessage()]
    assert [r.name for r in traceback_records] == ["mloda.core.runtime.run"]


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


class UnpicklableErrorFeatureGroup(FeatureGroup):
    """Root FG whose ``calculate_feature`` raises a non-picklable exception."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"exc_unpicklable_col"})

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
    """Row values in the exception message must not be logged to ERROR records."""
    with pytest.raises(RowLeakError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_row_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ROW_LEAK,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert _ROW_LEAK_MARKER in str(excinfo.value)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_ROW_LEAK_MARKER not in r.getMessage() for r in error_records)


def test_multiprocessing_row_leak_scrubbed_in_logs(flight_server: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Row values in the exception message must not be logged to ERROR records in MULTIPROCESSING."""
    with pytest.raises(RowLeakError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_row_leak_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ROW_LEAK,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

    assert _ROW_LEAK_MARKER in str(excinfo.value)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_ROW_LEAK_MARKER not in r.getMessage() for r in error_records)


def test_sync_row_leak_chained_scrubbed_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Row values in chained exception messages must not be logged to ERROR records."""
    with pytest.raises(RuntimeError) as excinfo:
        mloda.run_all(
            [Feature(name="exc_row_leak_chained_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_ROW_LEAK_CHAINED,
            parallelization_modes={ParallelizationMode.SYNC},
        )

    assert isinstance(excinfo.value.__cause__, RowLeakError)
    assert _ROW_LEAK_MARKER in str(excinfo.value.__cause__)
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    assert all(_ROW_LEAK_MARKER not in r.getMessage() for r in error_records)
