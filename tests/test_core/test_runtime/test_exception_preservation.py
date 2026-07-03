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

from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda

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
    def input_data(cls) -> Optional[BaseInputData]:
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
    def input_data(cls) -> Optional[BaseInputData]:
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
    def input_data(cls) -> Optional[BaseInputData]:
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
# 1. SYNC mode type preservation
# --------------------------------------------------------------------------- #


def test_sync_preserves_import_error_type() -> None:
    """SYNC ``run_all`` must surface the original ``ImportError`` type, not a bare Exception.

    FAILS today: ``_check_for_error`` raises a bare ``Exception`` so
    ``pytest.raises(ImportError)`` does not match and the Exception propagates.
    """
    with pytest.raises(ImportError, match="bm25s"):
        mloda.run_all(
            [Feature(name="exc_import_error_col")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_ENABLED_IMPORT_ERROR,
            parallelization_modes={ParallelizationMode.SYNC},
        )


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
    (``get_error_exception()`` returns ``None`` -- e.g. the internal critical
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
    cfw_register.get_error_exception.return_value = None
    cfw_register.get_error_msg.return_value = "critical error_out"
    cfw_register.get_error_exc_info.return_value = "critical error_out"
    orchestrator.cfw_register = cfw_register

    with pytest.raises(MlodaRunError):
        orchestrator._check_for_error()
