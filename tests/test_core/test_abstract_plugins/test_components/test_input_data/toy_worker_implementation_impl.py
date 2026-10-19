"""Module-level Implementation for ToyWorkerFG, imported only by the worker test so the group's module alone lacks it."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import SourceMatch
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_worker_implementation import (
    WORKER_COLUMN,
    WORKER_VALUES,
    ToyWorkerFG,
)


class ToyWorkerPythonDict(Implementation):
    feature_group = ToyWorkerFG
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(cls, group: Any, match: SourceMatch, features: Any) -> Any:
        return {WORKER_COLUMN: list(WORKER_VALUES)}
