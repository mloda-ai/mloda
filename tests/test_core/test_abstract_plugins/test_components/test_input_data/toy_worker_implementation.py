"""Test-only group and module-level Implementation in one importable module, for the spawn-worker case."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import SourceMatch
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_format_group import ToyFormatFG

WORKER_COLUMN = "toyfmt_worker"
WORKER_VALUES = [901, 902]


class ToyWorkerFG(ToyFormatFG):
    """Toy group that only this module's Implementation fast-paths."""


class ToyWorkerPythonDict(Implementation):
    feature_group = ToyWorkerFG
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(cls, group: Any, match: SourceMatch, features: Any) -> Any:
        return {WORKER_COLUMN: list(WORKER_VALUES)}
