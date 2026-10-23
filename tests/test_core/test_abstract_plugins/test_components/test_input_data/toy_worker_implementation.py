"""Test-only group for the spawn-worker case; its Implementation lives in toy_worker_implementation_impl."""

from __future__ import annotations

from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG

WORKER_COLUMN = "toyfmt_worker"
WORKER_VALUES = [901, 902]


class ToyWorkerFG(ToySourceFG):
    """Toy group that only this module's Implementation fast-paths."""
