"""Tests for CfwManager (mloda/core/core/cfw_manager.py).

set_child_bootstrap/get_child_bootstrap mirror the existing set_run_context/get_run_id/
get_carrier round-trip pattern: a caller may register a plain, picklable, no-argument callable
that mloda invokes once inside a spawned worker process, before that worker processes its
first command.
"""

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager


def _module_level_bootstrap() -> None:
    """Picklable module-level no-argument callable, standing in for a real bootstrap."""


class TestCfwManagerChildBootstrapDefault:
    """get_child_bootstrap() must return None until set_child_bootstrap() is called."""

    def test_default_child_bootstrap_is_none(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        assert cfw_register.get_child_bootstrap() is None


class TestCfwManagerChildBootstrapRoundTrip:
    """set_child_bootstrap() then get_child_bootstrap() must round-trip the exact callable."""

    def test_set_then_get_round_trips_a_module_level_picklable_callable(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        cfw_register.set_child_bootstrap(_module_level_bootstrap)

        assert cfw_register.get_child_bootstrap() is _module_level_bootstrap

    def test_set_none_after_a_previous_set_clears_it_back_to_none(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cfw_register.set_child_bootstrap(_module_level_bootstrap)

        cfw_register.set_child_bootstrap(None)

        assert cfw_register.get_child_bootstrap() is None
