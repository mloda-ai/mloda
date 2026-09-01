"""Tests for CfwManager child_bootstrap get/set round-trip."""

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager


def _module_level_bootstrap() -> None:
    pass


class TestCfwManagerChildBootstrapDefault:
    def test_default_child_bootstrap_is_none(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        assert cfw_register.get_child_bootstrap() is None


class TestCfwManagerChildBootstrapRoundTrip:
    def test_set_then_get_round_trips_a_module_level_picklable_callable(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})

        cfw_register.set_child_bootstrap(_module_level_bootstrap)

        assert cfw_register.get_child_bootstrap() is _module_level_bootstrap

    def test_set_none_after_a_previous_set_clears_it_back_to_none(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        cfw_register.set_child_bootstrap(_module_level_bootstrap)

        cfw_register.set_child_bootstrap(None)

        assert cfw_register.get_child_bootstrap() is None
