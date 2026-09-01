"""Tests for CfwManager child_bootstrap get/set round-trip and merge-relation cycles."""

from uuid import uuid4

import pytest

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


class TestCfwManagerFindLeftmostCycleDetection:
    """A merge-relation cycle must raise, not hang find_leftmost forever."""

    @pytest.mark.timeout(2)
    def test_find_leftmost_raises_value_error_on_a_two_node_cycle(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        uuid_a = uuid4()
        uuid_b = uuid4()
        cls_name = "SomeComputeFramework"

        cfw_register.add_to_merge_relation(uuid_a, uuid_b, cls_name)
        cfw_register.add_to_merge_relation(uuid_b, uuid_a, cls_name)

        with pytest.raises(ValueError):
            cfw_register.find_leftmost(uuid_a, cls_name)
