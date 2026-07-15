"""PreFilterPlugins.degrade_on_error is opt-in: the engine path fails fast, resolve_feature degrades.

A FeatureGroup whose compute_framework_rule() raises must propagate on the default (engine) path so a
broken plugin surfaces, but degrade to an empty framework set when a caller opts in (resolve_feature).
The broken class is scoped per test (del + gc.collect()) so it cannot leak into other tests.
"""

import gc

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import PreFilterPlugins


def _make_broken_rule_fg() -> type[FeatureGroup]:
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class PrefilterDegradeBrokenRuleFG(FeatureGroup):
        """Its compute_framework_rule raises when consulted."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise RuntimeError("prefilter degrade rule exploded")

    return PrefilterDegradeBrokenRuleFG


class TestPreFilterDegradeOnError:
    """The framework-definition guard only applies when the caller opts in."""

    def test_default_engine_path_fails_fast(self) -> None:
        broken_fg = _make_broken_rule_fg()
        # A bound 'except ... as e' keeps a traceback that pins the broken class; use an unbound
        # except so nothing outlives the block and the class cannot leak to other tests.
        raised = False
        try:
            try:
                PreFilterPlugins(PreFilterPlugins.get_cfw_subclasses(), None)
            except RuntimeError:
                raised = True
        finally:
            del broken_fg
            gc.collect()
        assert raised

    def test_degrade_on_error_maps_broken_group_to_empty_set(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            mapping = PreFilterPlugins(
                PreFilterPlugins.get_cfw_subclasses(), None, degrade_on_error=True
            ).get_accessible_plugins()
            assert broken_fg in mapping
            assert mapping[broken_fg] == set()
            del mapping
        finally:
            del broken_fg
            gc.collect()
