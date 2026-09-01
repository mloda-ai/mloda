"""Pins the three new ExtenderHook members for feature-group-matched, input-data-load, and join hooks."""

from mloda.core.abstract_plugins.function_extender import ExtenderHook


class TestExtenderHookNewMembers:
    """New ExtenderHook members required for Phase 1 of the match/fetch/join extender hooks."""

    def test_feature_group_matched_member_exists_with_value(self) -> None:
        assert ExtenderHook.FEATURE_GROUP_MATCHED.value == "feature_group_matched"

    def test_input_data_load_member_exists_with_value(self) -> None:
        assert ExtenderHook.INPUT_DATA_LOAD.value == "input_data_load"

    def test_join_member_exists_with_value(self) -> None:
        assert ExtenderHook.JOIN.value == "join"
