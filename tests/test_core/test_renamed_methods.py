"""Tests for method renames introduced by issue #266."""

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.input_data.base_input_data import get_all_filtered_subclasses
from mloda.core.filter.global_filter import GlobalFilter


class TestRenamedGetAllFilteredSubclasses:
    def test_importable(self) -> None:
        assert callable(get_all_filtered_subclasses)


class TestRenamedIdentifyMatchedFilters:
    def test_method_exists(self) -> None:
        gf = GlobalFilter()
        assert hasattr(gf, "identify_matched_filters")
        assert callable(gf.identify_matched_filters)


class TestRenamedIsAPartOf:
    def test_method_exists(self) -> None:
        idx = Index(("a",))
        assert hasattr(idx, "is_a_part_of")
        assert callable(idx.is_a_part_of)

    def test_single_is_part_of_multi(self) -> None:
        single = Index(("user_id",))
        multi = Index(("user_id", "timestamp"))
        assert single.is_a_part_of(multi) is True

    def test_multi_is_not_part_of_single(self) -> None:
        single = Index(("user_id",))
        multi = Index(("user_id", "timestamp"))
        assert multi.is_a_part_of(single) is False

    def test_equal_indexes(self) -> None:
        idx = Index(("user_id",))
        assert idx.is_a_part_of(Index(("user_id",))) is True

    def test_disjoint_indexes(self) -> None:
        a = Index(("user_id",))
        b = Index(("timestamp",))
        assert a.is_a_part_of(b) is False

    def test_old_name_removed(self) -> None:
        idx = Index(("a",))
        assert not hasattr(idx, "is_a_part_of_")
