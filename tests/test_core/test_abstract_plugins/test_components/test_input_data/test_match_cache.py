"""run_match_cache scope and run_cached memoization."""

from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached, run_match_cache


class TestRunCached:
    def test_outside_a_scope_computes_every_time(self) -> None:
        calls: list[int] = []

        def compute() -> int:
            calls.append(1)
            return len(calls)

        assert run_cached("k", compute) == 1
        assert run_cached("k", compute) == 2

    def test_inside_a_scope_computes_once_per_key(self) -> None:
        calls: list[str] = []

        def make(tag: str) -> object:
            calls.append(tag)
            return tag

        with run_match_cache():
            assert run_cached("a", lambda: make("a")) == "a"
            assert run_cached("a", lambda: make("a2")) == "a"
            assert run_cached("b", lambda: make("b")) == "b"

        assert calls == ["a", "b"]

    def test_nested_scope_reuses_the_outer_scope(self) -> None:
        calls: list[int] = []

        def compute() -> int:
            calls.append(1)
            return 7

        with run_match_cache():
            run_cached("k", compute)
            with run_match_cache():
                run_cached("k", compute)
            run_cached("k", compute)

        assert len(calls) == 1

    def test_a_new_scope_after_exit_starts_empty(self) -> None:
        calls: list[int] = []

        def compute() -> int:
            calls.append(1)
            return 1

        with run_match_cache():
            run_cached("k", compute)
        with run_match_cache():
            run_cached("k", compute)

        assert len(calls) == 2

    def test_cache_is_inactive_after_the_scope_exits(self) -> None:
        calls: list[int] = []

        def compute() -> int:
            calls.append(1)
            return 1

        with run_match_cache():
            run_cached("k", compute)
        run_cached("k", compute)
        run_cached("k", compute)

        assert len(calls) == 3

    def test_falsy_results_are_cached(self) -> None:
        calls: list[int] = []

        def compute() -> None:
            calls.append(1)

        with run_match_cache():
            assert run_cached("k", compute) is None
            assert run_cached("k", compute) is None

        assert len(calls) == 1
