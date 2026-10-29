"""The chain-check scope installs a lazy, memoized step-aside check for one feature name."""

from mloda.core.abstract_plugins.components.input_data.claim_route import chain_check_scope, computed_elsewhere


class _Computes:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.answer


def test_no_check_is_installed_by_default() -> None:
    assert computed_elsewhere("name") is False


def test_the_check_is_not_evaluated_until_consulted() -> None:
    computes = _Computes(True)

    with chain_check_scope("name", computes):
        assert computes.calls == 0

    assert computes.calls == 0


def test_the_check_is_evaluated_once_and_memoized() -> None:
    computes = _Computes(True)

    with chain_check_scope("name", computes):
        assert computed_elsewhere("name") is True
        assert computed_elsewhere("name") is True

    assert computes.calls == 1


def test_a_false_answer_is_memoized_too() -> None:
    computes = _Computes(False)

    with chain_check_scope("name", computes):
        assert computed_elsewhere("name") is False
        assert computed_elsewhere("name") is False

    assert computes.calls == 1


def test_another_name_is_false_and_never_evaluates() -> None:
    computes = _Computes(True)

    with chain_check_scope("name", computes):
        assert computed_elsewhere("other") is False

    assert computes.calls == 0


def test_the_scope_resets_on_exit() -> None:
    computes = _Computes(True)

    with chain_check_scope("name", computes):
        pass

    assert computed_elsewhere("name") is False
    assert computes.calls == 0


def test_a_reentrant_consult_while_running_answers_false() -> None:
    seen: list[bool] = []

    def computes() -> bool:
        seen.append(computed_elsewhere("name"))
        return True

    with chain_check_scope("name", computes):
        assert computed_elsewhere("name") is True

    assert seen == [False]


def test_scopes_nest_and_restore_the_outer_check() -> None:
    outer, inner = _Computes(True), _Computes(False)

    with chain_check_scope("name", outer):
        with chain_check_scope("name", inner):
            assert computed_elsewhere("name") is False
        assert computed_elsewhere("name") is True

    assert (outer.calls, inner.calls) == (1, 1)
