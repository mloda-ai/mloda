"""plan_freeze primitives: FrozenPlanError, ReadOnlyDict, ReadOnlySet."""

import pickle  # nosec B403
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.plan_freeze import FrozenPlanError, ReadOnlyDict, ReadOnlySet


def _ro() -> ReadOnlyDict:
    return ReadOnlyDict({"k": 1, "j": 2}, "options")


def test_frozen_plan_error_is_a_type_error() -> None:
    assert issubclass(FrozenPlanError, TypeError)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("k", 5),
        lambda d: d.__delitem__("k"),
        lambda d: d.update({"k": 5}),
        lambda d: d.pop("k"),
        lambda d: d.setdefault("k", 5),
        lambda d: d.__ior__({"k": 5}),
    ],
    ids=["setitem", "delitem", "update", "pop", "setdefault", "ior"],
)
def test_read_only_dict_mutators_raise_naming_the_key(mutate: Callable[[ReadOnlyDict], Any]) -> None:
    ro = _ro()
    with pytest.raises(FrozenPlanError, match="k"):
        mutate(ro)
    assert dict(ro) == {"k": 1, "j": 2}


def test_read_only_dict_popitem_and_clear_raise() -> None:
    ro = _ro()
    with pytest.raises(FrozenPlanError):
        ro.popitem()
    with pytest.raises(FrozenPlanError):
        ro.clear()
    assert dict(ro) == {"k": 1, "j": 2}


def test_read_only_dict_copies_are_plain_and_writable() -> None:
    ro = _ro()
    for plain in (dict(ro), {**ro}, ro.copy()):
        assert type(plain) is dict
        plain["new"] = 1
    assert "new" not in ro


def test_read_only_dict_equals_plain_dict() -> None:
    assert _ro() == {"k": 1, "j": 2}
    assert "__eq__" not in ReadOnlyDict.__dict__
    assert "__hash__" not in ReadOnlyDict.__dict__


def test_read_only_dict_pickle_keeps_type() -> None:
    restored = pickle.loads(pickle.dumps(_ro()))  # nosec B301
    assert type(restored) is ReadOnlyDict
    assert restored == {"k": 1, "j": 2}
    with pytest.raises(FrozenPlanError):
        restored["k"] = 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.add(9),
        lambda s: s.discard(1),
        lambda s: s.remove(1),
        lambda s: s.pop(),
        lambda s: s.clear(),
        lambda s: s.update({9}),
        lambda s: s.__ior__({9}),
        lambda s: s.__iand__({9}),
        lambda s: s.__isub__({1}),
        lambda s: s.__ixor__({1}),
    ],
    ids=["add", "discard", "remove", "pop", "clear", "update", "ior", "iand", "isub", "ixor"],
)
def test_read_only_set_mutators_raise(mutate: Callable[[ReadOnlySet], Any]) -> None:
    s = ReadOnlySet({1, 2})
    with pytest.raises(FrozenPlanError):
        mutate(s)
    assert set(s) == {1, 2}


def test_read_only_set_pickle_keeps_type_and_equals_plain_set() -> None:
    restored = pickle.loads(pickle.dumps(ReadOnlySet({1, 2})))  # nosec B301
    assert type(restored) is ReadOnlySet
    assert restored == {1, 2}
    with pytest.raises(FrozenPlanError):
        restored.add(3)
