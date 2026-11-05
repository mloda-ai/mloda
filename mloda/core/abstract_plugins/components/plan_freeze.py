"""Read-only containers and the error raised when a compiled plan is written to."""

from collections.abc import Iterable
from typing import Any, NoReturn

_ALTERNATIVES = (
    "Set it before prepare, or pass per-run values as run inputs "
    "(run(api_data=...), run(artifacts=...) read through the step view)."
)


class FrozenPlanError(TypeError):
    """Raised on a write into a plan that mloda.prepare() already compiled."""


def option_write_error(key: Any, label: str = "options") -> FrozenPlanError:
    return FrozenPlanError(
        f"Cannot write {label} key {key!r}: {label} are fixed once mloda.prepare() compiled the plan. {_ALTERNATIVES}"
    )


def check_options_writable(options: Any, key: Any = "<options>") -> None:
    """Bound as Options._check_writable; a def of that name would taint every set() call in the match sweep."""
    if options.__dict__.get("_frozen"):
        raise option_write_error(key)


def attribute_write_error(owner: str, attribute: str) -> FrozenPlanError:
    return FrozenPlanError(
        f"Cannot set {attribute!r} on a frozen {owner}: the plan is fixed once mloda.prepare() compiled it. "
        f"{_ALTERNATIVES}"
    )


class ReadOnlyDict(dict[Any, Any]):
    """A dict that raises FrozenPlanError on every write; dict(), {**d} and copy() give plain dicts."""

    def __init__(self, data: Any, label: str) -> None:
        super().__init__(data)
        self.label = label

    def _blocked_key(self, key: Any, *_: Any) -> NoReturn:
        raise option_write_error(key, self.label)

    def _blocked_many(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise option_write_error(sorted({*dict(*args), *kwargs}, key=str), self.label)

    def _blocked_all(self, *_: Any) -> NoReturn:
        raise option_write_error("<all>", self.label)

    __setitem__ = __delitem__ = pop = setdefault = _blocked_key
    update = __ior__ = _blocked_many
    popitem = clear = _blocked_all

    def __reduce__(self) -> tuple[Any, ...]:
        return (ReadOnlyDict, (dict(self), self.label))


class ReadOnlySet(set[Any]):
    """A set that raises FrozenPlanError on every write; set(), copy() give plain sets."""

    def __init__(self, iterable: Iterable[Any] = (), label: str = "features") -> None:
        super().__init__(iterable)
        self.label = label

    def _blocked(self, *_: Any) -> NoReturn:
        raise FrozenPlanError(
            f"Cannot modify the {self.label} set: the plan is fixed once mloda.prepare() compiled it. {_ALTERNATIVES}"
        )

    add = discard = remove = pop = clear = update = _blocked
    difference_update = intersection_update = symmetric_difference_update = _blocked
    __ior__ = __iand__ = __isub__ = __ixor__ = _blocked

    def __reduce__(self) -> tuple[Any, ...]:
        return (ReadOnlySet, (list(self), self.label))
