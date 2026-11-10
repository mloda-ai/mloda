"""A class based scope for a ContextVar: unlike @contextmanager it never assigns __traceback__ on the exception."""

from contextvars import ContextVar, Token
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class VarScope(Generic[T]):
    """Set a ContextVar to a value for a block, return the value on enter and reset on exit."""

    def __init__(self, var: ContextVar[T], value: T) -> None:
        self._var = var
        self._value = value
        self._tokens: list[Token[T]] = []

    def __enter__(self) -> T:
        self._tokens.append(self._var.set(self._value))
        return self._value

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        self._var.reset(self._tokens.pop())


class ResultScope(Generic[T, R]):
    """A VarScope whose enter returns a separate result object instead of the value."""

    def __init__(self, var: ContextVar[T], value: T, result: R) -> None:
        self._scope = VarScope(var, value)
        self._result = result

    def __enter__(self) -> R:
        self._scope.__enter__()
        return self._result

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        self._scope.__exit__(exc_type, exc, tb)


class PairScope:
    """Enter two VarScopes in order and exit them in reverse."""

    def __init__(self, first: VarScope[Any], second: VarScope[Any]) -> None:
        self._first = first
        self._second = second

    def __enter__(self) -> None:
        self._first.__enter__()
        self._second.__enter__()

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        self._second.__exit__(exc_type, exc, tb)
        self._first.__exit__(exc_type, exc, tb)
