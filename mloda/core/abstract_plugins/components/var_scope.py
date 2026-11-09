"""A class based scope for a ContextVar: unlike @contextmanager it never assigns __traceback__ on the exception."""

from contextvars import ContextVar, Token
from typing import Generic, TypeVar

T = TypeVar("T")


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
