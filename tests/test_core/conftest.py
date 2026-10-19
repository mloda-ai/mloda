"""Factory for function-local Implementation classes that are gone again when the test ends."""

import gc
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations


@pytest.fixture
def make_implementation() -> Iterator[Callable[..., type[Implementation]]]:
    """make(group, framework, fn=None, base=Implementation, name=None); teardown asserts every class is collected."""
    created: list[weakref.ref[type[Implementation]]] = []

    def make(
        group: Any,
        framework: Any,
        fn: Callable[[Any, Any, Any], Any] | None = None,
        base: type[Implementation] = Implementation,
        name: str | None = None,
    ) -> type[Implementation]:
        class _Impl(base):  # type: ignore[misc, valid-type]
            feature_group = group
            compute_framework = framework

            @classmethod
            def calculate(cls, serving: Any, match: Any, features: Any) -> Any:
                return None if fn is None else fn(serving, match, features)

        if name is not None:
            _Impl.__name__ = name
            _Impl.__qualname__ = name
        created.append(weakref.ref(_Impl))
        return _Impl

    yield make
    gc.collect()
    leaked = [ref() for ref in created if ref() is not None]
    assert not [impl for impl in loaded_implementations() if impl in leaked], f"leaked implementations: {leaked}"
