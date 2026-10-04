"""Read-only dict used for ingested carriers and declared attributes."""

from typing import Any


def _read_only(self: Any, *args: Any, **kwargs: Any) -> Any:
    raise TypeError("read-only dict")


class _ReadOnlyDict(dict[Any, Any]):
    """dict that rejects mutation and pickles/copies as a plain dict."""

    __init__ = __setitem__ = __delitem__ = update = pop = popitem = clear = setdefault = __ior__ = _read_only

    def __new__(cls, *args: Any) -> Any:
        # Rebuilding via type(x)(items) (dataclasses.asdict) yields a plain dict; ingest uses _frozen_dict.
        return dict(*args)

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, (dict(self),))


def _frozen_dict(source: dict[Any, Any]) -> Any:
    frozen = dict.__new__(_ReadOnlyDict)
    dict.update(frozen, source)
    return frozen
