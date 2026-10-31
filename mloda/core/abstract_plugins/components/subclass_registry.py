import gc
import threading
import weakref
from typing import Generic, TypeVar

T = TypeVar("T")


class SubclassRegistry(Generic[T]):
    """Weak registry of live subclasses with a cached snapshot that never keeps a dead class alive."""

    def __init__(self) -> None:
        self._members: weakref.WeakSet[T] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._snapshot: frozenset[T] | None = None
        self._gc_epoch = 0
        gc.callbacks.append(self._on_gc)

    def add(self, cls: T) -> None:
        with self._lock:
            self._members.add(cls)
            self._snapshot = None

    def members(self) -> set[T]:
        with self._lock:
            snapshot = self._snapshot  # local copy: the gc hook may clear the attribute at any time
            if snapshot is None:
                epoch = self._gc_epoch
                snapshot = frozenset(self._members)
                self._snapshot = snapshot
                if epoch != self._gc_epoch:  # checked after the store so a gc pass starting mid-build is caught
                    self._snapshot = None
        return set(snapshot)

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        # Classes die only in a gc pass, so dropping the snapshot at its start keeps them collectable.
        # No lock: a pass can start while this thread holds it.
        if phase == "start":
            self._gc_epoch += 1
            self._snapshot = None
