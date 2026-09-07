from __future__ import annotations

from typing import Any, Callable, TypeVar

import logging
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")

E = TypeVar("E", bound=BaseException)

# Provenance marker for a framework-owned raise. Not an exception type: the object must stay exactly as raised.
MATCH_ABORT_FLAG = "_mloda_match_abort"

# Exception classes a user callable raises when it merely cannot judge a value.
_EXPECTED_JUDGMENT_ERRORS: tuple[type[Exception], ...] = (TypeError, ValueError, AttributeError)


def contained_raise_log_level(exc: BaseException) -> int:
    """DEBUG for expected judgment failures, WARNING for classes that suggest a broken callable."""
    return logging.DEBUG if isinstance(exc, _EXPECTED_JUDGMENT_ERRORS) else logging.WARNING


def escalate_match_abort(exc: E) -> E:
    """Mark a framework-owned raise so the match seam re-raises it instead of containing it as a non-match.

    Mark-or-contain policy: see call_match_hook.
    """
    # __dict__, not setattr: setattr raises on a frozen-dataclass exception, and failing to mark must not
    # replace the exception being marked.
    try:
        exc.__dict__[MATCH_ABORT_FLAG] = True
    except Exception:  # noqa: BLE001  (marking is never worth losing the original raise)
        logger.debug("Could not mark %s as a match abort.", type(exc).__name__)
    return exc


def is_match_abort(exc: BaseException) -> bool:
    """Is this raise marked as framework-owned, so the match seam must not contain it."""
    # __dict__, not getattr: a custom __getattr__ could raise inside the seam's except block or fake the marker.
    return exc.__dict__.get(MATCH_ABORT_FLAG, False) is True


def safe_exc_str(exc: BaseException) -> str:
    """str(exc), guarded: a __str__ that itself raises degrades to the exception's type name."""
    try:
        return str(exc)
    except Exception:  # noqa: BLE001  (the exception's own __str__ is plugin-owned and must not escape)
        return type(exc).__name__


# Keys already warned via warn_once_for. Weak-referenceable keys (the common case: a class) live in the
# WeakSet, so a key is never pinned past its natural lifetime, consistent with
# _class_source_hash_cache's rationale in base_feature_group_version.py; keys that cannot be weakly
# referenced fall back to a plain set.
_warn_once_for_weak: "weakref.WeakSet[Any]" = weakref.WeakSet()
_warn_once_for_strong: set[object] = set()


def _warn_once_for_seen(key: object) -> bool:
    """True if `key` was already recorded by a prior warn_once_for call; otherwise records it and returns False.

    Total: any failure probing or recording `key` (a plugin-owned __hash__/__eq__/weakref hook can raise)
    degrades to "not seen", matching safe_exc_str/is_match_abort's never-break-the-safety-net idiom.
    """
    try:
        # type(key).__weakrefoffset__, not hasattr(key, "__weakref__"): the latter resolves through key's
        # MRO when key is a class, testing instances-of-key, not key itself.
        registry: "weakref.WeakSet[Any] | set[object]" = (
            _warn_once_for_weak if getattr(type(key), "__weakrefoffset__", 0) else _warn_once_for_strong
        )
        if key in registry:
            return True
        registry.add(key)
    except Exception:  # noqa: BLE001  (checking/recording the key is plugin-owned and must not escape)
        return False
    return False


def safe_field(
    read: Callable[[], T],
    fallback: T,
    catching: tuple[type[Exception], ...] = (Exception,),
    field: str = "",
    warn_once_for: object | None = None,
) -> T:
    """Annotate tier: degrade a single unreadable field to a fallback instead of failing the whole discovery call.

    A labelled read (non-empty `field`) warns on swallow; an unlabelled read degrades silently, because degrading
    there is expected. `warn_once_for` dedups that WARNING per key, so a hot call site warns only on the key's
    first swallow.
    """
    try:
        return read()
    except catching as exc:
        if field and (warn_once_for is None or not _warn_once_for_seen(warn_once_for)):
            # str(exc), not exc: a retained log record must not pin the traceback, its frames and the plugin class.
            logger.warning("Degraded field '%s': %s: %s", field, type(exc).__name__, safe_exc_str(exc))
        return fallback


def contained_raise_reason(exc: BaseException) -> str:
    """Text form of a contained raise: type and message, never the exception object."""
    return f"raised {type(exc).__name__}: {safe_exc_str(exc)}"


def safe_field_with_error(
    read: Callable[[], T],
    fallback: T,
    catching: tuple[type[Exception], ...] = (Exception,),
) -> tuple[T, str | None]:
    """Like safe_field but returns (value, None), else (fallback, str(exc) or the exception type name when blank)."""
    try:
        return read(), None
    except catching as exc:
        message = safe_exc_str(exc)
        return fallback, message if message.strip() else type(exc).__name__


def as_str(value: Any) -> str:
    """Return `value` unchanged, raising TypeError on a non-str so the guarded read that wraps it degrades."""
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def unhashable_part(value: Any, catching: tuple[type[Exception], ...] = (Exception,)) -> str | None:
    """Name of the first part of `value` whose hash raises one of `catching`, None when the whole value hashes."""
    # Probe the real hash, not isinstance(value, Hashable): a tuple carrying a dict and a __hash__ that
    # raises both report as hashable.
    if safe_field(lambda: isinstance(hash(value), int), False, catching=catching):
        return None
    if isinstance(value, tuple):
        for element in value:
            found = unhashable_part(element, catching=catching)
            if found is not None:
                return found
    return type(value).__name__


def get_all_subclasses(cls: Any) -> set[type[Any]]:
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses
