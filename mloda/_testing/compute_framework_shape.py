"""Shape check for a ComputeFramework: construction, is_available and the engine accessors are well formed."""

from __future__ import annotations

from uuid import UUID

from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass, safe_field_with_error
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.filter.filter_engine import BaseFilterEngine

_ABSENT = object()
_ASPECT_ACCESSORS = (("filter", "filter_engine"), ("mask", "mask_engine"), ("merge", "merge_engine"))
_Read = tuple[object, str | None]


def _construct_failures(cls: type[ComputeFramework]) -> list[str]:
    first, error = safe_field_with_error(lambda: cls(), None)
    if error is not None or first is None:
        return [f"construct: {cls.__name__}() failed: {error}"]
    second, error = safe_field_with_error(lambda: cls(), None)
    if error is not None or second is None:
        return [f"construct: {cls.__name__}() failed on a second call: {error}"]
    if not isinstance(first.uuid, UUID):
        return [f"construct: {cls.__name__}().uuid is not a UUID"]
    if first.uuid == second.uuid:
        return ["construct: two instances share the same uuid"]
    given = first.uuid
    seeded, error = safe_field_with_error(lambda: cls(uuid=given), None)
    if error is not None or seeded is None:
        return [f"construct: {cls.__name__} must accept the uuid= keyword: {error}"]
    if seeded.uuid != given:
        return [f"construct: {cls.__name__} must keep the uuid= keyword value as .uuid"]
    return []


def _read_accessor(cls: type[ComputeFramework], name: str) -> _Read:
    """Return (value, error); NotImplementedError reads as (_ABSENT, None)."""

    def read() -> _Read:
        return safe_field_with_error(lambda: getattr(cls, name)(), _ABSENT, (NotImplementedError,))

    fallback: _Read = (_ABSENT, None)
    result, error = safe_field_with_error(read, fallback)
    if error is not None:
        return _ABSENT, error
    return result[0], None


def _engine_failure(cls: type[ComputeFramework], name: str, base: type) -> str | None:
    value, error = _read_accessor(cls, name)
    if error is not None:
        return f"{name}: raised {error}"
    if value is _ABSENT or is_nominal_subclass(value, base):
        return None
    return f"{name}: must return a type subclassing {base.__name__}, got {value!r}"


def _mask_failure(cls: type[ComputeFramework]) -> str | None:
    value, error = _read_accessor(cls, "mask_engine")
    if error is not None:
        return f"mask_engine: raised {error}"
    if value is _ABSENT or value is None or is_nominal_subclass(value, BaseMaskEngine):
        return None
    return f"mask_engine: must return None or a type subclassing BaseMaskEngine, got {value!r}"


def _accessor_failures(cls: type[ComputeFramework]) -> list[str]:
    failures: list[str] = []
    data, error = safe_field_with_error(lambda: cls.expected_data_framework(), None)
    if error is not None:
        failures.append(f"expected_data_framework: raised {error}")
    elif data is not None and not isinstance(data, type):
        failures.append(f"expected_data_framework: must return None or a type, got {data!r}")
    for failure in (
        _engine_failure(cls, "filter_engine", BaseFilterEngine),
        _engine_failure(cls, "merge_engine", BaseMergeEngine),
        _mask_failure(cls),
    ):
        if failure is not None:
            failures.append(failure)
    return failures


def _is_available(cls: type[ComputeFramework]) -> tuple[bool, str | None]:
    value, error = safe_field_with_error(lambda: cls.is_available(), None)
    if error is not None:
        return False, f"is_available: raised {error}"
    if type(value) is not bool:
        return False, f"is_available: must return a bool, got {value!r}"
    return value, None


def applicable_aspects(cls: type[ComputeFramework]) -> tuple[str, ...]:
    """Return the subset of filter, mask, merge the framework serves; empty when it is unavailable."""
    available, _ = _is_available(cls)
    if not available:
        return ()
    aspects: list[str] = []
    for aspect, name in _ASPECT_ACCESSORS:
        value, error = _read_accessor(cls, name)
        if error is None and value is not _ABSENT and (aspect != "mask" or value is not None):
            aspects.append(aspect)
    return tuple(aspects)


def check_compute_framework_shape(cls: type[ComputeFramework]) -> CheckResult:
    """Check construction (including uuid=), is_available and the accessors.

    Accessors are skipped for unavailable frameworks."""
    failures = _construct_failures(cls)
    available, failure = _is_available(cls)
    if failure is not None:
        failures.append(failure)
    if available:
        failures.extend(_accessor_failures(cls))
    return CheckResult(check="compute_framework_shape", subject=cls.__name__, failures=tuple(failures))


def assert_compute_framework_shape(cls: type[ComputeFramework]) -> None:
    """Fail the running pytest test if the compute framework shape check reports failures."""
    assert_passed(check_compute_framework_shape(cls))
