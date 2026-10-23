import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.declared_attributes import read_declared_attributes
from mloda.core.abstract_plugins.components.utils import contained_raise_reason, is_match_abort, safe_field
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, _invoke_extender
from mloda.core.abstract_plugins.hook_context import HookContext


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReaderProbeScope:
    """The extender and planning identity active while reader probes run."""

    extender: Extender
    plan_id: str | None
    tenant_id: str | None
    project_id: str | None
    principal: str | None


_CURRENT: ContextVar[ReaderProbeScope | None] = ContextVar("reader_probe_scope", default=None)


@contextmanager
def reader_probe_scope(scope: ReaderProbeScope | None) -> Iterator[None]:
    """Make scope (or none) current for reader probes inside the block."""
    token = _CURRENT.set(scope)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@dataclass(frozen=True)
class SourceProbeOutcome:
    """refused is the refusal reason, None when the probe ran and result holds its value."""

    refused: str | None
    result: Any
    error: Exception | None = None


def _strip_tracebacks(exc: Exception) -> Exception:
    """Drop __traceback__ from exc and everything on its cause/context chain; returns exc."""
    seen: list[int] = []
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.append(id(current))
        current.__traceback__ = None
        pending.append(current.__cause__)
        pending.append(current.__context__)
    return exc


def _match_context(scope: ReaderProbeScope, reader: Any, feature_names: list[str]) -> HookContext:
    return HookContext(
        hook=ExtenderHook.INPUT_DATA_MATCH,
        feature_group_class=None,
        feature_group_version=None,
        compute_framework_name=None,
        feature_names=tuple(str(n) for n in feature_names),
        plan_id=scope.plan_id,
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        principal=scope.principal,
        reader_class=reader,
        data_access_format=reader.data_access_name(),
        declared_attributes=safe_field(lambda: read_declared_attributes(reader, None), None),
    )


def _log_refusal(scope: ReaderProbeScope, reader: Any, reason: str, label: str = "Reader") -> None:
    logger.log(
        logging.INFO if scope.extender.never_fall_back else logging.WARNING,
        "%s %s refused at INPUT_DATA_MATCH by %s: %s",
        label,
        reader.get_class_name(),
        type(scope.extender).__name__,
        reason,
    )


def probe_source_group(
    group: Any, probe: Callable[[], Any], data_access_collection: Any, base_name: str, options: Any
) -> SourceProbeOutcome:
    """Run a source group's source probe through the INPUT_DATA_MATCH extender when one is in scope."""
    scope = _CURRENT.get()
    if scope is None:
        return SourceProbeOutcome(None, probe())

    names = [base_name]
    finished = False
    result: Any = None
    probe_raised = False

    def func(access: Any, feature_names: list[str], options: Any) -> Any:
        nonlocal finished, result, probe_raised
        try:
            value = probe()
        except BaseException:
            probe_raised = True
            raise
        if not finished:
            finished = True
            result = value
        return None

    with _match_context(scope, group, names).activate():
        try:
            _invoke_extender(scope.extender, func, data_access_collection, names, options=options)
        # Contained: an extender refusing one candidate group declines it.
        except Exception as exc:
            if probe_raised or is_match_abort(exc):
                raise
            reason = contained_raise_reason(exc)
            _log_refusal(scope, group, reason, "Source group")
            return SourceProbeOutcome(reason, None, _strip_tracebacks(exc))
    if not finished:
        reason = "the extender did not run the probe"
        _log_refusal(scope, group, reason, "Source group")
        return SourceProbeOutcome(reason, None)
    return SourceProbeOutcome(None, result)
