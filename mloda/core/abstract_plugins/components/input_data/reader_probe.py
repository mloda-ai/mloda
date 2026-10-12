import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.declared_attributes import read_declared_attributes
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
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
class FormatProbeOutcome:
    """refused is the refusal reason, None when the probe ran and result holds its value."""

    refused: str | None
    result: Any


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


def _log_refusal(scope: ReaderProbeScope, reader: Any, reason: str) -> None:
    logger.log(
        logging.INFO if scope.extender.never_fall_back else logging.WARNING,
        "Reader %s refused at INPUT_DATA_MATCH by %s: %s",
        reader.get_class_name(),
        type(scope.extender).__name__,
        reason,
    )


def probe_format_group(
    group: Any, probe: Callable[[], Any], data_access_collection: Any, base_name: str, options: Any
) -> FormatProbeOutcome:
    """Run a format group's source probe through the INPUT_DATA_MATCH extender when one is in scope."""
    scope = _CURRENT.get()
    if scope is None:
        return FormatProbeOutcome(None, probe())

    names = [base_name]
    ran = False
    result: Any = None
    probe_raised = False

    def func(access: Any, feature_names: list[str], options: Any) -> Any:
        nonlocal ran, result, probe_raised
        ran = True
        try:
            result = probe()
        except BaseException:
            probe_raised = True
            raise
        return None

    with _match_context(scope, group, names).activate():
        try:
            _invoke_extender(scope.extender, func, data_access_collection, names, options=options)
        # Contained: an extender refusing one candidate group declines it.
        except Exception as exc:
            if probe_raised or is_match_abort(exc):
                raise
            reason = contained_raise_reason(exc)
            _log_refusal(scope, group, reason)
            return FormatProbeOutcome(reason, None)
    if not ran:
        reason = "the extender did not run the probe"
        _log_refusal(scope, group, reason)
        return FormatProbeOutcome(reason, None)
    return FormatProbeOutcome(None, result)


def probe_reader(reader: Any, data_access: Any, feature_names: list[str], options: Any) -> Any:
    """Run reader.match_subclass_data_access, through the INPUT_DATA_MATCH extender when one is in scope."""
    scope = _CURRENT.get()
    if scope is None:
        return reader.match_subclass_data_access(data_access, feature_names, options=options)

    context = _match_context(scope, reader, feature_names)
    probe_raised = False

    def probe(access: Any, names: list[str], options: Any) -> Any:
        nonlocal probe_raised
        try:
            return reader.match_subclass_data_access(access, names, options=options)
        except BaseException:
            probe_raised = True
            raise

    with context.activate():
        try:
            return _invoke_extender(scope.extender, probe, data_access, feature_names, options=options)
        # Contained: an extender refusing one candidate reader skips it, like a reader that did not match.
        except Exception as exc:
            if probe_raised or is_match_abort(exc):
                raise
            reason = contained_raise_reason(exc)
            _log_refusal(scope, reader, reason)
            record_match_rejection(
                reader.get_class_name(),
                f"refused at INPUT_DATA_MATCH: {reason}",
                stage=INPUT_DATA_STAGE,
            )
            return None
