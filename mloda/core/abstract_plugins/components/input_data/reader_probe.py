import logging
from collections.abc import Iterator
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


def probe_reader(reader: Any, data_access: Any, feature_names: list[str], options: Any) -> Any:
    """Run reader.match_subclass_data_access, through the INPUT_DATA_MATCH extender when one is in scope."""
    scope = _CURRENT.get()
    if scope is None:
        return reader.match_subclass_data_access(data_access, feature_names, options=options)

    context = HookContext(
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
            logger.log(
                logging.INFO if scope.extender.never_fall_back else logging.WARNING,
                "Reader %s refused at INPUT_DATA_MATCH by %s: %s",
                reader.get_class_name(),
                type(scope.extender).__name__,
                reason,
            )
            record_match_rejection(
                reader.get_class_name(),
                f"refused at INPUT_DATA_MATCH: {reason}",
                stage=INPUT_DATA_STAGE,
            )
            return None
