"""FRAMEWORK_TRANSFORM hook wiring: one extender-wrapped call per conversion hop."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, _invoke_extender
from mloda.core.abstract_plugins.hook_context import HookContext, _no_rows, instrument
from mloda.core.abstract_plugins.run_context import RunContext

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework


def transform_hook_invoker(
    extender: Extender | None,
    run_context: RunContext,
    step_uuid: UUID | None,
    source: "type[ComputeFramework]",
    target: "type[ComputeFramework]",
    worker_index: int | None = None,
) -> Callable[..., Any] | None:
    """Return invoke(func, *args) running each hop through the extender; None without an extender."""
    if extender is None:
        return None

    def invoke(func: Callable[..., Any], *args: Any) -> Any:
        context = HookContext(
            hook=ExtenderHook.FRAMEWORK_TRANSFORM,
            feature_group_class=None,
            feature_group_version=None,
            compute_framework_name=target.get_class_name(),
            source_compute_framework=source,
            target_compute_framework=target,
            step_uuid=step_uuid,
            run_id=run_context.run_id,
            plan_id=run_context.plan_id,
            structure_hash=run_context.structure_hash,
            carrier=run_context.carrier,
            tenant_id=run_context.tenant_id,
            project_id=run_context.project_id,
            principal=run_context.principal,
            worker_index=worker_index,
        )
        with context.activate():
            return _invoke_extender(extender, instrument(context, func, row_count=_no_rows), *args)

    return invoke
