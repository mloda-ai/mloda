import logging
import weakref
from abc import ABC
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import ExtenderHook, GateBypassError, _invoke_extender
from mloda.core.abstract_plugins.hook_context import HookContext, input_data_load_gate_scopes_active, instrument
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.core.abstract_plugins.components.declared_attributes import read_declared_attributes

logger = logging.getLogger(__name__)

RESERVED_READER_OPTION_KEY = "BaseInputData"

_REMOVED_HOOKS = ("match_subclass_data_access", "load_data", "READER_OPTIONS", "init_reader", "match_data_access")


def _format_keys(keys: Iterable[str]) -> str:
    return "{" + ", ".join(sorted(set(keys))) + "}"


_fallback_identity_warned: "weakref.WeakSet[type]" = weakref.WeakSet()


def _is_fallback_identity(data_access: Any, identity: str) -> bool:
    """True when identity is the access's type name or its bare mapping key names."""
    if identity == type(data_access).__name__:
        return True
    return isinstance(data_access, Mapping) and identity == _format_keys(str(key) for key in data_access)


def dispatch_input_data_load(
    owner: Any,
    owner_type: Any,
    load: Callable[[Any, FeatureSet], Any],
    data_access: Any,
    features: FeatureSet,
    *,
    identity: Callable[[], str],
    data_access_format: Callable[[], str],
    loader_name: str | None = None,
) -> Any:
    """Run load through the INPUT_DATA_LOAD extender when one is registered, instrumenting it with a HookContext
    that inherits identity fields from the active calculate-phase HookContext."""
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework

    cfw = ComputeFramework.current()
    if cfw is None:
        marker = features._load_marker
        if marker is not None:
            active = marker.active
            if active is not None and marker.has_extender:
                with active[0].activate(), active[1].activate():
                    return dispatch_input_data_load(
                        owner,
                        owner_type,
                        load,
                        data_access,
                        features,
                        identity=identity,
                        data_access_format=data_access_format,
                        loader_name=loader_name,
                    )
            if marker.gate:
                raise GateBypassError(
                    f"{owner_type.__qualname__} loaded its source after its calculation finished under an "
                    "INPUT_DATA_LOAD gate."
                )
            return load(data_access, features)
        if input_data_load_gate_scopes_active() > 0:
            raise GateBypassError(
                f"{owner_type.__qualname__} loaded its source outside the calculation context while an "
                "INPUT_DATA_LOAD gate is active (a thread hop lost it); run the thread's work with "
                "contextvars.copy_context().run."
            )
        return load(data_access, features)

    extender = cfw.get_function_extender(ExtenderHook.INPUT_DATA_LOAD)
    if extender is None:
        return load(data_access, features)

    calc_context = HookContext.current()
    if calc_context is None:
        if extender.never_fall_back:
            raise GateBypassError(
                f"{owner_type.__qualname__} loaded its source without a calculate-phase HookContext under an "
                "INPUT_DATA_LOAD gate; run the thread's work with contextvars.copy_context().run."
            )
        return load(data_access, features)

    access_identity = identity()
    is_fallback = _is_fallback_identity(data_access, access_identity)
    if is_fallback and owner_type not in _fallback_identity_warned:
        _fallback_identity_warned.add(owner_type)
        logger.warning(
            "%s.data_access_identity fell back to %r, which names no source, so INPUT_DATA_LOAD extenders "
            "cannot tell its sources apart; override data_access_identity on %s.",
            owner_type.__qualname__,
            access_identity,
            owner_type.__qualname__,
        )

    context = HookContext(
        hook=ExtenderHook.INPUT_DATA_LOAD,
        feature_group_class=calc_context.feature_group_class,
        feature_group_version=calc_context.feature_group_version,
        plugin_version=calc_context.plugin_version,
        feature_names=calc_context.feature_names,
        specialized_from=calc_context.specialized_from,
        input_features=calc_context.input_features,
        input_feature_edges=calc_context.input_feature_edges,
        compute_framework_name=cfw.get_class_name(),
        run_id=calc_context.run_id,
        plan_id=calc_context.plan_id,
        structure_hash=calc_context.structure_hash,
        step_uuid=calc_context.step_uuid,
        carrier=calc_context.carrier,
        tenant_id=calc_context.tenant_id,
        project_id=calc_context.project_id,
        principal=calc_context.principal,
        worker_index=calc_context.worker_index,
        data_access_identity=access_identity,
        data_access_identity_is_fallback=is_fallback,
        data_access_format=data_access_format(),
        data_access_dataset_version=None,
        data_access_loader=loader_name,
        implementation=calc_context.implementation,
        implementation_version=calc_context.implementation_version,
        declared_attributes=safe_field(
            lambda: read_declared_attributes(owner, features),
            None,
            field=f"{owner_type.__qualname__}.declared_attributes",
            warn_once_for=owner_type,
        ),
        reader_class=owner_type,
    )
    with context.activate():
        return _invoke_extender(extender, instrument(context, load, row_count=cfw._row_count), data_access, features)


class BaseInputData(ABC):
    """Marker for what a group's input_data() returns; reading files and sources is a FormatFeatureGroup."""

    def __init__(self) -> None:
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject the removed reader hooks at class definition, naming the replacement."""
        for klass in cls.__mro__:
            if klass is object or (klass is not cls and issubclass(klass, BaseInputData)):
                continue
            for hook in _REMOVED_HOOKS:
                if hook in klass.__dict__:
                    raise TypeError(
                        f"{cls.__name__} defines {hook}, which BaseInputData no longer supports; "
                        "subclass FormatFeatureGroup to read a source."
                    )
        super().__init_subclass__(**kwargs)

    @classmethod
    def data_access_name(cls) -> str:
        """This function should return the name of the data access."""
        return cls.__name__

    def matches(
        self,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        """A bare BaseInputData claims nothing; DataCreator and ApiInputData override this."""
        return False

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__
