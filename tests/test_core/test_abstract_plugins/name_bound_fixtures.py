"""Shared throwaway feature groups for name-bound option tests (pattern group plus root source).

Classes are created per call with a caller-chosen prefix (unique per call site) and the caller's
``__module__``, so the autouse registry no-leak guard still attributes them to the calling test module.
"""

from __future__ import annotations

from typing import Any, Callable

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import FeatureChainParserMixin, DataCreatorFeatureGroup, PropertySpec


def make_name_bound_fg(
    prefix: str,
    module: str,
    tag: str,
    bound_keys: tuple[str, ...],
    unbound_keys: tuple[str, ...] = (),
    defaults: dict[str, str] | None = None,
    seen: dict[str, Any] | None = None,
    input_column: str | None = None,
    bound_context: bool = True,
    converters: dict[str, Callable[[str], Any]] | None = None,
    capture: str = "[a-z]",
    calls: list[list[str]] | None = None,
) -> type[FeatureGroup]:
    """Pattern group binding ``bound_keys`` from ``<name>__<tag>_<x>_<y>``; records option tuples in ``seen``.

    ``defaults`` maps key to its declared default (None declares none). Without ``input_column`` the
    group returns ``data`` unchanged; with it, each feature gets that input column. ``converters`` maps
    key to its ``name_converter``; ``capture`` is the regex class of each captured segment. ``calls``
    gets one list of feature names (repeats kept) appended per calculate_feature call.
    """
    pattern = rf".*__{tag}_" + "_".join(rf"(?P<{key}>{capture})" for key in bound_keys) + "$"
    keys = (*bound_keys, *unbound_keys)
    mapping = {}
    for key in keys:
        context = bound_context if key in bound_keys else True
        extra: dict[str, Any] = {}
        if converters is not None and key in converters:
            extra["name_converter"] = converters[key]
        if defaults is None:
            mapping[key] = PropertySpec("Declared key.", context=context, **extra)
        else:
            mapping[key] = PropertySpec("Declared key.", context=context, default=defaults[key], **extra)

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        if calls is not None:
            calls.append([str(feature.name) for feature in features.features])
        if seen is not None:
            for feature in features.features:
                seen[str(feature.name)] = tuple(feature.options.get(key) for key in keys)
        if input_column is None:
            return data
        return {str(feature.name): data[input_column] for feature in features.features}

    namespace = {
        "__module__": module,
        "PREFIX_PATTERN": pattern,
        "MIN_IN_FEATURES": 1,
        "MAX_IN_FEATURES": 1,
        "PROPERTY_MAPPING": mapping,
        "calculate_feature": classmethod(calculate_feature),
    }
    return type(f"{prefix}NameBoundFeatureGroup", (FeatureChainParserMixin, FeatureGroup), namespace)


def make_root_fg(
    prefix: str,
    module: str,
    column: str,
    rows: list[Any],
    seen: dict[str, Any] | None = None,
    observe_key: str | None = None,
) -> type[FeatureGroup]:
    """Root group serving ``column``; with ``seen`` and ``observe_key`` it records that option as ``seen["child"]``."""

    def feature_names_supported(cls: Any) -> set[str]:
        return {column}

    def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
        if seen is not None and observe_key is not None:
            for feature in features.features:
                seen["child"] = feature.options.get(observe_key)
        return {column: rows}

    namespace = {
        "__module__": module,
        "feature_names_supported": classmethod(feature_names_supported),
        "calculate_feature": classmethod(calculate_feature),
    }
    return type(f"{prefix}RootFeatureGroup", (DataCreatorFeatureGroup,), namespace)
