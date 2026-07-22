from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING, cast
from copy import deepcopy

from mloda.core.abstract_plugins.components.hashable_dict import _make_hashable
from mloda.core.abstract_plugins.components.validators.options_validator import OptionsValidator
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature import Feature

logger = logging.getLogger(__name__)

NON_FORWARDED_KEYS: frozenset[str] = frozenset({DefaultOptionKeys.in_features})


def _safe_deepcopy(value: Any, memo: dict[int, Any]) -> Any:
    """Deep-copy a single value, falling back to sharing the value by reference when it cannot be
    deep-copied for ANY reason (e.g. unpicklable objects, or values whose deepcopy raises under some
    Python versions such as uuid.UUID on 3.14)."""
    try:
        return deepcopy(value, memo)
    except Exception:
        # If the value cannot be deep-copied for any reason, share it by reference.
        return value


def _isolate_forwarded_value(value: Any, memo: dict[int, Any]) -> Any:
    """Copy the mutable-container spine so nested-container mutation cannot leak back to the
    consumer, while sharing every non-container leaf (validators, models, handles, unpicklable
    objects) by reference to preserve the identity the framework relies on for dedup, hashing,
    and conflict detection. Custom container types (anything other than dict/list/set/tuple) are
    shared by reference (documented limitation)."""
    vid = id(value)
    if vid in memo:
        return memo[vid]
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        memo[vid] = result
        for k, v in value.items():
            result[k] = _isolate_forwarded_value(v, memo)
        return result
    if isinstance(value, list):
        result_list: list[Any] = []
        memo[vid] = result_list
        for v in value:
            result_list.append(_isolate_forwarded_value(v, memo))
        return result_list
    if isinstance(value, set):
        result_set = {_isolate_forwarded_value(v, memo) for v in value}
        memo[vid] = result_set
        return result_set
    if isinstance(value, tuple):
        result_tuple = tuple(_isolate_forwarded_value(v, memo) for v in value)
        memo[vid] = result_tuple
        return result_tuple
    return value


def _normalize_reader_class_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Reader class keys normalize to data_access_name() so both documented key forms
    (class object and class-name string) share one identity."""
    if not any(isinstance(k, type) for k in d):
        return d
    return {
        (k.data_access_name() if isinstance(k, type) and hasattr(k, "data_access_name") else k): v for k, v in d.items()
    }


def validate_forwarding_directives(
    forward_group: frozenset[str] | bool | None, forward_group_exclude: frozenset[str]
) -> None:
    """Reject the contradictory forward_group=False + non-empty forward_group_exclude combination."""
    if forward_group is False and forward_group_exclude:
        raise ValueError(
            "forward_group=False cannot be combined with a non-empty forward_group_exclude: "
            "opting out of all group options contradicts excluding single keys."
        )


class Options:
    """
    Configuration container for features with group/context separation.

    Architecture:
    - group: Parameters affecting Feature Group resolution/splitting (used in hashing/equality)
    - context: Metadata parameters that don't affect splitting (excluded from hashing)

    Initialization:
    - Options() - Empty options (both group and context are empty)
    - Options({...}) - Positional dict goes to group
    - Options(group={...}) - Explicit group parameters
    - Options(context={...}) - Explicit context parameters
    - Options(group={...}, context={...}) - Both specified

    Common Methods:
    - .get(key[, default]) - Read value (searches group, then context; default when absent)
    - .set(key, value) - Write value (auto-placement)
    - .items() / .keys() - Iterate over all options
    - key in options - Check existence

    Direct Access (when category matters):
    - .group dict or .add_to_group(key, value)
    - .context dict or .add_to_context(key, value)

    Constraint: A key cannot exist in both group and context simultaneously.

    Note: Option keys are NOT validated at construction time. Valid keys depend
    on which feature groups resolve the feature at runtime. Typos in key names
    will not raise errors; the key will simply be ignored by all feature groups.
    Use ``DefaultOptionKeys`` constants where available to prevent typos.

    Examples:
        >>> # Basic usage with positional dict (goes to group)
        >>> opts = Options({"data_source": "prod"})
        >>> opts.group
        {'data_source': 'prod'}

        >>> # Explicit group/context separation
        >>> opts = Options(
        ...     group={"data_source": "prod"},
        ...     context={"debug_mode": True}
        ... )
        >>> opts.get("data_source")
        'prod'
        >>> opts.get("debug_mode")
        True

        >>> # Using helper methods
        >>> opts = Options()
        >>> opts.add_to_group("model_type", "classifier")
        >>> opts.add_to_context("log_level", "INFO")
        >>> "model_type" in opts
        True
    """

    def __init__(
        self,
        group: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        propagate_context_keys: frozenset[str] | None = None,
    ) -> None:
        self.group = _normalize_reader_class_keys(group) if group else {}
        self.context = _normalize_reader_class_keys(context) if context else {}
        self.propagate_context_keys: frozenset[str] = propagate_context_keys or frozenset()
        self.inherited_group_keys: frozenset[str] = frozenset()
        self.inherited_context_keys: frozenset[str] = frozenset()
        self.last_forwarded_group_keys: frozenset[str] = frozenset()
        OptionsValidator.validate_no_duplicate_keys(self.group, self.context)
        OptionsValidator.validate_propagate_keys_in_context(self.propagate_context_keys, self.context)

    def add_to_group(self, key: str, value: Any) -> None:
        """Add parameter to group (affects Feature Group resolution/splitting)."""
        OptionsValidator.validate_can_add_to_group(key, value, self.group, self.context)
        self.group[key] = value

    def add_to_context(self, key: str, value: Any) -> None:
        """Add parameter to context (metadata only, doesn't affect splitting)."""
        OptionsValidator.validate_can_add_to_context(key, value, self.group, self.context)
        self.context[key] = value

    def __hash__(self) -> int:
        """
        Hash based only on group parameters.
        Context parameters don't affect Feature Group resolution/splitting.
        """
        return hash(_make_hashable(self.group))

    def __eq__(self, other: object) -> bool:
        """
        Equality based only on group parameters.
        Context parameters don't affect Feature Group resolution/splitting.
        """
        if not isinstance(other, Options):
            return False
        return self.group == other.group

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value, searching group then context; return ``default`` only when the key is absent from both.

        Dict-style like ``dict.get``: a present key returns its stored value even when falsy
        (``0``, ``""``, ``False``, ``None``); ``default`` is used only for an absent key. Single-argument
        ``get(key)`` returns ``None`` for an absent key, unchanged.
        """
        if key in self.group:
            return self.group[key]
        if key in self.context:
            return self.context[key]
        return default

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def items(self) -> list[tuple[str, Any]]:
        """
        Get all key-value pairs from both group and context.

        Returns a list of tuples containing all options.
        Group options are returned first, followed by context options.
        """
        return list(self.group.items()) + list(self.context.items())

    def keys(self) -> list[str]:
        """
        Get all keys from both group and context.

        Returns a list of all option keys.
        """
        return list(self.group.keys()) + list(self.context.keys())

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in either group or context.

        Supports the 'in' operator: 'key' in options
        """
        return key in self.group or key in self.context

    def set(self, key: str, value: Any) -> None:
        """
        Set a value, automatically placing it in group or context.

        If the key already exists, update it in its current location.
        If the key is new, add it to group by default.
        """
        if key in self.group:
            self.group[key] = value
        elif key in self.context:
            self.context[key] = value
        else:
            # New key, add to group by default
            self.group[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def get_in_features(self) -> "frozenset[Feature]":
        val = self.get(DefaultOptionKeys.in_features)

        if not val:
            raise ValueError(
                f"Input features not found in options. Please ensure that the key '{DefaultOptionKeys.in_features}' is set."
            )

        def _convert_to_feature(item: Any) -> "Feature":
            """Convert item to Feature object if possible."""
            if hasattr(item, "options"):  # Already a Feature object
                return cast("Feature", item)
            elif isinstance(item, str):
                # Import Feature locally to avoid circular import
                from mloda.core.abstract_plugins.components.feature import Feature

                return Feature(item)
            else:
                raise TypeError(f"Cannot convert {type(item)} to Feature. Expected Feature object or str.")

        if isinstance(val, (list, tuple, set, frozenset)):
            return frozenset(_convert_to_feature(item) for item in val)
        elif isinstance(val, str):
            # Handle comma-separated strings
            if "," in val:
                feature_names = [name.strip() for name in val.split(",")]
                return frozenset(_convert_to_feature(name) for name in feature_names)
            else:
                return frozenset([_convert_to_feature(val)])
        elif hasattr(val, "options"):  # Handle Feature objects
            return frozenset([_convert_to_feature(val)])
        else:
            raise TypeError(
                f"Unsupported type for source feature: {type(val)}. "
                "Expected list, tuple, set, frozenset, str, or Feature object."
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> "Options":
        def safe_deepcopy_dict(d: dict[str, Any]) -> dict[str, Any]:
            """Safely deepcopy a dictionary, falling back to shallow copy for unpickleable objects."""
            return {key: _safe_deepcopy(value, memo) for key, value in d.items()}

        copied_group = safe_deepcopy_dict(self.group)
        copied_context = safe_deepcopy_dict(self.context)
        copied = Options(group=copied_group, context=copied_context, propagate_context_keys=self.propagate_context_keys)
        copied.inherited_group_keys = self.inherited_group_keys
        copied.inherited_context_keys = self.inherited_context_keys
        copied.last_forwarded_group_keys = self.last_forwarded_group_keys
        return copied

    def __str__(self) -> str:
        parts = f"Options(group={self.group}, context={self.context}"
        if self.propagate_context_keys:
            parts += f", propagate_context_keys={self.propagate_context_keys}"
        parts += ")"
        return parts

    def inherit_from(
        self,
        consumer: "Options",
        forward_group: frozenset[str] | bool | None = None,
        forward_group_exclude: frozenset[str] = frozenset(),
        inherit_context_keys: frozenset[str] = frozenset(),
        owner: str | None = None,
    ) -> frozenset[str]:
        """
        Inherit options from the consumer feature that declared this input feature.

        Called on the INPUT feature's Options (self, mutated in place); the consumer is never
        mutated. By default ALL consumer group keys flow to self: forwarding is opt-out.

        Flows:
        - forward_group: None (the unspecified default) and True copy all consumer.group keys.
          False is the explicit opt-out and copies nothing. A frozenset restricts the copy to
          the listed keys. Group-to-group only; keys absent from consumer.group are skipped
          silently.
        - forward_group_exclude: keys subtracted from whatever set forward_group produced.
          Combining it (non-empty) with forward_group=False is contradictory and raises.
        - inherit_context_keys: consumer.context keys pulled into self.context. Context-to-context
          only; keys absent from consumer.context are skipped silently.
        - consumer.propagate_context_keys: consumer-side push of context keys into self.context.
          The push is skipped when forward_group is False (only the literal False blocks it; an
          empty frozenset allowlist does not).

        DefaultOptionKeys.in_features is never inherited through any flow.

        Every key actually forwarded (including keys self already held with an equal value) is
        unioned into self.inherited_group_keys, so provenance accumulates across consumers.

        Forwarded values are isolated by a container-spine copy as they are stored: the mutable
        container spine (dict/list/set/tuple) is copied recursively so nested mutation on the child
        never leaks back to the consumer or to a sibling input feature, while every non-container
        leaf (validators, models, handles, unpicklable values) is shared by reference to preserve
        the identity the framework relies on for dedup, hashing, and conflict detection. Custom
        container types are shared by reference. Both string-declared and object input features run
        this same merge.

        The optional ``owner`` (the input feature's name) only enriches the forwarding-conflict
        error message; it changes no behavior otherwise.

        Returns the group keys forwarded this call; the self-merge no-op returns frozenset().

        Raises:
            ValueError: If a forwarded group key already exists on self with a different value
                        (rich message naming the key, both values, and the opt-out remedies,
                        including ``owner`` when given),
                        or an inherited key exists in the opposite category (group/context
                        cross-conflict),
                        or forward_group=False is combined with a non-empty forward_group_exclude.
        """
        if consumer is self:
            # A child sharing the consumer's own Options has nothing to forward: no-op. Return
            # frozenset() without touching last_forwarded_group_keys, so a shared instance is never
            # clobbered. Bundled code never hits this (string children get a fresh Options).
            return frozenset()

        memo: dict[int, Any] = {}

        excluded = NON_FORWARDED_KEYS
        validate_forwarding_directives(forward_group, forward_group_exclude)

        if forward_group is False:
            group_keys: frozenset[str] = frozenset()
        elif forward_group is None or forward_group is True:
            group_keys = frozenset(consumer.group.keys())
        else:
            group_keys = forward_group

        group_keys = group_keys - forward_group_exclude

        # Stage all mutations into locals and run every conflict check first; self is left
        # untouched until the commit block, so any raise leaves self completely unchanged.
        new_group = dict(self.group)
        new_context = dict(self.context)

        inherited: set[str] = set()
        for key in sorted(group_keys):
            if key in excluded or key not in consumer.group:
                continue
            if key in new_context and new_context[key] != consumer.group[key]:
                owner_clause = f" on input feature '{owner}'" if owner is not None else " on the input feature"
                raise ValueError(
                    f"Option key '{key}' forwarded from the consumer as a group option conflicts with the "
                    f"same key held in the child's context{owner_clause}: consumer='{consumer.group[key]}', "
                    f"child context='{new_context[key]}'. Keep the key off the child with "
                    f"forward_group_exclude={{'{key}'}}, an allowlist, or forward_group=False."
                )
            if key in new_group and new_group[key] != consumer.group[key]:
                owner_clause = f" on input feature '{owner}'" if owner is not None else " on the input feature"
                raise ValueError(
                    f"Option key '{key}' forwarded from the consumer conflicts with the value already set"
                    f"{owner_clause}: consumer='{consumer.group[key]}', child='{new_group[key]}'. "
                    f"Keep the key off the child with forward_group_exclude={{'{key}'}}, an allowlist, "
                    "or forward_group=False."
                )
            value = _isolate_forwarded_value(consumer.group[key], memo)
            OptionsValidator.validate_can_add_to_group(key, value, new_group, new_context)
            new_group[key] = value
            inherited.add(key)

        inherited_context: set[str] = set()
        for key in inherit_context_keys:
            if key in excluded or key not in consumer.context:
                continue
            value = _isolate_forwarded_value(consumer.context[key], memo)
            OptionsValidator.validate_can_add_to_context(key, value, new_group, new_context)
            new_context[key] = value
            inherited_context.add(key)

        if consumer.propagate_context_keys and forward_group is not False:
            propagating = {
                k: v for k, v in consumer.context.items() if k in consumer.propagate_context_keys and k not in excluded
            }

            OptionsValidator.validate_no_context_group_conflicts(set(propagating.keys()), set(new_group.keys()))

            for key, value in propagating.items():
                if key in new_context and new_context[key] != value:
                    raise ValueError(f"Context key '{key}' conflict: consumer='{value}', child='{new_context[key]}'")

            new_context.update({key: _isolate_forwarded_value(value, memo) for key, value in propagating.items()})
            inherited_context.update(propagating.keys())

        self.group.clear()
        self.group.update(new_group)
        self.context.clear()
        self.context.update(new_context)
        self.inherited_group_keys = self.inherited_group_keys | frozenset(inherited)
        self.last_forwarded_group_keys = frozenset(inherited)
        self.inherited_context_keys = self.inherited_context_keys | frozenset(inherited_context)
        return frozenset(inherited)
