"""unify_options fills a filter feature from its host, importing every key the filter feature omits.

A Feature value, nested containers included, arrives as a value-equal copy and every container spine
is rebuilt; a non-container leaf that is no Feature stays shared by reference, and a key the filter
feature declares itself is never rewritten.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys
from mloda.user import GlobalFilter

# The ufd_ prefix keeps these names unique repo-wide.
UFD_LEAF = "ufd_leaf"
UFD_GROUP_KEY = "ufd_group_key"
UFD_CONTEXT_KEY = "ufd_context_key"
UFD_DECLARED_KEY = "ufd_declared_key"
UFD_LIST_KEY = "ufd_list_key"
UFD_SET_KEY = "ufd_set_key"
UFD_FROZENSET_KEY = "ufd_frozenset_key"
UFD_DICT_KEY = "ufd_dict_key"
UFD_INNER_KEY = "ufd_inner_key"
UFD_HOST_VALUE = "ufd_host_value"
UFD_FILTER_VALUE = "ufd_filter_value"


class UfdHandle:
    """A non-container leaf standing for the validators, models and handles Options shares by reference."""


def _unify(host: Options, declared: Options | None = None) -> Options:
    """Unify a host's options into a filter feature's own options, as identify_matched_filters does."""
    return GlobalFilter().unify_options(host, Options() if declared is None else declared)


def test_the_hosts_input_feature_declaration_reaches_the_filter_feature() -> None:
    """in_features is a required PROPERTY_MAPPING key, so withholding it detaches config-created hosts."""
    host = Options(context={DefaultOptionKeys.in_features.value: Feature(UFD_LEAF)})

    unified = _unify(host)

    assert DefaultOptionKeys.in_features in unified, f"the host's input-feature key must arrive: {unified!s}"


def test_an_imported_input_feature_is_a_copy_of_the_hosts_feature() -> None:
    """Value-equal keeps every match working; a distinct object keeps the host's re-stamps out of the copy."""
    nested = Feature(UFD_LEAF)
    host = Options(context={DefaultOptionKeys.in_features.value: nested})

    imported = _unify(host).get(DefaultOptionKeys.in_features)

    assert imported == nested, f"the imported value must stay value-equal to the host's Feature: {imported!r}"
    assert imported is not nested, "the imported Feature must be a copy, not the host's own object"


def test_a_feature_inside_a_list_is_imported_as_a_copy() -> None:
    """A container is no hiding place: an aliased Feature strands the stored filter from any depth."""
    nested = Feature(UFD_LEAF)
    host = Options(group={UFD_LIST_KEY: [nested]})

    imported = _unify(host).get(UFD_LIST_KEY)

    assert imported == [nested], f"the imported list must stay value-equal to the host's: {imported!r}"
    assert all(element is not nested for element in imported), (
        "a Feature inside a list must not reach the filter feature as the host's own object"
    )


def test_a_feature_inside_a_set_is_imported_as_a_copy() -> None:
    """The same for a set: the copy is value-equal, so the set keeps its single member."""
    nested = Feature(UFD_LEAF)
    host = Options(group={UFD_SET_KEY: {nested}})

    imported = _unify(host).get(UFD_SET_KEY)

    assert imported == {nested}, f"the imported set must stay value-equal to the host's: {imported!r}"
    assert all(element is not nested for element in imported), (
        "a Feature inside a set must not reach the filter feature as the host's own object"
    )


def test_a_feature_inside_a_frozenset_is_imported_as_a_copy() -> None:
    """frozenset is the canonical in_features form, so it must be no hiding place under any key either."""
    nested = Feature(UFD_LEAF)
    host = Options(group={UFD_FROZENSET_KEY: frozenset({nested})})

    imported = _unify(host).get(UFD_FROZENSET_KEY)

    assert imported == frozenset({nested}), f"the imported frozenset must stay value-equal to the host's: {imported!r}"
    assert isinstance(imported, frozenset), f"an imported frozenset must stay hashable: {type(imported).__name__}"
    assert all(element is not nested for element in imported), (
        "a Feature inside a frozenset must not reach the filter feature as the host's own object"
    )


def test_a_feature_free_container_is_rebuilt_around_its_shared_leaves() -> None:
    """Every container spine is rebuilt, Feature or not; only the non-container leaves keep their identity."""
    handle = UfdHandle()
    payload: dict[str, Any] = {UFD_INNER_KEY: handle}
    host = Options(group={UFD_GROUP_KEY: UFD_HOST_VALUE, UFD_DICT_KEY: payload})

    unified = _unify(host)
    imported = unified.get(UFD_DICT_KEY)

    assert unified.get(UFD_GROUP_KEY) is UFD_HOST_VALUE, "a plain value must reach the filter feature unchanged"
    assert imported == payload, f"the imported container must stay value-equal to the host's: {imported!r}"
    assert imported is not payload, "a container spine must be rebuilt, so a nested mutation cannot leak back"
    assert imported[UFD_INNER_KEY] is handle, "a non-container leaf that is no Feature must stay shared by reference"


def test_a_key_the_filter_feature_declares_is_never_rewritten() -> None:
    """Guard: the filter feature's own value still wins over the host's, copy or not."""
    unified = _unify(
        Options(group={UFD_DECLARED_KEY: Feature(UFD_LEAF)}), Options(group={UFD_DECLARED_KEY: UFD_FILTER_VALUE})
    )

    assert unified.get(UFD_DECLARED_KEY) == UFD_FILTER_VALUE, f"a declared value must survive: {unified!s}"


def test_every_imported_key_keeps_its_own_category() -> None:
    """Guard: copying a value must not move it between group and context."""
    host = Options(group={UFD_GROUP_KEY: UFD_HOST_VALUE}, context={UFD_CONTEXT_KEY: Feature(UFD_LEAF)})

    unified = _unify(host)

    assert unified.group.get(UFD_GROUP_KEY) == UFD_HOST_VALUE, f"a host group key must flow: {unified!s}"
    assert UFD_CONTEXT_KEY in unified.context, f"a host context key must flow: {unified!s}"
    assert UFD_CONTEXT_KEY not in unified.group, f"a context key must not leak into group: {unified!s}"
