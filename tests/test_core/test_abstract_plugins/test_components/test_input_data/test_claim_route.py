"""ClaimRoute, NamePolicy and SourceMatch value objects, and their public exports."""

from typing import Any

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import (
    ChainCheck,
    ChainCheckNeeded,
    ClaimRoute,
    NamePolicy,
    SourceMatch,
    chain_check_scope,
    computed_elsewhere,
    remembered_probe,
)


def test_no_check_installed_means_not_computed_elsewhere() -> None:
    assert computed_elsewhere("name") is False


def test_an_unknown_answer_raises_the_signal_carrying_the_check() -> None:
    check = ChainCheck("name")

    with chain_check_scope(check):
        with pytest.raises(ChainCheckNeeded) as caught:
            computed_elsewhere("name")

    assert caught.value.check is check
    assert not isinstance(caught.value, Exception)


@pytest.mark.parametrize("answer", [True, False])
def test_a_known_answer_is_returned(answer: bool) -> None:
    with chain_check_scope(ChainCheck("name", answer=answer)):
        assert computed_elsewhere("name") is answer


def test_another_name_is_false_and_never_signals() -> None:
    with chain_check_scope(ChainCheck("name")):
        assert computed_elsewhere("other") is False


def test_the_check_scope_resets_on_exit() -> None:
    with chain_check_scope(ChainCheck("name", answer=True)):
        pass

    assert computed_elsewhere("name") is False


def test_remembered_probe_runs_once_per_group_under_a_check() -> None:
    calls: list[str] = []

    def probe_for(label: str) -> Any:
        def probe() -> str:
            calls.append(label)
            return label

        return probe

    class GroupA:
        pass

    class GroupB:
        pass

    with chain_check_scope(ChainCheck("name")):
        assert remembered_probe("name", GroupA, probe_for("a")) == "a"
        assert remembered_probe("name", GroupA, probe_for("a2")) == "a"
        assert remembered_probe("name", GroupB, probe_for("b")) == "b"

    assert calls == ["a", "b"]


def test_remembered_probe_runs_every_time_without_a_check() -> None:
    calls: list[int] = []

    def probe() -> int:
        calls.append(1)
        return len(calls)

    class Group:
        pass

    assert remembered_probe("name", Group, probe) == 1
    assert remembered_probe("name", Group, probe) == 2
    with chain_check_scope(ChainCheck("other")):
        assert remembered_probe("name", Group, probe) == 3


def test_claim_route_is_hashable_and_compares_by_value() -> None:
    assert ClaimRoute("a", NamePolicy.OPEN, False, ("x",)) == ClaimRoute("a", NamePolicy.OPEN, False, ("x",))
    assert ClaimRoute("a", NamePolicy.OPEN, False) != ClaimRoute("a", NamePolicy.OPEN, False, ("x",))
    assert len({ClaimRoute("a", NamePolicy.OPEN, False), ClaimRoute("a", NamePolicy.OPEN, False)}) == 1


def test_source_match_equality_and_hash_use_source_only() -> None:
    one = SourceMatch(source="h:t", access={"detail": 1})
    two = SourceMatch(source="h:t", access={"detail": 2})
    assert one == two
    assert hash(one) == hash(two)
    assert one != SourceMatch(source="h:other", access={"detail": 1})
    assert len({one, two}) == 1


def test_source_match_repr_excludes_access() -> None:
    match = SourceMatch(source="h:t", access={"detail": "abc-value-xyz"})
    assert "abc-value-xyz" not in repr(match)
    assert "h:t" in repr(match)


def test_provider_exports_are_the_core_objects() -> None:
    assert provider.ClaimRoute is ClaimRoute
    assert provider.NamePolicy is NamePolicy
    assert provider.SourceMatch is SourceMatch
    assert provider.QUERY_OPTION == "query_text"
    assert provider.HANDLE_OPTION == "data_access_handle"
    for name in ("ClaimRoute", "NamePolicy", "SourceMatch", "SourceFeatureGroup", "QUERY_OPTION", "HANDLE_OPTION"):
        assert name in provider.__all__


@pytest.mark.parametrize(
    "kwargs,field_name",
    [
        ({"source_kind": ""}, "source_kind"),
        ({"source_kind": 5}, "source_kind"),
        ({"names": "open"}, "names"),
        ({"searched": "yes"}, "searched"),
        ({"searched": 1}, "searched"),
        ({"required_options": ["q"]}, "required_options"),
        ({"required_options": (1,)}, "required_options"),
    ],
)
def test_claim_route_rejects_a_badly_typed_field(kwargs: dict[str, Any], field_name: str) -> None:
    fields: dict[str, Any] = {"source_kind": "x", "names": NamePolicy.OPEN, "searched": False}
    fields.update(kwargs)
    with pytest.raises(TypeError, match=field_name):
        ClaimRoute(**fields)


def test_provider_exports_read_file_fg() -> None:
    from mloda.core.abstract_plugins.components.input_data.read_file_fg import ReadFileFG

    assert provider.ReadFileFG is ReadFileFG
    assert "ReadFileFG" in provider.__all__


def test_provider_exports_implementation() -> None:
    from mloda.core.abstract_plugins.implementation import Implementation

    assert provider.Implementation is Implementation
    assert "Implementation" in provider.__all__
