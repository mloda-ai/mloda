"""Tests for RunContext: the frozen dataclass bundling per-run run_id/carrier/child_bootstrap."""

import dataclasses
import pickle  # nosec B403

import pytest

from mloda.core.abstract_plugins.run_context import RunContext


def _module_level_bootstrap() -> None:
    pass


class TestRunContextDefaults:
    def test_all_three_fields_default_to_none(self) -> None:
        ctx = RunContext()

        assert ctx.run_id is None
        assert ctx.carrier is None
        assert ctx.child_bootstrap is None


class TestRunContextFrozen:
    def test_assigning_a_field_raises_frozen_instance_error(self) -> None:
        ctx = RunContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.run_id = "some-run-id"  # type: ignore[misc]


class TestRunContextCarrierCopiedOnIngest:
    def test_carrier_is_equal_but_not_the_same_object(self) -> None:
        given = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}

        ctx = RunContext(carrier=given)

        assert ctx.carrier == given
        assert ctx.carrier is not given

    def test_mutating_the_stored_carrier_does_not_leak_into_the_given_dict(self) -> None:
        given = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}

        ctx = RunContext(carrier=given)
        assert ctx.carrier is not None
        ctx.carrier["mutated"] = "yes"

        assert "mutated" not in given

    def test_carrier_none_stays_none(self) -> None:
        ctx = RunContext(carrier=None)

        assert ctx.carrier is None


class TestRunContextPickleRoundTrip:
    def test_pickle_round_trip_preserves_all_three_fields(self) -> None:
        ctx = RunContext(
            run_id="01909a3b-1234-7abc-8def-0123456789ab",
            carrier={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            child_bootstrap=_module_level_bootstrap,
        )

        restored = pickle.loads(pickle.dumps(ctx))  # nosec B301

        assert restored.run_id == ctx.run_id
        assert restored.carrier == ctx.carrier
        assert restored.child_bootstrap is _module_level_bootstrap


class TestRunContextHash:
    def test_hash_of_context_with_a_carrier_returns_an_int(self) -> None:
        ctx = RunContext(run_id="r", carrier={"k": "v"})

        assert isinstance(hash(ctx), int)

    def test_two_equal_contexts_hash_equal(self) -> None:
        ctx_a = RunContext(run_id="r", carrier={"k": "v"})
        ctx_b = RunContext(run_id="r", carrier={"k": "v"})

        assert ctx_a == ctx_b
        assert hash(ctx_a) == hash(ctx_b)

    def test_equality_still_compares_the_carrier(self) -> None:
        assert RunContext(carrier={"a": "1"}) != RunContext(carrier={"a": "2"})


class TestRunContextReplace:
    def test_replace_carrier_keeps_run_id_and_child_bootstrap(self) -> None:
        ctx = RunContext(run_id="some-run-id", carrier={"k": "v"}, child_bootstrap=_module_level_bootstrap)

        replaced = dataclasses.replace(ctx, carrier={"other": "carrier"})

        assert replaced.run_id == "some-run-id"
        assert replaced.child_bootstrap is _module_level_bootstrap
        assert replaced.carrier == {"other": "carrier"}

    def test_replace_without_changes_copies_the_carrier(self) -> None:
        ctx = RunContext(carrier={"k": "v"})

        replaced = dataclasses.replace(ctx)

        assert replaced.carrier == ctx.carrier
        assert replaced.carrier is not ctx.carrier
