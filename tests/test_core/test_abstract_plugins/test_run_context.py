"""Tests for RunContext: the frozen dataclass bundling per-run run_id/carrier/child_bootstrap."""

import dataclasses
import json
import pickle  # nosec B403
from datetime import datetime, timezone

import pytest
from mloda.core.abstract_plugins.plan_context import PlanContext

import mloda.provider as provider
import mloda.steward as steward
import mloda.user as user
from mloda.core.abstract_plugins.run_context import RunContext


def _module_level_bootstrap() -> None:
    pass


class TestRunContextDefaults:
    def test_all_three_fields_default_to_none(self) -> None:
        ctx = RunContext()

        assert ctx.run_id is None
        assert ctx.carrier is None
        assert ctx.child_bootstrap is None

    def test_tenant_id_project_id_principal_default_to_none(self) -> None:
        ctx = RunContext()

        assert ctx.tenant_id is None
        assert ctx.project_id is None
        assert ctx.principal is None

    def test_tenant_id_project_id_principal_can_be_set_via_constructor(self) -> None:
        ctx = RunContext(tenant_id="acme", project_id="proj1", principal="hash123")

        assert ctx.tenant_id == "acme"
        assert ctx.project_id == "proj1"
        assert ctx.principal == "hash123"


class TestRunContextPlanIdAndStartedAt:
    def test_plan_id_and_started_at_default_to_none(self) -> None:
        ctx = RunContext()

        assert ctx.plan_id is None
        assert ctx.started_at is None

    def test_plan_id_and_started_at_round_trip_through_constructor_and_pickle(self) -> None:
        started = datetime.now(timezone.utc)
        ctx = RunContext(run_id="r", plan_id="p", started_at=started)

        restored = pickle.loads(pickle.dumps(ctx))  # nosec B301

        assert (restored.plan_id, restored.started_at) == ("p", started)

    def test_plan_id_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            RunContext(plan_id="p").plan_id = "forged"  # type: ignore[misc]


def _plan_context(**overrides: object) -> PlanContext:
    fields: dict[str, object] = {
        "plan_id": "plan-1",
        "tenant_id": "acme",
        "project_id": "proj1",
        "principal": "hash123",
        "created_at": datetime.now(timezone.utc),
    }
    fields.update(overrides)
    return PlanContext(**fields)  # type: ignore[arg-type]


class TestPlanContext:
    def test_exposes_the_five_fields(self) -> None:
        created = datetime.now(timezone.utc)

        ctx = _plan_context(created_at=created)

        assert (ctx.plan_id, ctx.tenant_id, ctx.project_id, ctx.principal, ctx.created_at) == (
            "plan-1",
            "acme",
            "proj1",
            "hash123",
            created,
        )

    @pytest.mark.parametrize("field", ["plan_id", "tenant_id", "project_id", "principal", "created_at"])
    def test_is_frozen(self, field: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(_plan_context(), field, "forged")

    def test_pickle_round_trip(self) -> None:
        ctx = _plan_context()

        assert pickle.loads(pickle.dumps(ctx)) == ctx  # nosec B301


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

    def test_mutating_the_stored_carrier_raises_and_does_not_leak_into_the_given_dict(self) -> None:
        given = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}

        ctx = RunContext(carrier=given)
        assert ctx.carrier is not None
        with pytest.raises(TypeError):
            ctx.carrier["mutated"] = "yes"

        assert "mutated" not in given
        assert "mutated" not in ctx.carrier

    def test_carrier_none_stays_none(self) -> None:
        ctx = RunContext(carrier=None)

        assert ctx.carrier is None


class TestRunContextCarrierReadOnly:
    @staticmethod
    def _assert_read_only(carrier: object) -> None:
        assert isinstance(carrier, dict)
        with pytest.raises(TypeError):
            carrier["x"] = "y"
        with pytest.raises(TypeError):
            carrier.pop("k")
        with pytest.raises(TypeError):
            carrier.clear()

    @pytest.mark.parametrize("variant", ["replace_unchanged", "replace_new_carrier", "pickle"])
    def test_read_only_after_replace_or_pickle(self, variant: str) -> None:
        if variant == "replace_unchanged":
            replaced = dataclasses.replace(RunContext(carrier={"k": "v"}))
            self._assert_read_only(replaced.carrier)
            assert replaced.carrier == {"k": "v"}
        elif variant == "replace_new_carrier":
            replaced = dataclasses.replace(RunContext(), carrier={"k": "v"})
            self._assert_read_only(replaced.carrier)
        else:
            ctx = RunContext(run_id="r", carrier={"k": "v"})
            restored = pickle.loads(pickle.dumps(ctx))  # nosec B301
            self._assert_read_only(restored.carrier)
            assert restored.carrier == {"k": "v"}
            assert restored == ctx

    @pytest.mark.parametrize("variant", ["constructed", "replace_unchanged", "replace_new", "pickle"])
    def test_plugin_versions_is_a_read_only_dict(self, variant: str) -> None:
        if variant == "constructed":
            versions = RunContext(plugin_versions={"m": "1"}).plugin_versions
        elif variant == "replace_unchanged":
            versions = dataclasses.replace(RunContext(plugin_versions={"m": "1"})).plugin_versions
        elif variant == "replace_new":
            versions = dataclasses.replace(RunContext(), plugin_versions={"m": "1"}).plugin_versions
        else:
            versions = pickle.loads(pickle.dumps(RunContext(plugin_versions={"m": "1"}))).plugin_versions  # nosec B301
        self._assert_read_only(versions)
        assert versions == {"m": "1"}

    def test_stays_a_dict_and_serializes_to_json(self) -> None:
        ctx = RunContext(carrier={"k": "v"})

        assert isinstance(ctx.carrier, dict)
        assert ctx.carrier == {"k": "v"}
        assert json.dumps(ctx.carrier) == '{"k": "v"}'


class TestRunContextPluginVersionsCopiedOnIngest:
    def test_plugin_versions_is_equal_but_not_the_same_object(self) -> None:
        given = {"some.plugin.module": "1.2.3"}

        ctx = RunContext(plugin_versions=given)

        assert ctx.plugin_versions == given
        assert ctx.plugin_versions is not given


class TestRunContextPickleRoundTrip:
    def test_pickle_round_trip_preserves_run_id_carrier_child_bootstrap_and_plugin_versions(self) -> None:
        ctx = RunContext(
            run_id="01909a3b-1234-7abc-8def-0123456789ab",
            carrier={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
            child_bootstrap=_module_level_bootstrap,
            plugin_versions={"m": "1.0"},
        )

        restored = pickle.loads(pickle.dumps(ctx))  # nosec B301

        assert restored.run_id == ctx.run_id
        assert restored.carrier == ctx.carrier
        assert restored.child_bootstrap is _module_level_bootstrap
        assert restored.plugin_versions == ctx.plugin_versions

    def test_pickle_round_trip_preserves_tenant_id_project_id_and_principal(self) -> None:
        ctx = RunContext(tenant_id="acme", project_id="proj1", principal="hash123")

        restored = pickle.loads(pickle.dumps(ctx))  # nosec B301

        assert restored.tenant_id == ctx.tenant_id
        assert restored.project_id == ctx.project_id
        assert restored.principal == ctx.principal


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

    def test_two_equal_contexts_with_tenant_project_principal_hash_equal(self) -> None:
        ctx_a = RunContext(tenant_id="acme", project_id="proj1", principal="hash123")
        ctx_b = RunContext(tenant_id="acme", project_id="proj1", principal="hash123")

        assert ctx_a == ctx_b
        assert hash(ctx_a) == hash(ctx_b)

    def test_equality_compares_tenant_id_project_id_and_principal(self) -> None:
        assert RunContext(tenant_id="acme") != RunContext(tenant_id="other")
        assert RunContext(project_id="proj1") != RunContext(project_id="proj2")
        assert RunContext(principal="hash1") != RunContext(principal="hash2")

    def test_plugin_versions_is_excluded_from_the_hash(self) -> None:
        assert hash(RunContext(plugin_versions={"m": "1"})) == hash(RunContext())


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

    def test_replace_tenant_id_project_id_principal_keeps_other_fields(self) -> None:
        ctx = RunContext(run_id="some-run-id", carrier={"k": "v"}, child_bootstrap=_module_level_bootstrap)

        replaced = dataclasses.replace(ctx, tenant_id="acme", project_id="proj1", principal="hash123")

        assert replaced.run_id == "some-run-id"
        assert replaced.carrier == {"k": "v"}
        assert replaced.child_bootstrap is _module_level_bootstrap
        assert replaced.tenant_id == "acme"
        assert replaced.project_id == "proj1"
        assert replaced.principal == "hash123"


class TestRunContextGracefulShutdownTimeout:
    def test_default_is_two_point_zero_seconds(self) -> None:
        ctx = RunContext()

        assert ctx.graceful_shutdown_timeout == 2.0

    def test_custom_value_round_trips_through_the_constructor(self) -> None:
        ctx = RunContext(graceful_shutdown_timeout=7.5)

        assert ctx.graceful_shutdown_timeout == 7.5


class TestRunContextIsInternal:
    def test_docstring_names_the_steward_export(self) -> None:
        assert RunContext.__doc__ is not None
        assert "mloda.steward" in RunContext.__doc__

    def test_only_exported_from_the_steward_facade(self) -> None:
        assert "RunContext" not in provider.__all__
        assert not hasattr(provider, "RunContext")

        assert "RunContext" in steward.__all__
        assert steward.RunContext is RunContext

        assert "RunContext" not in user.__all__
        assert not hasattr(user, "RunContext")
