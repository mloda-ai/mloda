"""Tests for VerifiedContext/set_verified_context/current_verified_context in isolation."""

import dataclasses

import pytest

from mloda.core.abstract_plugins.verified_context import (
    VerifiedContext,
    current_verified_context,
    set_verified_context,
)


class TestVerifiedContextDefaults:
    def test_all_three_fields_default_to_none(self) -> None:
        ctx = VerifiedContext()

        assert ctx.tenant_id is None
        assert ctx.project_id is None
        assert ctx.principal is None


class TestVerifiedContextFrozen:
    def test_assigning_a_field_raises_frozen_instance_error(self) -> None:
        ctx = VerifiedContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.tenant_id = "acme"  # type: ignore[misc]


class TestCurrentVerifiedContextOutsideAnyScope:
    def test_returns_none_when_no_scope_active(self) -> None:
        assert current_verified_context() is None


class TestSetVerifiedContextScope:
    def test_sets_values_for_the_scope_of_the_with_block(self) -> None:
        with set_verified_context(tenant_id="acme", project_id="proj1", principal="hash123"):
            ctx = current_verified_context()
            assert ctx == VerifiedContext(tenant_id="acme", project_id="proj1", principal="hash123")

    def test_restores_previous_value_after_the_with_block_exits(self) -> None:
        with set_verified_context(tenant_id="acme"):
            pass

        assert current_verified_context() is None

    def test_all_kwargs_optional_and_default_to_none(self) -> None:
        with set_verified_context():
            assert current_verified_context() == VerifiedContext()


class TestSetVerifiedContextNesting:
    def test_nested_scope_restores_outer_value_not_none(self) -> None:
        with set_verified_context(tenant_id="outer-tenant", project_id="outer-proj", principal="outer-hash"):
            outer = current_verified_context()
            with set_verified_context(tenant_id="inner-tenant", project_id="inner-proj", principal="inner-hash"):
                inner = current_verified_context()
                assert inner == VerifiedContext(
                    tenant_id="inner-tenant", project_id="inner-proj", principal="inner-hash"
                )

            assert current_verified_context() == outer
            assert current_verified_context() == VerifiedContext(
                tenant_id="outer-tenant", project_id="outer-proj", principal="outer-hash"
            )

        assert current_verified_context() is None


class TestSetVerifiedContextRestoresOnException:
    def test_restores_previous_value_even_if_the_with_block_raises(self) -> None:
        with set_verified_context(tenant_id="outer-tenant"):
            with pytest.raises(ValueError, match="boom"):
                with set_verified_context(tenant_id="inner-tenant"):
                    raise ValueError("boom")

            assert current_verified_context() == VerifiedContext(tenant_id="outer-tenant")

        assert current_verified_context() is None
