"""Tests for the private credential grant holder: lookup lifecycle, masking scope and the test reader."""

import contextvars
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.credential_grant import (
    CredentialLookup,
    StaticGrant,
    _read_granted,
    grant_scope,
)
from mloda.core.abstract_plugins.components.credential_scrub import (
    current_identity_secrets,
    current_registered_secrets,
)

_HMAC = "hmac-grant-Qv7Lk2Xz9"  # nosec B105
_KEY = "key-grant-Wn4Rt8Bm3"  # nosec B105
_HOST = "grant-host.internal.example"
_BARE = "bare-grant-Hd5Pc1Jy6"  # nosec B105


def _credential() -> dict[str, Any]:
    return {"host": _HOST, "hmac": _HMAC, "key": _KEY}


class TestStaticGrant:
    def test_open_is_one_shot_and_empties_the_grant(self) -> None:
        grant = StaticGrant({"h": _credential()})

        lookup = grant.open()

        assert lookup.credential("h") == _credential()
        second = grant.open()
        with pytest.raises(KeyError, match="h"):
            second.credential("h")


class TestCredentialLookup:
    def test_ungranted_handle_raises_naming_the_handle_and_never_a_value(self) -> None:
        lookup = CredentialLookup({"granted": _credential()})

        with pytest.raises(KeyError) as info:
            lookup.credential("missing_handle")

        assert "missing_handle" in str(info.value)
        assert _HMAC not in str(info.value)

    def test_get_after_close_refuses_naming_the_handle_without_a_value(self) -> None:
        lookup = CredentialLookup({"granted": _credential()})
        assert not lookup.closed

        lookup.close()

        assert lookup.closed
        with pytest.raises(Exception, match="granted") as info:
            lookup.credential("granted")
        assert _HMAC not in str(info.value)

    def test_close_and_leaving_the_scope_do_not_mutate_the_nested_values(self) -> None:
        value = _credential()
        lookup = CredentialLookup({"granted": value})

        lookup.close()

        assert value == _credential()

        scoped = _credential()
        with grant_scope(StaticGrant({"h": scoped})):
            pass

        assert scoped == _credential()


class TestGrantScope:
    def test_read_outside_any_scope_refuses_naming_the_handle(self) -> None:
        with pytest.raises(Exception, match="some_handle"):
            _read_granted("some_handle")

    def test_declared_value_is_readable_inside_and_refused_after(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            assert _read_granted("h") == _credential()

        with pytest.raises(RuntimeError, match="not available outside a step"):
            _read_granted("h")

    def test_lookup_is_closed_after_exit_even_through_a_context_captured_inside(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            captured = contextvars.copy_context()

        with pytest.raises(RuntimeError, match="no longer available") as info:
            captured.run(_read_granted, "h")
        assert _HMAC not in str(info.value)

    def test_ungranted_handle_inside_the_scope_names_the_handle(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            with pytest.raises(KeyError, match="other"):
                _read_granted("other")

    def test_secret_leaves_are_registered_inside_and_gone_after(self) -> None:
        before = current_registered_secrets()

        with grant_scope(StaticGrant({"h": _credential()})):
            inside = current_registered_secrets()
            assert {_HMAC, _KEY} <= inside

        assert current_registered_secrets() == before
        assert _HMAC not in current_registered_secrets()

    def test_only_credential_shaped_leaves_reach_the_identity_set(self) -> None:
        shaped = "pw-grant-Zx3Cv7Bn5"  # nosec B105
        value = {"host": _HOST, "password": shaped, "hmac": _HMAC}  # nosec B105
        before = current_identity_secrets()

        with grant_scope(StaticGrant({"h": value})):
            assert {shaped, _HMAC} <= current_registered_secrets()
            assert shaped in current_identity_secrets()
            assert _HMAC not in current_identity_secrets()

        assert current_identity_secrets() == before
        assert shaped not in current_identity_secrets()
        assert shaped not in current_registered_secrets()

    def test_a_safe_key_leaf_is_not_registered(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            assert _HOST not in current_registered_secrets()

    def test_a_handle_named_like_a_safe_key_does_not_make_its_leaves_safe(self) -> None:
        with grant_scope(StaticGrant({"host": {"token": _KEY}})):
            assert _KEY in current_registered_secrets()

    def test_an_exception_echoing_a_grant_only_value_is_redacted_on_exit(self) -> None:
        boom = RuntimeError(f"bad {_HMAC} and {_HOST}")

        with pytest.raises(RuntimeError) as info:
            with grant_scope(StaticGrant({"h": _credential()})):
                raise boom

        assert info.value is boom
        assert _HMAC not in str(boom)
        assert _HOST in str(boom)

    @pytest.mark.parametrize(
        "grant_value, secret",
        [
            pytest.param(_BARE, _BARE, id="bare_str"),
            pytest.param(Credential({"token": _KEY}), _KEY, id="credential_wrapped"),  # nosec B106
        ],
    )
    def test_a_secret_grant_value_is_registered_and_masked(self, grant_value: Any, secret: str) -> None:
        with pytest.raises(RuntimeError) as info:
            with grant_scope(StaticGrant({"h": grant_value})):
                assert secret in current_registered_secrets()
                raise RuntimeError(f"failed with {secret}")

        assert secret not in str(info.value)

    def test_none_grant_works_and_registers_nothing(self) -> None:
        before = current_registered_secrets()

        with grant_scope(None):
            assert current_registered_secrets() == before
            with pytest.raises(Exception, match="h"):
                _read_granted("h")

        with pytest.raises(Exception, match="h"):
            _read_granted("h")

    def test_a_second_scope_does_not_see_the_first_scopes_handle(self) -> None:
        with grant_scope(StaticGrant({"first": _credential()})):
            pass

        with grant_scope(StaticGrant({"second": {"token": _KEY}})):
            with pytest.raises(KeyError, match="first"):
                _read_granted("first")
            assert _HMAC not in current_registered_secrets()
