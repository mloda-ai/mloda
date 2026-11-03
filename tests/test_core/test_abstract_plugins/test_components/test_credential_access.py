"""Tests for read_credential, credentials_scope, the readable windows and ResolvedCredential."""

import asyncio
import contextvars
import copy
import json
import pickle  # nosec B403
import threading
from collections.abc import MutableMapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.credential_access import (
    CredentialFieldMissing,
    CredentialReadRefused,
    ResolvedCredential,
    credentials_scope,
    hidden_credentials,
    read_credential,
    readable,
    readable_credentials,
)
from mloda.core.abstract_plugins.components.credential_grant import StaticGrant, grant_scope
from datetime import UTC

_HOST = "access-host.internal.example"
_HMAC = "hmac-access-Qv7Lk2Xz9"  # nosec B105
_KEY = "key-access-Wn4Rt8Bm3"  # nosec B105


def _credential() -> dict[str, Any]:
    return {"host": _HOST, "hmac": _HMAC, "key": _KEY}


def _scope() -> Any:
    return credentials_scope({"h": _credential()})


class _Reader:
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> str:
        return str(read_credential("h")["host"])

    def method(self) -> str:
        return str(read_credential("h")["host"])


class TestResolvedCredential:
    def test_is_a_mutable_mapping_and_not_a_dict(self) -> None:
        cred = ResolvedCredential("h", _credential())

        assert isinstance(cred, MutableMapping)
        assert not isinstance(cred, dict)

    def test_item_access_get_iteration_and_unpacking(self) -> None:
        cred = ResolvedCredential("h", _credential())

        assert cred["host"] == _HOST
        assert cred.get("hmac") == _HMAC
        assert cred.get("absent") is None
        assert set(cred) == {"host", "hmac", "key"}
        assert len(cred) == 3
        assert {**cred} == _credential()

    def test_set_and_delete_item(self) -> None:
        cred = ResolvedCredential("h", _credential())

        cred["extra"] = "x"
        del cred["key"]

        assert cred["extra"] == "x"
        assert "key" not in cred

    def test_handle_is_set_and_static_metadata_is_none(self) -> None:
        cred = ResolvedCredential("h", _credential())

        assert cred.handle == "h"
        assert cred.version is None
        assert cred.expires_at is None
        assert cred.store is None

    def test_has_no_instance_dict_so_fields_are_reachable_only_by_mapping_access(self) -> None:
        cred = ResolvedCredential("h", _credential())

        with pytest.raises(TypeError):
            vars(cred)
        assert cred["host"] == _HOST

    def test_metadata_can_be_passed_at_construction(self) -> None:
        from datetime import datetime

        expires = datetime(2030, 1, 1, tzinfo=UTC)

        cred = ResolvedCredential("h", _credential(), version="v1", expires_at=expires, store="vault")

        assert cred.version == "v1"
        assert cred.expires_at == expires
        assert cred.store == "vault"
        assert cred["host"] == _HOST

    def test_missing_field_raises_the_typed_key_error_naming_handle_and_field(self) -> None:
        cred = ResolvedCredential("h", _credential())

        with pytest.raises(CredentialFieldMissing) as info:
            cred["nope"]

        assert isinstance(info.value, KeyError)
        text = str(info.value)
        assert "h" in text
        assert "nope" in text
        assert _HMAC not in text
        assert _HOST not in text

    def test_repr_and_str_redact_values(self) -> None:
        cred = ResolvedCredential("h", _credential())

        for text in (repr(cred), str(cred)):
            assert "h" in text
            assert _HMAC not in text
            assert _KEY not in text
            assert _HOST not in text

    @pytest.mark.parametrize("protocol", [0, 2, pickle.HIGHEST_PROTOCOL])
    def test_pickle_raises_naming_the_handle_never_the_value(self, protocol: int) -> None:
        cred = ResolvedCredential("the-handle", _credential())

        with pytest.raises(TypeError) as info:
            pickle.dumps(cred, protocol=protocol)

        assert "the-handle" in str(info.value)
        assert _HMAC not in str(info.value)

    @pytest.mark.parametrize("clone", [copy.deepcopy, copy.copy])
    def test_copy_and_deepcopy_raise_naming_the_handle(self, clone: Any) -> None:
        cred = ResolvedCredential("the-handle", _credential())

        with pytest.raises(TypeError) as info:
            clone(cred)

        assert "the-handle" in str(info.value)
        assert _HMAC not in str(info.value)

    def test_json_dumps_raises_a_type_error_without_the_value(self) -> None:
        cred = ResolvedCredential("h", _credential())

        with pytest.raises(TypeError) as info:
            json.dumps(cred)

        assert _HMAC not in str(info.value)

    def test_unhashable(self) -> None:
        with pytest.raises(TypeError):
            hash(ResolvedCredential("h", _credential()))

    def test_construction_copies_nested_containers(self) -> None:
        source: dict[str, Any] = {"nested": {"items": [1, 2]}}

        cred = ResolvedCredential("h", source)
        source["nested"]["items"].append(3)

        assert cred["nested"]["items"] == [1, 2]

    def test_dict_of_a_credential_copies_the_values(self) -> None:
        copied = dict(ResolvedCredential("h", _credential()))

        assert copied == _credential()


class TestReadCredential:
    def test_outside_a_scope_raises_naming_the_handle(self) -> None:
        with pytest.raises(CredentialReadRefused, match="some_handle"):
            read_credential("some_handle")

    def test_the_refusal_is_not_a_lookup_error(self) -> None:
        assert not issubclass(CredentialReadRefused, LookupError)

    def test_reads_a_declared_handle_as_a_resolved_credential(self) -> None:
        with _scope():
            cred = read_credential("h")

        assert isinstance(cred, ResolvedCredential)
        assert cred.handle == "h"
        assert cred["host"] == _HOST
        assert cred["hmac"] == _HMAC

    def test_each_read_is_a_fresh_copy_with_nested_containers_copied(self) -> None:
        with credentials_scope({"h": {"nested": {"items": [1]}}}):
            first = read_credential("h")
            first["nested"]["items"].append(2)
            first["added"] = True
            second = read_credential("h")

        assert second["nested"]["items"] == [1]
        assert "added" not in second

    def test_undeclared_handle_is_refused_naming_the_declared_ones_not_values(self) -> None:
        with credentials_scope({"b": _credential(), "a": _credential()}):
            with pytest.raises(CredentialReadRefused) as info:
                read_credential("other")

        text = str(info.value)
        assert "other" in text
        assert text.index("'a'") < text.index("'b'")
        assert _HMAC not in text

    def test_a_read_through_a_copied_context_after_the_scope_exited_is_refused(self) -> None:
        with _scope():
            captured = contextvars.copy_context()

        with pytest.raises(CredentialReadRefused, match="h"):
            captured.run(read_credential, "h")

    def test_the_same_scope_instance_entered_twice_in_sequence_reads_both_times(self) -> None:
        scope = _scope()

        with scope:
            assert read_credential("h")["host"] == _HOST
        with scope:
            assert read_credential("h")["host"] == _HOST
        with pytest.raises(CredentialReadRefused):
            read_credential("h")

    def test_a_read_after_the_scope_exited_is_refused(self) -> None:
        with _scope():
            pass

        with pytest.raises(CredentialReadRefused):
            read_credential("h")

    def test_a_readable_scope_opened_with_no_holder_refuses_every_read(self) -> None:
        with readable_credentials():
            with pytest.raises(CredentialReadRefused, match="h"):
                read_credential("h")

    def test_a_grant_without_a_readable_scope_refuses_reads(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            with pytest.raises(CredentialReadRefused, match="h"):
                read_credential("h")

    def test_hidden_credentials_refuses_inside_and_restores_after(self) -> None:
        with _scope():
            with hidden_credentials():
                with pytest.raises(CredentialReadRefused):
                    read_credential("h")
            assert read_credential("h")["host"] == _HOST

    def test_readable_credentials_opens_the_window_over_a_held_grant(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            with readable_credentials():
                assert read_credential("h")["host"] == _HOST
            with pytest.raises(CredentialReadRefused):
                read_credential("h")

    def test_readable_wrapper_opens_the_window_for_the_call_only(self) -> None:
        with grant_scope(StaticGrant({"h": _credential()})):
            wrapped = readable(_Reader.calculate_feature)

            assert wrapped(None, None) == _HOST
            with pytest.raises(CredentialReadRefused):
                read_credential("h")

    def test_readable_wrapper_keeps_the_bound_instance_and_name(self) -> None:
        instance = _Reader()

        wrapped = readable(instance.method)

        assert wrapped.__self__ is instance  # type: ignore[attr-defined]
        assert wrapped.__name__ == "method"

    def test_readable_wrapper_of_a_classmethod_keeps_its_class(self) -> None:
        wrapped = readable(_Reader.calculate_feature)

        assert wrapped.__self__ is _Reader  # type: ignore[attr-defined]
        assert wrapped.__name__ == "calculate_feature"

    def test_credentials_scope_lets_a_unit_test_call_calculate_feature_directly(self) -> None:
        with _scope():
            assert _Reader.calculate_feature(None, None) == _HOST

    def test_an_error_echoing_a_secret_inside_the_scope_is_masked(self) -> None:
        with pytest.raises(ValueError) as info:
            with _scope():
                raise ValueError(f"failed hmac={read_credential('h')['hmac']}")

        assert _HMAC not in str(info.value)
        assert _HMAC not in repr(info.value.args)

    def test_a_plain_thread_started_inside_the_scope_has_no_scope(self) -> None:
        outcome: list[str] = []

        def target() -> None:
            try:
                read_credential("h")
                outcome.append("read")
            except CredentialReadRefused:
                outcome.append("refused")

        with _scope():
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=5)

        assert outcome == ["refused"]

    def test_a_thread_running_a_copied_context_reads(self) -> None:
        outcome: list[str] = []

        with _scope():
            ctx = contextvars.copy_context()
            thread = threading.Thread(target=lambda: outcome.append(ctx.run(read_credential, "h")["host"]))
            thread.start()
            thread.join(timeout=5)

        assert outcome == [_HOST]

    def test_an_awaited_asyncio_task_inherits_the_scope(self) -> None:
        async def read() -> str:
            return str(read_credential("h")["host"])

        async def main() -> str:
            return await asyncio.create_task(read())

        with _scope():
            assert asyncio.run(main()) == _HOST
