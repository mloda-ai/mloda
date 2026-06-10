"""Contract tests for the typed ``Credential`` class (cycle 1, issue #511).

``Credential`` wraps exactly one credential mapping so that
``DataAccessCollection(credentials=...)`` can no longer mistake a single
credential dict for a ``{handle: value}`` registry.
"""

import pytest

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.hashable_dict import HashableDict


class TestCredentialConstruction:
    """Constructor accepts a mapping, kwargs, or both; ``.data`` exposes the merged dict."""

    def test_positional_mapping_is_stored(self) -> None:
        cred = Credential({"sqlite": "/data/x.db"})
        assert cred.data == {"sqlite": "/data/x.db"}

    def test_kwargs_form_is_equivalent_to_mapping(self) -> None:
        cred = Credential(sqlite="/data/x.db")
        assert cred.data == {"sqlite": "/data/x.db"}

    def test_mixed_mapping_and_kwargs_merge_with_kwargs_winning(self) -> None:
        """Mapping and kwargs merge with dict() semantics: kwargs override mapping keys."""
        cred = Credential({"a": 1}, b=2)
        assert cred.data == {"a": 1, "b": 2}
        overridden = Credential({"a": 1}, a=3)
        assert overridden.data == {"a": 3}


class TestCredentialRejectsBadInput:
    """Non-mapping positional input and empty credentials are rejected loudly."""

    def test_string_positional_raises_type_error_mentioning_mapping(self) -> None:
        with pytest.raises(TypeError, match="(?i)dict|mapping"):
            Credential("oops")  # type: ignore[arg-type]

    def test_list_of_pairs_positional_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="(?i)dict|mapping"):
            Credential([("a", 1)])  # type: ignore[arg-type]

    def test_no_arguments_raises_value_error_mentioning_at_least_one_field(self) -> None:
        with pytest.raises(ValueError, match="(?i)at least one"):
            Credential()

    def test_empty_mapping_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="(?i)at least one"):
            Credential({})


class TestCredentialDataIsDefensiveCopy:
    """``.data`` hands out a copy; neither side can mutate the stored credential."""

    def test_mutating_returned_dict_does_not_affect_subsequent_access(self) -> None:
        cred = Credential(sqlite="/data/x.db")
        leaked = cred.data
        leaked["sqlite"] = "/tampered.db"
        leaked["extra"] = "injected"
        assert cred.data == {"sqlite": "/data/x.db"}

    def test_mutating_original_constructor_dict_does_not_affect_data(self) -> None:
        original = {"sqlite": "/data/x.db"}
        cred = Credential(original)
        original["sqlite"] = "/tampered.db"
        original["extra"] = "injected"
        assert cred.data == {"sqlite": "/data/x.db"}


class TestCredentialReprRedactsValues:
    """``repr()`` must never leak secret values; it shows keys with redacted values."""

    def test_repr_shows_class_and_keys_but_redacts_values(self) -> None:
        rendered = repr(Credential(sqlite="/secret/path.db"))
        assert "Credential" in rendered
        assert "sqlite" in rendered
        assert "/secret/path.db" not in rendered
        assert "***" in rendered


class TestCredentialPublicSurface:
    """``Credential`` is part of the public ``mloda.user`` API."""

    def test_importable_from_mloda_user(self) -> None:
        from mloda.user import Credential as UserCredential

        assert UserCredential is Credential

    def test_listed_in_mloda_user_all(self) -> None:
        import mloda.user

        assert "Credential" in mloda.user.__all__


class TestDataAccessCollectionUnwrapsCredential:
    """DataAccessCollection unwraps Credential at registration time.

    Downstream consumers keep seeing plain dicts, never Credential instances.
    """

    def test_single_credential_registers_one_auto_named_plain_dict(self) -> None:
        dac = DataAccessCollection(credentials=Credential(sqlite="/data/x.db"))
        resolved = dac.resolve("credentials")
        assert type(resolved) is dict
        assert resolved == {"sqlite": "/data/x.db"}
        registered = dac.handles()
        assert len(registered) == 1
        (handle, kind) = next(iter(registered.items()))
        assert kind == "credentials"
        assert handle.startswith("_auto_credentials_")

    def test_list_mixing_credential_and_plain_dict_registers_two_plain_dicts(self) -> None:
        dac = DataAccessCollection(credentials=[Credential(sqlite="/a.db"), {"pg": {"host": "h"}}])
        assert len(dac.credentials) == 2
        for value in dac.credentials.values():
            assert type(value) is dict
        assert list(dac.credentials.values()) == [{"sqlite": "/a.db"}, {"pg": {"host": "h"}}]

    def test_named_form_with_credential_value_stores_plain_dict(self) -> None:
        dac = DataAccessCollection(credentials={"prod": Credential(sqlite="/a.db")})
        assert dac.credentials["prod"] == {"sqlite": "/a.db"}
        assert type(dac.credentials["prod"]) is dict

    def test_add_credentials_single_arg_unwraps_credential(self) -> None:
        dac = DataAccessCollection()
        dac.add_credentials(Credential(sqlite="/a.db"))
        assert len(dac.credentials) == 1
        (only_handle,) = dac.credentials.keys()
        assert only_handle.startswith("_auto_credentials_")
        assert dac.credentials[only_handle] == {"sqlite": "/a.db"}
        assert type(dac.credentials[only_handle]) is dict

    def test_add_credentials_two_arg_unwraps_credential(self) -> None:
        dac = DataAccessCollection()
        dac.add_credentials("named", Credential(sqlite="/a.db"))
        assert dac.credentials == {"named": {"sqlite": "/a.db"}}
        assert type(dac.credentials["named"]) is dict

    def test_dict_positional_credential_registers_one_auto_named_plain_dict(self) -> None:
        """Dict-positional form Credential({...}) flows through the collection like the kwargs form."""
        dac = DataAccessCollection(credentials=Credential({"sqlite": "/data/x.db"}))
        resolved = dac.resolve("credentials")
        assert type(resolved) is dict
        assert resolved == {"sqlite": "/data/x.db"}
        registered = dac.handles()
        assert len(registered) == 1
        (handle, kind) = next(iter(registered.items()))
        assert kind == "credentials"
        assert handle.startswith("_auto_credentials_")


class TestNamedFormRejectsNonMappingValues:
    """Cycle 2 of issue #511: the named form ``{handle: value}`` requires mapping values.

    A non-mapping value means the caller almost certainly mis-wrapped a single
    credential, e.g. ``credentials={"sqlite": "/data/x.db"}`` where the whole dict
    was meant to BE the credential. This must fail loudly at construction time
    instead of silently never matching later in matcher selection.
    """

    def test_single_field_mis_wrap_raises_value_error_naming_handle(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"sqlite": "/data/x.db"})
        msg = str(excinfo.value)
        assert "sqlite" in msg
        assert "mapping" in msg.lower()

    def test_error_message_points_to_all_three_correct_alternatives(self) -> None:
        """The message must offer the list form, the typed form, and the named-with-mapping form."""
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"sqlite": "/data/x.db"})
        msg = str(excinfo.value)
        assert "Credential" in msg
        assert "[" in msg
        assert "list" in msg.lower()
        assert "{" in msg

    def test_multi_field_mis_wrap_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"host": "h", "port": 5432})
        msg = str(excinfo.value)
        assert ("host" in msg) or ("port" in msg)
        assert "mapping" in msg.lower()

    def test_add_credentials_two_arg_non_mapping_value_raises_value_error(self) -> None:
        dac = DataAccessCollection()
        with pytest.raises(ValueError) as excinfo:
            dac.add_credentials("h", "/data/x.db")
        msg = str(excinfo.value)
        assert "mapping" in msg.lower()
        assert "Credential" in msg


class TestNamedFormAcceptsMappingValues:
    """Legitimate named-form mapping values (plain dict, Credential) keep working."""

    def test_plain_dict_value_does_not_raise_and_is_stored_as_is(self) -> None:
        dac = DataAccessCollection(credentials={"pg-prod": {"host": "h"}})
        assert dac.credentials == {"pg-prod": {"host": "h"}}
        assert type(dac.credentials["pg-prod"]) is dict

    def test_add_credentials_two_arg_mapping_value_still_works(self) -> None:
        dac = DataAccessCollection()
        dac.add_credentials("h", {"sqlite": "/x.db"})
        assert dac.credentials == {"h": {"sqlite": "/x.db"}}


class TestCredentialsPathRejectsHashableDict:
    """Issue #519: HashableDict is rejected as a credential VALUE on every entry path.

    Before 0.7.0 ``DataAccessCollection`` force-wrapped credentials in HashableDict,
    so readers grew isinstance branches for it. With the typed ``Credential`` class in
    place, HashableDict on the credentials path is now a migration error. HashableDict
    itself stays valid for Options group values; only the credentials path rejects it.

    The migration ValueError must name ``Credential`` and ``dict`` (the supported
    replacements), mention ``HashableDict`` (what was rejected), and never echo the
    secret value.
    """

    SECRET = "/secret/credential/path.db"  # nosec B105

    def _assert_migration_error(self, msg: str) -> None:
        lowered = msg.lower()
        assert "hashabledict" in lowered
        assert "credential" in lowered
        assert "dict" in lowered
        assert self.SECRET not in msg

    def test_named_form_rejects_hashable_dict_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"db": HashableDict({"sqlite": self.SECRET})})
        self._assert_migration_error(str(excinfo.value))

    def test_list_form_rejects_hashable_dict_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials=[HashableDict({"sqlite": self.SECRET})])
        self._assert_migration_error(str(excinfo.value))

    def test_add_credentials_single_arg_rejects_hashable_dict(self) -> None:
        dac = DataAccessCollection()
        with pytest.raises(ValueError) as excinfo:
            dac.add_credentials(HashableDict({"sqlite": self.SECRET}))
        self._assert_migration_error(str(excinfo.value))

    def test_add_credentials_two_arg_rejects_hashable_dict_value(self) -> None:
        dac = DataAccessCollection()
        with pytest.raises(ValueError) as excinfo:
            dac.add_credentials("named", HashableDict({"sqlite": self.SECRET}))
        self._assert_migration_error(str(excinfo.value))


class TestResolveAmbiguityRedactsCredentialValues:
    """Cycle 3, Finding A (security): the all-auto ambiguity error in ``resolve()``
    must not print stored credential values verbatim.

    For kind 'credentials' the candidate listing must redact values (keys stay
    visible, values replaced, e.g. with '***'). Other kinds keep showing values.
    """

    def test_two_auto_credentials_ambiguity_redacts_secret_values(self) -> None:
        dac = DataAccessCollection(
            credentials=[
                Credential(host="db1", password="hunter2"),  # nosec B106
                Credential(host="db2", password="swordfish"),  # nosec B106
            ]
        )
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("credentials")
        msg = str(excinfo.value)
        assert "hunter2" not in msg
        assert "swordfish" not in msg
        assert "host" in msg
        assert "password" in msg
        assert "data_access_handle" in msg

    def test_auto_named_files_ambiguity_still_shows_values(self) -> None:
        """Guardrail: redaction is credentials-only; file paths stay visible in the listing."""
        dac = DataAccessCollection(files=["/data/a.csv", "/data/b.csv"])
        with pytest.raises(ValueError) as excinfo:
            dac.resolve("file")
        msg = str(excinfo.value)
        assert "/data/a.csv" in msg


class TestMisWrapErrorSuggestionsAreSafeAndComplete:
    """Cycle 3, Finding B: the named-form mis-wrap ValueError must build its
    suggestions from ALL keys of the offending input, never include the
    offending values (they may be secrets), and use the dict-positional form
    ``Credential({...})`` so the suggestion stays valid Python for any key.
    """

    def test_multi_field_suggestion_lists_all_keys_and_uses_dict_positional_form(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"host": "h", "port": 5432})
        msg = str(excinfo.value)
        assert "host" in msg
        assert "port" in msg
        assert "Credential({" in msg
        # Offending values must not be echoed back ('h' alone is untestable as a
        # substring because 'host' contains it; the numeric value is unambiguous).
        assert "5432" not in msg

    def test_non_identifier_handle_never_yields_invalid_kwargs_suggestion(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"pg-prod": "dsn-string"})
        msg = str(excinfo.value)
        assert "pg-prod" in msg
        assert "Credential({" in msg
        assert "Credential(pg-prod=" not in msg

    def test_secret_value_never_appears_in_mis_wrap_error(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials={"pg-prod": "dsn-string"})
        msg = str(excinfo.value)
        assert "dsn-string" not in msg
