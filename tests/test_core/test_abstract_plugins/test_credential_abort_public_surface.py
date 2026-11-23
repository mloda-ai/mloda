"""Credential refusals can be recognized without importing core internals."""

import pytest

import mloda.provider as provider
from mloda.user import Credential, DataAccessCollection, Feature, PluginCollector, mloda
from mloda.user.pandas import PandasDataFrame
from tests.helpers.credential_handle_stubs import HANDLE_KEY, ChsVaultFG


def test_is_credential_abort_is_public() -> None:
    assert "is_credential_abort" in provider.__all__
    assert callable(provider.is_credential_abort)


@pytest.mark.parametrize("options", [{}, {HANDLE_KEY: "unregistered-handle"}], ids=["missing", "unregistered"])
def test_public_predicate_recognizes_plan_time_handle_refusals(options: dict[str, str]) -> None:
    secret = "credential-abort-test-value"  # nosec B105
    dac = DataAccessCollection(credentials={"vault-main": Credential(key=secret)})

    with pytest.raises(ValueError) as exc:
        mloda.prepare(
            [Feature("chs_vault_a", options=options)],
            compute_frameworks=[PandasDataFrame],
            data_access_collection=dac,
            plugin_collector=PluginCollector.enabled_feature_groups({ChsVaultFG}),
        )

    assert HANDLE_KEY in str(exc.value)
    assert secret not in str(exc.value)
    assert provider.is_credential_abort(exc.value)
    wrapped = RuntimeError("plugin-owned text")
    wrapped.__cause__ = exc.value
    assert not provider.is_credential_abort(wrapped)


def test_public_predicate_does_not_authorize_other_error_text() -> None:
    exc = ValueError("an option value may occur here")
    assert not provider.is_credential_abort(exc)
    provider.escalate_match_abort(exc)
    assert provider.is_match_abort(exc)
    assert not provider.is_credential_abort(exc)
