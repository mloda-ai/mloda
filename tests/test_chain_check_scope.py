"""Regression: input-chain step-aside must not re-fire FEATURE_GROUP_MATCHED or recurse deeply."""

from mloda.core.abstract_plugins.components.input_data.claim_route import (
    chain_check_scope,
    in_chain_check,
)


def test_in_chain_check_defaults_false() -> None:
    assert in_chain_check() is False


def test_chain_check_scope_sets_and_resets() -> None:
    with chain_check_scope():
        assert in_chain_check() is True
    assert in_chain_check() is False


def test_chain_check_scope_nests() -> None:
    with chain_check_scope():
        with chain_check_scope():
            assert in_chain_check() is True
        assert in_chain_check() is True
    assert in_chain_check() is False
