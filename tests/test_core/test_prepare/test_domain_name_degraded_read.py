"""Direct unit tests for IdentifyFeatureGroupClass._domain_name and _fails_domain_gate.

A plugin-owned Domain read (get_domain() or the Domain.name property) must never escape either
method, and a RAISE inside that read must leave the domain gate undecided, not decided/dead.
"""

from __future__ import annotations

import logging

import pytest

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from tests.helpers.plugin_stubs import StubFeatureGroup

IDENTIFY_LOGGER_NAME = IdentifyFeatureGroupClass.__module__


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == IDENTIFY_LOGGER_NAME
    ]


class RaisingNameDomain(Domain):
    """A Domain subclass that IS-A Domain but whose name property raises."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:  # type: ignore[override]  # deliberate LSP violation: the test needs a raising read
        raise RuntimeError("domain name explodes")


class BadStrError(Exception):
    def __str__(self) -> str:
        raise RuntimeError("str fails")


def _build_raising_name_domain_fg() -> type[StubFeatureGroup]:
    """Mint a StubFeatureGroup subclass, per test, whose get_domain() returns a raising-name Domain.

    Minted inside a function rather than at module level: a module-level FeatureGroup subclass stays
    forever in FeatureGroup.__subclasses__() for the rest of the pytest-xdist worker's session.
    """

    class RaisingNameDomainFG(StubFeatureGroup):
        @classmethod
        def get_domain(cls) -> Domain:
            return RaisingNameDomain()

    return RaisingNameDomainFG


def _build_raising_get_domain_fg() -> type[StubFeatureGroup]:
    """Mint a StubFeatureGroup subclass, per test, whose get_domain() itself raises a BadStrError."""

    class RaisingGetDomainFG(StubFeatureGroup):
        @classmethod
        def get_domain(cls) -> Domain:
            raise BadStrError("original message")

    return RaisingGetDomainFG


class TestDomainNameSwallowsRaisingNameProperty:
    """get_domain() returns a Domain instance whose .name property raises."""

    def test_returns_none_instead_of_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        raising_name_domain_fg = _build_raising_name_domain_fg()

        with caplog.at_level(logging.WARNING, logger=IDENTIFY_LOGGER_NAME):
            result = IdentifyFeatureGroupClass()._domain_name(raising_name_domain_fg)

        assert result is None

    def test_logs_one_warning_naming_the_field(self, caplog: pytest.LogCaptureFixture) -> None:
        raising_name_domain_fg = _build_raising_name_domain_fg()

        with caplog.at_level(logging.WARNING, logger=IDENTIFY_LOGGER_NAME):
            IdentifyFeatureGroupClass()._domain_name(raising_name_domain_fg)

        messages = _warning_messages(caplog)
        assert len(messages) == 1, f"Expected exactly one WARNING, got {messages}"
        assert f"{raising_name_domain_fg.get_class_name()}.get_domain" in messages[0]


class TestDomainNameSwallowsRaisingStrOnGetDomainError:
    """get_domain() raises an exception whose own __str__ raises."""

    def test_returns_none_instead_of_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        raising_get_domain_fg = _build_raising_get_domain_fg()

        with caplog.at_level(logging.WARNING, logger=IDENTIFY_LOGGER_NAME):
            result = IdentifyFeatureGroupClass()._domain_name(raising_get_domain_fg)

        assert result is None

    def test_logs_one_warning_naming_the_field_and_exception_type(self, caplog: pytest.LogCaptureFixture) -> None:
        raising_get_domain_fg = _build_raising_get_domain_fg()

        with caplog.at_level(logging.WARNING, logger=IDENTIFY_LOGGER_NAME):
            IdentifyFeatureGroupClass()._domain_name(raising_get_domain_fg)

        messages = _warning_messages(caplog)
        assert len(messages) == 1, f"Expected exactly one WARNING, got {messages}"
        # The full format, not two substring checks: "BadStrError" fills both the type-name slot and the
        # guarded-str fallback slot, so a substring check alone would pass even if the guard were broken.
        field = f"{raising_get_domain_fg.get_class_name()}.get_domain"
        assert messages[0] == f"Degraded field '{field}': raised BadStrError: BadStrError"


class TestFailsDomainGateSwallowsRaisingNameProperty:
    """A candidate whose Domain.name raises must leave the domain gate undecided, not decided/dead.

    _fails_domain_gate's own contract (see its docstring) is: a RAISE anywhere in reading the domain
    leaves the gate undecided (False, candidate stays live), while a MALFORMED RETURN is decided (True,
    candidate drops). get_domain() here returns a real Domain whose .name property raises: a RAISE, not
    a malformed return, so the gate must return False.
    """

    def test_raising_name_property_leaves_gate_undecided(self) -> None:
        raising_name_domain_fg = _build_raising_name_domain_fg()
        feature = Feature("some_name", domain="some_domain")

        result = IdentifyFeatureGroupClass()._fails_domain_gate(raising_name_domain_fg, feature)
        # Dropped before the assert: a failing assert's traceback would otherwise pin this local, keeping
        # the class reachable past teardown's gc.collect() and tripping the registry-leak fixture (#845).
        del raising_name_domain_fg

        assert result is False
