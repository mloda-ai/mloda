"""Tests for the safe_field plugin-catalog graceful-degradation helper.

safe_field is the "annotate" tier: it reads one introspected field and, if the
read raises an exception in `catching`, returns a caller-supplied fallback
instead of propagating. Exceptions outside `catching` still propagate.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import safe_field


class TestSafeFieldSuccessPath:
    """When the read succeeds, its value is returned and the fallback is ignored."""

    def test_returns_read_value_on_success(self) -> None:
        assert safe_field(lambda: 42, -1) == 42


class TestSafeFieldDefaultCatching:
    """The default catching=(Exception,) swallows ordinary exceptions."""

    def test_default_catching_swallows_runtime_error(self) -> None:
        def raises() -> str:
            raise RuntimeError("boom")

        assert safe_field(raises, "unavailable") == "unavailable"


class TestSafeFieldNarrowCatching:
    """A narrow catching tuple swallows listed types and propagates the rest."""

    def test_returns_fallback_for_listed_exception_type(self) -> None:
        def raises() -> str:
            raise OSError("disk gone")

        assert safe_field(raises, "fallback", catching=(OSError, TypeError)) == "fallback"

    def test_propagates_unlisted_exception_type(self) -> None:
        def raises() -> str:
            raise ValueError("not caught")

        with pytest.raises(ValueError):
            safe_field(raises, "fallback", catching=(OSError, TypeError))


class TestSafeFieldFallbackIdentity:
    """The fallback value is returned as-is, preserving type and identity."""

    @pytest.mark.parametrize("fallback", [False, [], None, "", 0])
    def test_fallback_returned_as_is_on_failure(self, fallback: Any) -> None:
        def raises() -> Any:
            raise RuntimeError("boom")

        result = safe_field(raises, fallback)

        assert result is fallback
