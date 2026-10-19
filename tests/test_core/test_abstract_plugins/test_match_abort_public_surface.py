"""``escalate_match_abort`` and ``is_match_abort`` must be public on ``mloda.provider`` (issue #2027).

Marking a raise as framework-owned is how a plugin escalates a match-time misconfiguration, but the helpers
lived only behind the internal path ``mloda.core.abstract_plugins.components.utils``. The public module must
re-export both names as the same function objects, and the published feature-group matching page must
document them as the supported way to escalate.
"""

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components import utils
from tests.docs_corpus import REPO_ROOT


class TestMatchAbortSurfaceIsExported:
    @pytest.mark.parametrize(
        "name",
        (
            "escalate_match_abort",
            "is_match_abort",
        ),
    )
    def test_name_is_public(self, name: str) -> None:
        assert hasattr(provider, name)
        assert name in provider.__all__, f"{name} should be listed in mloda.provider.__all__"


class TestMatchAbortHelpersAreTheCoreFunctions:
    """Why identity and not just callable behavior.

    The public path must be the same function object as the internal path so existing plugins importing
    from ``mloda.core.abstract_plugins.components.utils`` and new callers importing from ``mloda.provider``
    cannot diverge: a wrapper or duplicate would let the two entry points drift apart.
    """

    def test_escalate_match_abort_is_the_core_function(self) -> None:
        assert provider.escalate_match_abort is utils.escalate_match_abort

    def test_is_match_abort_is_the_core_function(self) -> None:
        assert provider.is_match_abort is utils.is_match_abort


class TestMatchAbortRoundTripThroughPublicPath:
    def test_escalate_returns_the_same_instance_and_marks_it(self) -> None:
        exc = ValueError("boom")

        assert provider.escalate_match_abort(exc) is exc
        assert provider.is_match_abort(exc) is True

    def test_unmarked_value_error_is_not_a_match_abort(self) -> None:
        assert provider.is_match_abort(ValueError("unmarked")) is False

    def test_unmarked_type_error_is_not_a_match_abort(self) -> None:
        assert provider.is_match_abort(TypeError("unmarked")) is False


class TestMatchAbortIsDocumented:
    def test_feature_group_matching_page_names_both_helpers(self) -> None:
        """The docs half of the definition of done: the page must teach both helper names."""
        docs_page = REPO_ROOT / "docs" / "docs" / "in_depth" / "feature-group-matching.md"
        content = docs_page.read_text(encoding="utf-8")

        assert "escalate_match_abort" in content, f"escalate_match_abort is not documented in {docs_page}"
        assert "is_match_abort" in content, f"is_match_abort is not documented in {docs_page}"
