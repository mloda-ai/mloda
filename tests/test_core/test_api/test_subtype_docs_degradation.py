"""Tests for the subtype surfaces around resolve_feature and the docs (issue #639 follow-up).

After #755 resolve_feature delegates matching to the engine seam, which does not guard
supports_compute_framework: a raising hook propagates out of evaluate. resolve_feature catches it to
preserve its never-raises contract and fails closed (no winner, error carries the hook's failure
text). The old "degrade OPEN + warn" behavior is gone. A bogus 'supported' framework name still
surfaces as subtype_error in the docs.
"""

import logging
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureGroup, SubtypeDeclaration, property_spec
from mloda.steward import FeatureGroupInfo, get_feature_group_docs, resolve_feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


SBFIX_HOOK_FEATURE = "sbfix_raising_hook_feature"
SBFIX_DOC_KEY = "sbfixdoc_kind"
SBFIX_BOGUS_FRAMEWORK = "SbfixNoSuchDocFramework"


class SbfixRaisingHookFG(FeatureGroup):
    """Hand-written supports_compute_framework hook that raises when consulted."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SBFIX_HOOK_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        raise RuntimeError("sbfix hook exploded")

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SbfixDocBogusSupportedFG(FeatureGroup):
    """Declaration whose 'supported' names a framework outside compute_framework_rule()."""

    SUBTYPES = SubtypeDeclaration(
        key=SBFIX_DOC_KEY,
        supported={SBFIX_BOGUS_FRAMEWORK: {"sum"}},
    )
    PROPERTY_MAPPING = {
        SBFIX_DOC_KEY: property_spec(
            "Operation kind with a bogus supported framework.",
            strict=True,
            allowed_values={"sum": "Sum", "median": "Median"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _sbfix_doc_for(name: str) -> FeatureGroupInfo:
    exact = [doc for doc in get_feature_group_docs(name=name) if doc.name == name]
    assert len(exact) == 1, f"expected exactly one doc for {name}, got {[doc.name for doc in exact]}"
    return exact[0]


class TestSbfixRaisingHookFailsClosed:
    """A raising capability hook fails closed: no winner, error carries the hook text, no raise."""

    def test_raising_hook_fails_closed_without_raising(self) -> None:
        # The seam does not guard supports_compute_framework, so the hook raises from inside evaluate.
        # resolve_feature catches it: no winner, and the hook's failure text surfaces in the error.
        result = resolve_feature(SBFIX_HOOK_FEATURE)
        assert result.feature_group is None
        assert result.error is not None
        assert "sbfix hook exploded" in result.error

    def test_raising_hook_is_surfaced_not_logged_as_degradation(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = resolve_feature(SBFIX_HOOK_FEATURE)
        assert result.feature_group is None
        assert result.error is not None
        assert "sbfix hook exploded" in result.error


class TestSbfixBogusSupportedSurfacedInDocs:
    """Docs surface the undeclared 'supported' framework as subtype_error."""

    def test_docs_report_subtype_error_with_empty_support(self) -> None:
        doc = _sbfix_doc_for("SbfixDocBogusSupportedFG")
        assert doc.subtype_key == SBFIX_DOC_KEY
        assert doc.subtype_support == {}
        assert doc.subtype_error is not None
        assert SBFIX_BOGUS_FRAMEWORK in doc.subtype_error
        assert "SbfixDocBogusSupportedFG" in doc.subtype_error
