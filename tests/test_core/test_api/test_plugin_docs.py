import gc
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from mloda.core.api.plugin_info import (
    FeatureGroupInfo,
    ComputeFrameworkInfo,
    ExtenderInfo,
)
from mloda.core.api.plugin_docs import (
    get_feature_group_docs,
    get_compute_framework_docs,
    get_extender_docs,
    _safe_version,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.user import PluginLoader
from tests.helpers.plugin_stubs import make_raising_fg

# Docs enumeration source-hashes every FeatureGroup subclass; a cold cache under xdist load can exceed the default timeout.
pytestmark = pytest.mark.timeout(30)

SAFE_FIELD_LOGGER = "mloda.core.abstract_plugins.components.utils"


class _DocsCatalogExtender(Extender):
    """Module-level Extender double so get_extender_docs() has a subclass to find in isolation."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


@pytest.fixture(autouse=True)
def _reap_pending_dead_plugin_classes() -> None:
    """Collect dead test-local plugin classes before each test.

    Plugin docs enumeration walks live __subclasses__() registries. Local plugin
    classes from earlier tests on the same worker stay visible there until a gc
    pass runs, so a pass landing between two enumeration calls inside one test
    changes the result mid-test. Collecting up front makes enumeration stable
    for the duration of each test.
    """
    gc.collect()


class TestFeatureGroupInfo:
    def test_feature_group_info_instantiation(self) -> None:
        info = FeatureGroupInfo(
            name="test_group",
            description="A test feature group",
            version="1.0.0",
            module="mloda_plugins.test_module",
            compute_frameworks=["pandas", "polars"],
            supported_feature_names={"feature1", "feature2"},
            prefix="test_",
        )
        assert info.name == "test_group"
        assert info.description == "A test feature group"
        assert info.version == "1.0.0"
        assert info.module == "mloda_plugins.test_module"
        assert info.compute_frameworks == ["pandas", "polars"]
        assert info.supported_feature_names == {"feature1", "feature2"}
        assert info.prefix == "test_"


class TestComputeFrameworkInfo:
    def test_compute_framework_info_instantiation(self) -> None:
        info = ComputeFrameworkInfo(
            name="pandas",
            description="Pandas compute framework",
            module="mloda_plugins.pandas_framework",
            is_available=True,
            expected_data_framework="pandas",
            has_merge_engine=True,
            has_filter_engine=True,
        )
        assert info.name == "pandas"
        assert info.description == "Pandas compute framework"
        assert info.module == "mloda_plugins.pandas_framework"
        assert info.is_available is True
        assert info.expected_data_framework == "pandas"
        assert info.has_merge_engine is True
        assert info.has_filter_engine is True


class TestExtenderInfo:
    def test_extender_info_instantiation(self) -> None:
        info = ExtenderInfo(
            name="test_extender",
            description="An extender that wraps other frameworks",
            module="mloda_plugins.extender_module",
            wraps=["pandas", "polars"],
        )
        assert info.name == "test_extender"
        assert info.description == "An extender that wraps other frameworks"
        assert info.module == "mloda_plugins.extender_module"
        assert info.wraps == ["pandas", "polars"]


@dataclass(frozen=True)
class DocKind:
    label: str
    get_docs: Callable[..., list[Any]]
    info_class: type[Any]


DOC_KINDS: list[DocKind] = [
    DocKind("feature group", get_feature_group_docs, FeatureGroupInfo),
    DocKind("compute framework", get_compute_framework_docs, ComputeFrameworkInfo),
    DocKind("extender", get_extender_docs, ExtenderInfo),
]


@pytest.mark.parametrize("kind", DOC_KINDS, ids=[kind.label.replace(" ", "_") for kind in DOC_KINDS])
class TestDocsGetterSharedBehaviour:
    """Enumeration and the name=/search= filters behave the same for every doc kind."""

    def test_returns_list(self, kind: DocKind) -> None:
        assert isinstance(kind.get_docs(), list)

    def test_returns_non_empty_list(self, kind: DocKind) -> None:
        assert len(kind.get_docs()) > 0, f"Expected at least one {kind.label} to be discovered"

    def test_returns_info_objects(self, kind: DocKind) -> None:
        result = kind.get_docs()
        assert len(result) > 0, "Need at least one result to validate type"
        for item in result:
            assert isinstance(item, kind.info_class)

    def test_name_filter_exact(self, kind: DocKind) -> None:
        all_results = kind.get_docs()
        assert len(all_results) > 0, f"Need at least one {kind.label} for filtering"

        target_name = all_results[0].name
        filtered = kind.get_docs(name=target_name)

        assert len(filtered) >= 1
        assert all(target_name.lower() in entry.name.lower() for entry in filtered)

    def test_name_filter_partial(self, kind: DocKind) -> None:
        all_results = kind.get_docs()
        assert len(all_results) > 0, f"Need at least one {kind.label} for filtering"

        target_name = next((entry.name for entry in all_results if len(entry.name) > 3), None)
        assert target_name is not None, f"Need a {kind.label} name long enough to slice a substring from"

        partial = target_name[:3]
        filtered = kind.get_docs(name=partial)

        assert len(filtered) >= 1
        assert all(partial.lower() in entry.name.lower() for entry in filtered)

    def test_name_filter_case_insensitive(self, kind: DocKind) -> None:
        all_results = kind.get_docs()
        assert len(all_results) > 0, f"Need at least one {kind.label} for filtering"

        target_name = all_results[0].name
        filtered_lower = kind.get_docs(name=target_name.lower())
        filtered_upper = kind.get_docs(name=target_name.upper())

        assert len(filtered_lower) == len(filtered_upper)
        assert len(filtered_lower) >= 1

    def test_search_filter(self, kind: DocKind) -> None:
        all_results = kind.get_docs()
        assert len(all_results) > 0, f"Need at least one {kind.label} for filtering"

        description_words = all_results[0].description.split()
        assert len(description_words) > 0, f"Need a {kind.label} description to take a search term from"

        search_term = description_words[0]
        filtered = kind.get_docs(search=search_term)

        assert len(filtered) >= 1
        assert all(search_term.lower() in entry.description.lower() for entry in filtered)

    def test_search_filter_case_insensitive(self, kind: DocKind) -> None:
        all_results = kind.get_docs()
        assert len(all_results) > 0, f"Need at least one {kind.label} for filtering"

        description_words = all_results[0].description.split()
        assert len(description_words) > 0, f"Need a {kind.label} description to take a search term from"

        search_term = description_words[0]
        filtered_lower = kind.get_docs(search=search_term.lower())
        filtered_upper = kind.get_docs(search=search_term.upper())

        assert len(filtered_lower) == len(filtered_upper)
        assert len(filtered_lower) >= 1


class TestGetFeatureGroupDocs:
    def test_get_feature_group_docs_has_required_fields(self) -> None:
        """Test that each FeatureGroupInfo has all required fields populated."""
        result = get_feature_group_docs()
        assert len(result) > 0, "Need at least one result to validate fields"

        for fg_info in result:
            # All fields should be populated
            assert isinstance(fg_info.name, str) and len(fg_info.name) > 0
            assert isinstance(fg_info.description, str) and len(fg_info.description) > 0
            assert isinstance(fg_info.version, str) and len(fg_info.version) > 0
            assert isinstance(fg_info.module, str) and len(fg_info.module) > 0
            assert isinstance(fg_info.compute_frameworks, list)
            assert isinstance(fg_info.supported_feature_names, set)
            assert isinstance(fg_info.prefix, str) and len(fg_info.prefix) > 0

    def test_get_feature_group_docs_compute_framework_filter_case_insensitive(self) -> None:
        """Test that the compute_framework filter matches the framework name case-insensitively.

        Issue #537 requirement 2: the compute_framework filter compares names
        case-sensitively, so a casing mismatch wrongly returns an empty list.
        """
        all_results = get_feature_group_docs()
        assert len(all_results) > 0, "Need at least one feature group for filtering"

        # Pick a real framework name that at least one feature group supports.
        canonical_name: str | None = None
        for fg in all_results:
            if len(fg.compute_frameworks) > 0:
                canonical_name = fg.compute_frameworks[0]
                break
        assert canonical_name is not None, "Need a feature group with at least one compute framework"

        # Derive the expected match count from the single unfiltered enumeration
        # above, so we do not pay for an extra canonical-case enumeration here.
        expected = sum(1 for fg in all_results if canonical_name.lower() in {c.lower() for c in fg.compute_frameworks})
        assert expected >= 1, "Canonical framework should match at least one feature group"

        lower_filtered = get_feature_group_docs(compute_framework=canonical_name.lower())
        upper_filtered = get_feature_group_docs(compute_framework=canonical_name.upper())

        assert len(lower_filtered) == expected
        assert len(upper_filtered) == expected


# Not frozen: a row's expected value is the very list or set the docs field returns, and the __hash__
# frozen generates would raise on those.
@dataclass
class DegradedFieldCase:
    """One broken hook, the labelled read it degrades, the docs field that read feeds and its fallback."""

    hook: str
    read: str
    class_name: str
    field: str
    expected: str | list[str] | set[str]
    doc: str | None = None

    @property
    def case_id(self) -> str:
        """Names the hook; a hook read twice is told apart by the docstring the double carries."""
        return self.hook if self.doc is not None else f"{self.hook}_without_docstring"


DEGRADED_FIELD_CASES: list[DegradedFieldCase] = [
    DegradedFieldCase(
        hook="get_class_name",
        read="get_class_name",
        class_name="_DocsGetClassNameBoomFG",
        field="name",
        expected="_DocsGetClassNameBoomFG",
        doc="Test double whose get_class_name() raises.",
    ),
    DegradedFieldCase(
        # The fallback is base-class-derived, not "": an empty description would hide the broken
        # plugin from every search= query, which is the masking risk the degradation avoids.
        hook="description",
        read="description",
        class_name="_DocsDescriptionBoomFG",
        field="description",
        expected="Test double whose description() raises.",
        doc="Test double whose description() raises.",
    ),
    DegradedFieldCase(
        # No docstring to fall back on, so the fallback walks one step further, to the class name.
        hook="description",
        read="description",
        class_name="_DocsDescriptionNoDocstringFG",
        field="description",
        expected="_DocsDescriptionNoDocstringFG",
    ),
    DegradedFieldCase(
        hook="compute_framework_definition",
        read="compute_framework_definition",
        class_name="_DocsFrameworkBoomFG",
        field="compute_frameworks",
        expected=[],
        doc="Test double whose compute_framework_definition() raises.",
    ),
    DegradedFieldCase(
        # The realistic break: the definition hook is @final, so a real plugin breaks framework
        # discovery by raising from the overridable rule hook that final method calls. The raise
        # therefore surfaces at that final method, which is where the read is labelled.
        hook="compute_framework_rule",
        read="compute_framework_definition",
        class_name="_DocsFrameworkRuleBoomFG",
        field="compute_frameworks",
        expected=[],
        doc="Test double whose compute_framework_rule() raises.",
    ),
    DegradedFieldCase(
        hook="feature_names_supported",
        read="feature_names_supported",
        class_name="_DocsFeatureNamesBoomFG",
        field="supported_feature_names",
        expected=set(),
        doc="Test double whose feature_names_supported() raises.",
    ),
    DegradedFieldCase(
        # The base-class convention "<__name__>_".
        hook="prefix",
        read="prefix",
        class_name="_DocsPrefixBoomFG",
        field="prefix",
        expected="_DocsPrefixBoomFG_",
        doc="Test double whose prefix() raises.",
    ),
]


class TestGetFeatureGroupDocsDegradedFieldReads:
    """A FeatureGroup with one broken introspection hook degrades that field, it does not sink the catalog.

    ``get_feature_group_docs`` reads five plugin-overridable classmethods per class.
    Each read routes through ``safe_field`` with a base-class-derived fallback
    (issue #609), mirroring the ``is_available`` guard already proven for compute
    frameworks. Degraded entries stay in the catalog and are filtered on their
    degraded values.

    Isolation: every test double is minted inside its test function and reaped in a
    ``finally`` block (``del`` plus ``gc.collect()``), because plugin docs walk the live
    ``__subclasses__()`` registry and a leaked class would corrupt sibling tests' catalog
    calls. No fixture may own the double, since only the test-local name holds it. This
    mirrors ``test_get_compute_framework_docs_degrades_when_is_available_raises``.
    """

    @pytest.mark.parametrize("case", DEGRADED_FIELD_CASES, ids=[case.case_id for case in DEGRADED_FIELD_CASES])
    def test_raising_hook_degrades_its_field_to_the_base_class_fallback(
        self, case: DegradedFieldCase, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One broken hook degrades one documented field, the class stays in the catalog."""
        # The hook is a string here, so a misspelled one would mint a double that breaks nothing.
        assert hasattr(FeatureGroup, case.hook), f"{case.hook} is not a FeatureGroup hook"

        double = make_raising_fg(case.class_name, case.hook, doc=case.doc)
        try:
            with caplog.at_level(logging.WARNING, logger=SAFE_FIELD_LOGGER):
                by_name = {fg.name: fg for fg in get_feature_group_docs()}
            assert case.class_name in by_name, f"{case.hook} raising must not drop the class from the catalog"
            degraded: str | list[str] | set[str] = getattr(by_name[case.class_name], case.field)
            assert degraded == case.expected

            # The value alone is vacuous: for most rows the fallback is also the healthy answer, so only
            # the warning the guarded read emits proves THIS hook is the one that degraded.
            expected_warning = f"Degraded field '{case.class_name}.{case.read}'"
            messages = [
                record.getMessage()
                for record in caplog.records
                if record.levelno == logging.WARNING and record.name == SAFE_FIELD_LOGGER
            ]
            assert any(expected_warning in message for message in messages), (
                f"Expected a WARNING naming {expected_warning}, got {messages}"
            )
        finally:
            del double
            gc.collect()

    def test_degraded_compute_framework_rule_excluded_by_compute_framework_filter(self) -> None:
        """A class whose compute_framework_rule() raises is excluded by a compute_framework= filter."""
        baseline = get_feature_group_docs()
        canonical_name: str | None = None
        for fg in baseline:
            if len(fg.compute_frameworks) > 0:
                canonical_name = fg.compute_frameworks[0]
                break
        assert canonical_name is not None, "Need a feature group with at least one compute framework"

        double = make_raising_fg(
            "_DocsFrameworkRuleFilterBoomFG",
            "compute_framework_rule",
            doc="Test double whose compute_framework_rule() raises.",
        )
        try:
            unfiltered = {fg.name for fg in get_feature_group_docs()}
            assert "_DocsFrameworkRuleFilterBoomFG" in unfiltered, "Degraded class must still be documented"

            filtered = get_feature_group_docs(compute_framework=canonical_name)
            assert len(filtered) > 0, "Healthy feature groups must still match the framework filter"
            assert "_DocsFrameworkRuleFilterBoomFG" not in {fg.name for fg in filtered}
        finally:
            del double
            gc.collect()

    def test_broken_feature_group_does_not_sink_the_catalog(self) -> None:
        """With a raising subclass live, every other feature group is still listed."""
        baseline = {fg.name for fg in get_feature_group_docs()}
        assert len(baseline) > 0, "Need a populated baseline catalog"

        double = make_raising_fg("_DocsSinkBoomFG", "description", doc="Test double whose description() raises.")
        try:
            degraded = {fg.name for fg in get_feature_group_docs()}
            assert baseline.issubset(degraded), "A broken plugin must not drop healthy feature groups"
            assert "_DocsSinkBoomFG" in degraded
        finally:
            del double
            gc.collect()

    def test_degraded_name_still_findable_by_name_filter(self) -> None:
        """A class whose get_class_name() raises is findable via name=<its real __name__>."""
        double = make_raising_fg(
            "_DocsNameFilterBoomFG", "get_class_name", doc="Test double whose get_class_name() raises."
        )
        try:
            filtered = get_feature_group_docs(name="_DocsNameFilterBoomFG")
            assert [fg.name for fg in filtered] == ["_DocsNameFilterBoomFG"]
        finally:
            del double
            gc.collect()

    def test_degraded_description_still_findable_by_search_filter(self) -> None:
        """A class whose description() raises stays findable via search= on its docstring.

        Degrading to the docstring keeps a broken plugin discoverable; degrading to ""
        would make it invisible to every search query.
        """
        double = make_raising_fg(
            "_DocsSearchBoomFG", "description", doc="Test double whose description() raises, keyword lodestarquux."
        )
        try:
            searched = {fg.name for fg in get_feature_group_docs(search="lodestarquux")}
            assert "_DocsSearchBoomFG" in searched, "A degraded description must stay searchable via the docstring"
        finally:
            del double
            gc.collect()

    def test_degraded_description_without_docstring_findable_by_class_name_search(self) -> None:
        """With no docstring, the degraded description is the class name and search= finds it there."""
        double = make_raising_fg("_DocsSearchNoDocstringBoomFG", "description")
        try:
            searched = {fg.name for fg in get_feature_group_docs(search="_DocsSearchNoDocstringBoomFG")}
            assert "_DocsSearchNoDocstringBoomFG" in searched
        finally:
            del double
            gc.collect()

    def test_degraded_compute_frameworks_excluded_by_compute_framework_filter(self) -> None:
        """A class whose compute_framework_definition() raises is excluded by a compute_framework= filter.

        This mirrors how a compute framework degraded to is_available=False is excluded
        by available_only=True: the degraded value, not the exception, drives the filter.
        """
        baseline = get_feature_group_docs()
        canonical_name: str | None = None
        for fg in baseline:
            if len(fg.compute_frameworks) > 0:
                canonical_name = fg.compute_frameworks[0]
                break
        assert canonical_name is not None, "Need a feature group with at least one compute framework"

        double = make_raising_fg(
            "_DocsFrameworkFilterBoomFG",
            "compute_framework_definition",
            doc="Test double whose compute_framework_definition() raises.",
        )
        try:
            unfiltered = {fg.name for fg in get_feature_group_docs()}
            assert "_DocsFrameworkFilterBoomFG" in unfiltered, "Degraded class must still be documented"

            filtered = get_feature_group_docs(compute_framework=canonical_name)
            assert len(filtered) > 0, "Healthy feature groups must still match the framework filter"
            assert "_DocsFrameworkFilterBoomFG" not in {fg.name for fg in filtered}
        finally:
            del double
            gc.collect()


class TestGetFeatureGroupDocsContractViolations:
    """A plugin that returns the WRONG TYPE degrades through the same path as one that raises.

    Raising is not the only way a plugin breaks its contract. A ``description()`` that
    returns ``None`` raises nothing, so a guard keyed only on exceptions never fires and
    the catalog call dies later, in the ``search=`` filter, with an ``AttributeError``.
    The two reads that feed filters (``get_class_name`` for ``name=`` and the sort,
    ``description`` for ``search=``) must validate the returned type inside the guard so a
    contract violation degrades to the same base-class-derived fallback.

    Isolation follows the sibling class: doubles are function-local and reaped in ``finally``.
    """

    def test_description_returning_none_does_not_sink_the_search_filter(self) -> None:
        """description() -> None must not blow up search=, it degrades to the docstring."""

        class _DocsDescriptionNoneFG(FeatureGroup):
            """Test double whose description() returns None, keyword ripcordquux."""

            @classmethod
            def description(cls) -> Any:
                return None

        try:
            searched = {fg.name for fg in get_feature_group_docs(search="ripcordquux")}
            assert "_DocsDescriptionNoneFG" in searched, "A None description must degrade to the docstring"
        finally:
            del _DocsDescriptionNoneFG
            gc.collect()

    def test_description_returning_none_degrades_to_class_docstring(self) -> None:
        """The unfiltered catalog reports the base-class-derived fallback, not None."""

        class _DocsDescriptionNoneUnfilteredFG(FeatureGroup):
            """Test double whose description() returns None."""

            @classmethod
            def description(cls) -> Any:
                return None

        try:
            by_name = {fg.name: fg for fg in get_feature_group_docs()}
            assert "_DocsDescriptionNoneUnfilteredFG" in by_name, "Contract-violating class must still be documented"
            assert (
                by_name["_DocsDescriptionNoneUnfilteredFG"].description
                == "Test double whose description() returns None."
            )
        finally:
            del _DocsDescriptionNoneUnfilteredFG
            gc.collect()

    def test_get_class_name_returning_non_str_does_not_sink_the_name_filter(self) -> None:
        """get_class_name() -> non-str must not blow up name=, it degrades to the class __name__."""

        class _DocsClassNameNonStrFG(FeatureGroup):
            """Test double whose get_class_name() returns an int."""

            @classmethod  # type: ignore[misc]
            def get_class_name(cls) -> Any:
                return 42

        try:
            filtered = get_feature_group_docs(name="_DocsClassNameNonStrFG")
            assert [fg.name for fg in filtered] == ["_DocsClassNameNonStrFG"]
        finally:
            del _DocsClassNameNonStrFG
            gc.collect()

    def test_get_class_name_returning_non_str_degrades_in_unfiltered_catalog(self) -> None:
        """The unfiltered catalog (which sorts by name) also survives a non-str name."""

        class _DocsClassNameNonStrUnfilteredFG(FeatureGroup):
            """Test double whose get_class_name() returns an int."""

            @classmethod  # type: ignore[misc]
            def get_class_name(cls) -> Any:
                return 42

        try:
            by_name = {fg.name: fg for fg in get_feature_group_docs()}
            assert "_DocsClassNameNonStrUnfilteredFG" in by_name, "Contract-violating class must still be documented"
            assert by_name["_DocsClassNameNonStrUnfilteredFG"].name == "_DocsClassNameNonStrUnfilteredFG"
        finally:
            del _DocsClassNameNonStrUnfilteredFG
            gc.collect()

    def test_version_returning_non_str_does_not_sink_the_version_filter(self) -> None:
        """version() -> non-str must not blow up version_contains=, it degrades to "unavailable".

        The third read that feeds a filter. A non-str version passes the exception-only guard
        and then dies in ``version_contains not in version``, exactly as the None description
        dies in the search filter.
        """

        class _DocsVersionNonStrFG(FeatureGroup):
            """Test double whose version() returns an int."""

            @classmethod
            def version(cls) -> Any:
                return 42

        try:
            filtered = get_feature_group_docs(version_contains="unavailable")
            assert "_DocsVersionNonStrFG" in {fg.name for fg in filtered}, (
                "A non-str version must degrade to the documented sentinel and stay filterable"
            )
        finally:
            del _DocsVersionNonStrFG
            gc.collect()

    def test_version_returning_non_str_degrades_to_unavailable(self) -> None:
        """The unfiltered catalog reports the "unavailable" sentinel, not the non-str value."""

        class _DocsVersionNonStrUnfilteredFG(FeatureGroup):
            """Test double whose version() returns an int."""

            @classmethod
            def version(cls) -> Any:
                return 42

        try:
            by_name = {fg.name: fg for fg in get_feature_group_docs()}
            assert "_DocsVersionNonStrUnfilteredFG" in by_name, "Contract-violating class must still be documented"
            assert by_name["_DocsVersionNonStrUnfilteredFG"].version == "unavailable"
        finally:
            del _DocsVersionNonStrUnfilteredFG
            gc.collect()

    def test_version_degradation_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """_safe_version is unlabelled by design, so a degraded version read logs nothing."""

        class _DocsVersionSilentFG(FeatureGroup):
            """Test double whose version() returns an int."""

            @classmethod
            def version(cls) -> Any:
                return 42

        try:
            with caplog.at_level(logging.DEBUG, logger=SAFE_FIELD_LOGGER):
                by_name = {fg.name: fg for fg in get_feature_group_docs()}

            assert by_name["_DocsVersionSilentFG"].version == "unavailable"

            messages = [record.getMessage() for record in caplog.records if record.name == SAFE_FIELD_LOGGER]
            assert messages == [], f"An unlabelled version degradation must be silent, got {messages}"
        finally:
            del _DocsVersionSilentFG
            gc.collect()


class TestDegradedReadLogging:
    """Feature-group reads are labelled, so they warn. Compute-framework reads are not, so they stay silent.

    The labels are the only thing that makes a degraded plugin diagnosable, so they are
    asserted here: dropping ``field=`` from the feature-group call sites must fail a test.
    The compute-framework call sites are deliberately unlabelled, because they degrade by
    design on a healthy system (an uninstalled optional backend, a deliberately
    unimplemented merge engine), and warning on those is log spam.
    """

    def test_degraded_feature_group_read_logs_warning_naming_class_and_field(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _DocsWarnBoomFG(FeatureGroup):
            """Test double whose description() raises."""

            @classmethod
            def description(cls) -> str:
                raise RuntimeError("boom")

        try:
            with caplog.at_level(logging.WARNING, logger=SAFE_FIELD_LOGGER):
                get_feature_group_docs()

            messages = [
                record.getMessage()
                for record in caplog.records
                if record.levelno == logging.WARNING and record.name == SAFE_FIELD_LOGGER
            ]
            matching = [msg for msg in messages if "_DocsWarnBoomFG.description" in msg]
            assert len(matching) >= 1, f"Expected a WARNING naming '_DocsWarnBoomFG.description', got {messages}"
            assert "boom" in matching[0], "Warning must carry the swallowed exception message"
        finally:
            del _DocsWarnBoomFG
            gc.collect()

    def test_degraded_compute_framework_read_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Compute-framework field reads are unlabelled by design, so a degraded read is silent."""

        class _DocsSilentBoomCFW(ComputeFramework):
            """Test double whose availability probe raises."""

            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("boom")

        try:
            with caplog.at_level(logging.DEBUG, logger=SAFE_FIELD_LOGGER):
                results = get_compute_framework_docs()

            by_name = {cfw.name: cfw for cfw in results}
            assert "_DocsSilentBoomCFW" in by_name, "Degraded framework must still be documented"
            assert by_name["_DocsSilentBoomCFW"].is_available is False

            messages = [record.getMessage() for record in caplog.records if record.name == SAFE_FIELD_LOGGER]
            assert messages == [], f"Unlabelled compute-framework reads must degrade silently, got {messages}"
        finally:
            del _DocsSilentBoomCFW
            gc.collect()


class TestGetComputeFrameworkDocs:
    def test_get_compute_framework_docs_has_required_fields(self) -> None:
        """Test that each ComputeFrameworkInfo has all required fields populated."""
        result = get_compute_framework_docs()
        assert len(result) > 0, "Need at least one result to validate fields"

        for cfw_info in result:
            # All fields should be populated
            assert isinstance(cfw_info.name, str) and len(cfw_info.name) > 0
            assert isinstance(cfw_info.description, str) and len(cfw_info.description) > 0
            assert isinstance(cfw_info.module, str) and len(cfw_info.module) > 0
            assert isinstance(cfw_info.is_available, bool)
            assert isinstance(cfw_info.expected_data_framework, str)
            assert isinstance(cfw_info.has_merge_engine, bool)
            assert isinstance(cfw_info.has_filter_engine, bool)

    def test_get_compute_framework_docs_available_only_true_filters_correctly(self) -> None:
        """Test that available_only=True filters to only available frameworks."""
        result = get_compute_framework_docs(available_only=True)
        # All results should have is_available=True
        for cfw_info in result:
            assert cfw_info.is_available is True, f"Framework {cfw_info.name} should be available"

    def test_get_compute_framework_docs_available_only_false_returns_all(self) -> None:
        """Test that available_only=False returns all frameworks regardless of availability."""
        all_results = get_compute_framework_docs(available_only=False)
        available_only_results = get_compute_framework_docs(available_only=True)

        # available_only=False should return same or more frameworks than available_only=True
        assert len(all_results) >= len(available_only_results)

    def test_get_compute_framework_docs_available_only_default_lists_all(self) -> None:
        """Test that the default call does not filter by availability.

        Issue #537 requirement 1: the default should be available_only=False so a
        bare doc call lists ALL frameworks (with is_available as a flag) rather
        than silently dropping frameworks whose backing library is not importable.

        The behavioral set-equality assertion below only diverges when some
        framework is unavailable, which is not guaranteed in every environment
        (e.g. an all-extras venv has every backing library installed). The
        signature-default assertion encodes the requirement deterministically:
        the default must be available_only=False.
        """
        default_value = inspect.signature(get_compute_framework_docs).parameters["available_only"].default
        assert default_value is False, "Default of available_only should be False so a bare call lists all frameworks"

        default_names = {cfw.name for cfw in get_compute_framework_docs()}
        all_names = {cfw.name for cfw in get_compute_framework_docs(available_only=False)}
        assert default_names == all_names

    def test_get_compute_framework_docs_degrades_when_is_available_raises(self) -> None:
        """A framework whose is_available() raises must degrade to is_available=False, not sink the catalog.

        Like the sibling field reads (expected_data_framework, merge_engine, filter_engine),
        the availability probe routes through safe_field, and a degraded framework is excluded
        by available_only=True exactly as a genuinely unavailable one would be (issue #533).
        """

        class _DocsIsAvailableBoomCFW(ComputeFramework):
            """Test double whose availability probe raises."""

            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("boom")

        try:
            # The broken class must not take the whole catalog call down.
            results = get_compute_framework_docs(available_only=False)
            assert len(results) > 0, "Broken framework must not sink the whole catalog"

            by_name = {cfw.name: cfw for cfw in results}
            assert "_DocsIsAvailableBoomCFW" in by_name, "Broken framework should still be documented"
            # It degrades to unavailable, matching the sibling guards.
            assert by_name["_DocsIsAvailableBoomCFW"].is_available is False

            # available_only=True must exclude it because it degraded to unavailable.
            available_names = {cfw.name for cfw in get_compute_framework_docs(available_only=True)}
            assert "_DocsIsAvailableBoomCFW" not in available_names
        finally:
            # Reap the test-local subclass so sibling tests are unaffected (live __subclasses__).
            del _DocsIsAvailableBoomCFW
            gc.collect()


class TestSafeVersionGuard:
    """Characterization tests pinning the narrow exception guard in ``_safe_version``.

    ``_safe_version`` delegates to ``safe_field(..., catching=(OSError, TypeError))``.
    The guard must stay narrow: only the listed types degrade to "unavailable";
    every other exception (e.g. ``ValueError``) must propagate so unrelated bugs
    are not silently swallowed. If a future edit drops the ``catching=`` argument,
    the guard widens to ``except Exception`` and the ValueError case below flips
    from raising to returning "unavailable", failing this test.
    """

    def test_safe_version_reraises_unlisted_exception(self) -> None:
        """A ValueError from version() is NOT in (OSError, TypeError), so it propagates."""

        class _StubFG:
            @staticmethod
            def version() -> str:
                raise ValueError("unlisted exception must propagate")

        with pytest.raises(ValueError):
            _safe_version(_StubFG)  # type: ignore[arg-type]

    def test_safe_version_degrades_on_listed_oserror(self) -> None:
        """An OSError from version() IS listed, so it degrades to "unavailable"."""

        class _StubFG:
            @staticmethod
            def version() -> str:
                raise OSError("listed exception degrades")

        assert _safe_version(_StubFG) == "unavailable"  # type: ignore[arg-type]

    def test_safe_version_degrades_when_attribute_lookup_raises(self) -> None:
        """A TypeError raised during the ``version`` attribute lookup itself degrades to "unavailable".

        The annotate tier must keep the whole read inside the guard, including
        attribute resolution, not just the ``version()`` call.
        """

        class _RaisingDescriptor:
            def __get__(self, obj: Any, owner: type) -> Any:
                raise TypeError("attribute lookup fails")

        class _StubFG:
            version = _RaisingDescriptor()

        assert _safe_version(_StubFG) == "unavailable"  # type: ignore[arg-type]


class TestGetExtenderDocs:
    def test_get_extender_docs_has_required_fields(self) -> None:
        """Test that each ExtenderInfo has all required fields populated."""
        result = get_extender_docs()
        assert len(result) > 0, "Need at least one result to validate fields"

        for ext_info in result:
            # All fields should be populated
            assert isinstance(ext_info.name, str) and len(ext_info.name) > 0
            assert isinstance(ext_info.description, str) and len(ext_info.description) > 0
            assert isinstance(ext_info.module, str) and len(ext_info.module) > 0
            assert isinstance(ext_info.wraps, list)

    def test_get_extender_docs_wraps_filter(self) -> None:
        """Test that wraps filter works when filtering by wrapped function type."""
        all_results = get_extender_docs()
        assert len(all_results) > 0, "Need at least one extender for filtering"

        # Find an extender that wraps at least one function type
        target = None
        for ext in all_results:
            if len(ext.wraps) > 0:
                target = ext
                break

        if target and len(target.wraps) > 0:
            # Use the first wrapped function type as filter
            wrap_type = target.wraps[0]
            filtered = get_extender_docs(wraps=wrap_type)

            # Should find at least the target
            assert len(filtered) >= 1
            # All results should wrap this function type
            assert all(wrap_type in ext.wraps for ext in filtered)
