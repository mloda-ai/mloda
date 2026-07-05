import gc
import inspect
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
from mloda.user import PluginLoader


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


class TestGetFeatureGroupDocs:
    def test_get_feature_group_docs_returns_list(self) -> None:
        """Test that get_feature_group_docs returns a list."""
        result = get_feature_group_docs()
        assert isinstance(result, list)

    def test_get_feature_group_docs_returns_non_empty_list(self) -> None:
        """Test that get_feature_group_docs returns a non-empty list (feature groups exist in codebase)."""
        result = get_feature_group_docs()
        assert len(result) > 0, "Expected at least one feature group to be discovered"

    def test_get_feature_group_docs_returns_feature_group_info_objects(self) -> None:
        """Test that all items in the returned list are FeatureGroupInfo instances."""
        result = get_feature_group_docs()
        assert len(result) > 0, "Need at least one result to validate type"
        for item in result:
            assert isinstance(item, FeatureGroupInfo)

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

    def test_get_feature_group_docs_name_filter_exact(self) -> None:
        """Test that name filter works with exact match."""
        # First get all to find a name to filter on
        all_results = get_feature_group_docs()
        assert len(all_results) > 0, "Need at least one feature group for filtering"

        # Pick the first one and filter by exact name
        target_name = all_results[0].name
        filtered = get_feature_group_docs(name=target_name)

        assert len(filtered) >= 1
        assert all(target_name.lower() in fg.name.lower() for fg in filtered)

    def test_get_feature_group_docs_name_filter_partial(self) -> None:
        """Test that name filter works with partial match."""
        # First get all to find a name to filter on
        all_results = get_feature_group_docs()
        assert len(all_results) > 0, "Need at least one feature group for filtering"

        # Pick the first one and use a substring of its name
        target_name = all_results[0].name
        if len(target_name) > 3:
            partial = target_name[:3]
            filtered = get_feature_group_docs(name=partial)

            assert len(filtered) >= 1
            assert all(partial.lower() in fg.name.lower() for fg in filtered)

    def test_get_feature_group_docs_name_filter_case_insensitive(self) -> None:
        """Test that name filter is case-insensitive."""
        all_results = get_feature_group_docs()
        assert len(all_results) > 0, "Need at least one feature group for filtering"

        target_name = all_results[0].name

        # Filter with lowercase
        filtered_lower = get_feature_group_docs(name=target_name.lower())
        # Filter with uppercase
        filtered_upper = get_feature_group_docs(name=target_name.upper())

        # Both should return the same results
        assert len(filtered_lower) == len(filtered_upper)
        assert len(filtered_lower) >= 1

    def test_get_feature_group_docs_search_filter(self) -> None:
        """Test that search filter works on description."""
        all_results = get_feature_group_docs()
        assert len(all_results) > 0, "Need at least one feature group for filtering"

        # Find a feature group with a description we can search for
        target = all_results[0]
        # Pick a word from the description (if multi-word)
        description_words = target.description.split()
        if len(description_words) > 0:
            search_term = description_words[0]
            filtered = get_feature_group_docs(search=search_term)

            # Should find at least the target
            assert len(filtered) >= 1
            # All results should have the search term in their description
            assert all(search_term.lower() in fg.description.lower() for fg in filtered)

    def test_get_feature_group_docs_search_filter_case_insensitive(self) -> None:
        """Test that search filter is case-insensitive."""
        all_results = get_feature_group_docs()
        assert len(all_results) > 0, "Need at least one feature group for filtering"

        target = all_results[0]
        description_words = target.description.split()
        if len(description_words) > 0:
            search_term = description_words[0]

            # Filter with lowercase
            filtered_lower = get_feature_group_docs(search=search_term.lower())
            # Filter with uppercase
            filtered_upper = get_feature_group_docs(search=search_term.upper())

            # Both should return the same results
            assert len(filtered_lower) == len(filtered_upper)
            assert len(filtered_lower) >= 1

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


class TestGetComputeFrameworkDocs:
    def test_get_compute_framework_docs_returns_list(self) -> None:
        """Test that get_compute_framework_docs returns a list."""
        result = get_compute_framework_docs()
        assert isinstance(result, list)

    def test_get_compute_framework_docs_returns_non_empty_list(self) -> None:
        """Test that get_compute_framework_docs returns a non-empty list (compute frameworks exist in codebase)."""
        result = get_compute_framework_docs()
        assert len(result) > 0, "Expected at least one compute framework to be discovered"

    def test_get_compute_framework_docs_returns_compute_framework_info_objects(self) -> None:
        """Test that all items in the returned list are ComputeFrameworkInfo instances."""
        result = get_compute_framework_docs()
        assert len(result) > 0, "Need at least one result to validate type"
        for item in result:
            assert isinstance(item, ComputeFrameworkInfo)

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

    def test_get_compute_framework_docs_name_filter_exact(self) -> None:
        """Test that name filter works with exact match."""
        # First get all to find a name to filter on
        all_results = get_compute_framework_docs()
        assert len(all_results) > 0, "Need at least one compute framework for filtering"

        # Pick the first one and filter by exact name
        target_name = all_results[0].name
        filtered = get_compute_framework_docs(name=target_name)

        assert len(filtered) >= 1
        assert all(target_name.lower() in cfw.name.lower() for cfw in filtered)

    def test_get_compute_framework_docs_name_filter_partial(self) -> None:
        """Test that name filter works with partial match."""
        # First get all to find a name to filter on
        all_results = get_compute_framework_docs()
        assert len(all_results) > 0, "Need at least one compute framework for filtering"

        # Pick the first one and use a substring of its name
        target_name = all_results[0].name
        if len(target_name) > 3:
            partial = target_name[:3]
            filtered = get_compute_framework_docs(name=partial)

            assert len(filtered) >= 1
            assert all(partial.lower() in cfw.name.lower() for cfw in filtered)

    def test_get_compute_framework_docs_name_filter_case_insensitive(self) -> None:
        """Test that name filter is case-insensitive."""
        all_results = get_compute_framework_docs()
        assert len(all_results) > 0, "Need at least one compute framework for filtering"

        target_name = all_results[0].name

        # Filter with lowercase
        filtered_lower = get_compute_framework_docs(name=target_name.lower())
        # Filter with uppercase
        filtered_upper = get_compute_framework_docs(name=target_name.upper())

        # Both should return the same results
        assert len(filtered_lower) == len(filtered_upper)
        assert len(filtered_lower) >= 1

    def test_get_compute_framework_docs_search_filter(self) -> None:
        """Test that search filter works on description."""
        all_results = get_compute_framework_docs()
        assert len(all_results) > 0, "Need at least one compute framework for filtering"

        # Find a compute framework with a description we can search for
        target = all_results[0]
        # Pick a word from the description (if multi-word)
        description_words = target.description.split()
        if len(description_words) > 0:
            search_term = description_words[0]
            filtered = get_compute_framework_docs(search=search_term)

            # Should find at least the target
            assert len(filtered) >= 1
            # All results should have the search term in their description
            assert all(search_term.lower() in cfw.description.lower() for cfw in filtered)

    def test_get_compute_framework_docs_search_filter_case_insensitive(self) -> None:
        """Test that search filter is case-insensitive."""
        all_results = get_compute_framework_docs()
        assert len(all_results) > 0, "Need at least one compute framework for filtering"

        target = all_results[0]
        description_words = target.description.split()
        if len(description_words) > 0:
            search_term = description_words[0]

            # Filter with lowercase
            filtered_lower = get_compute_framework_docs(search=search_term.lower())
            # Filter with uppercase
            filtered_upper = get_compute_framework_docs(search=search_term.upper())

            # Both should return the same results
            assert len(filtered_lower) == len(filtered_upper)
            assert len(filtered_lower) >= 1

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
    def test_get_extender_docs_returns_list(self) -> None:
        """Test that get_extender_docs returns a list."""
        result = get_extender_docs()
        assert isinstance(result, list)

    def test_get_extender_docs_returns_non_empty_list(self) -> None:
        """Test that get_extender_docs returns a non-empty list (extenders exist in codebase)."""
        result = get_extender_docs()
        assert len(result) > 0, "Expected at least one extender to be discovered"

    def test_get_extender_docs_returns_extender_info_objects(self) -> None:
        """Test that all items in the returned list are ExtenderInfo instances."""
        result = get_extender_docs()
        assert len(result) > 0, "Need at least one result to validate type"
        for item in result:
            assert isinstance(item, ExtenderInfo)

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

    def test_get_extender_docs_name_filter_exact(self) -> None:
        """Test that name filter works with exact match."""
        # First get all to find a name to filter on
        all_results = get_extender_docs()
        assert len(all_results) > 0, "Need at least one extender for filtering"

        # Pick the first one and filter by exact name
        target_name = all_results[0].name
        filtered = get_extender_docs(name=target_name)

        assert len(filtered) >= 1
        assert all(target_name.lower() in ext.name.lower() for ext in filtered)

    def test_get_extender_docs_name_filter_partial(self) -> None:
        """Test that name filter works with partial match."""
        # First get all to find a name to filter on
        all_results = get_extender_docs()
        assert len(all_results) > 0, "Need at least one extender for filtering"

        # Pick the first one and use a substring of its name
        target_name = all_results[0].name
        if len(target_name) > 3:
            partial = target_name[:3]
            filtered = get_extender_docs(name=partial)

            assert len(filtered) >= 1
            assert all(partial.lower() in ext.name.lower() for ext in filtered)

    def test_get_extender_docs_name_filter_case_insensitive(self) -> None:
        """Test that name filter is case-insensitive."""
        all_results = get_extender_docs()
        assert len(all_results) > 0, "Need at least one extender for filtering"

        target_name = all_results[0].name

        # Filter with lowercase
        filtered_lower = get_extender_docs(name=target_name.lower())
        # Filter with uppercase
        filtered_upper = get_extender_docs(name=target_name.upper())

        # Both should return the same results
        assert len(filtered_lower) == len(filtered_upper)
        assert len(filtered_lower) >= 1

    def test_get_extender_docs_search_filter(self) -> None:
        """Test that search filter works on description."""
        all_results = get_extender_docs()
        assert len(all_results) > 0, "Need at least one extender for filtering"

        # Find an extender with a description we can search for
        target = all_results[0]
        # Pick a word from the description (if multi-word)
        description_words = target.description.split()
        if len(description_words) > 0:
            search_term = description_words[0]
            filtered = get_extender_docs(search=search_term)

            # Should find at least the target
            assert len(filtered) >= 1
            # All results should have the search term in their description
            assert all(search_term.lower() in ext.description.lower() for ext in filtered)

    def test_get_extender_docs_search_filter_case_insensitive(self) -> None:
        """Test that search filter is case-insensitive."""
        all_results = get_extender_docs()
        assert len(all_results) > 0, "Need at least one extender for filtering"

        target = all_results[0]
        description_words = target.description.split()
        if len(description_words) > 0:
            search_term = description_words[0]

            # Filter with lowercase
            filtered_lower = get_extender_docs(search=search_term.lower())
            # Filter with uppercase
            filtered_upper = get_extender_docs(search=search_term.upper())

            # Both should return the same results
            assert len(filtered_lower) == len(filtered_upper)
            assert len(filtered_lower) >= 1

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
