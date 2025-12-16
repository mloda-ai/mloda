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
)
from mloda.user import PluginLoader


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


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
        """Test that available_only=True (default) filters to only available frameworks."""
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
