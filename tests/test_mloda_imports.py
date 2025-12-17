"""
Test file demonstrating the mloda namespace imports.

Structure:
- `mloda` - Quick start essentials (5 items)
- `mloda.user` - Full Data User toolkit (17 items)
- `mloda.provider` - Data Provider base classes (23 items)
- `mloda.steward` - Data Steward governance (8 items)
"""

import os


def test_import_quick_start() -> None:
    """import mloda pattern for quick start"""
    import mloda
    from mloda import API

    # Verify module-level function is callable
    assert callable(mloda.run_all)

    # Verify all expected exports exist
    assert mloda.Feature is not None
    assert mloda.Options is not None
    assert mloda.FeatureGroup is not None
    assert mloda.ComputeFramework is not None
    assert mloda.API is not None
    assert mloda.API is API

    # When running in installed testenv, verify it's from site-packages
    if os.environ.get("MLODA_INSTALLED_TEST"):
        assert "site-packages" in mloda.__file__, f"Expected installed package, got: {mloda.__file__}"


def test_import_user_full() -> None:
    """from mloda.user import ... (17 items for Data Users)"""
    from mloda.user import (
        # API
        API,
        # Features
        Feature,
        Features,
        FeatureName,
        Options,
        Domain,
        # Link & Index
        Link,
        JoinType,
        JoinSpec,
        Index,
        # Filtering
        GlobalFilter,
        SingleFilter,
        FilterType,
        # Data access
        DataAccessCollection,
        # Types
        DataType,
        ParallelizationMode,
        # Plugin discovery
        PluginLoader,
        PluginCollector,
    )

    # API
    assert API is not None
    # Features
    assert Feature is not None
    assert Features is not None
    assert FeatureName is not None
    assert Options is not None
    assert Domain is not None
    # Link & Index
    assert Link is not None
    assert JoinType is not None
    assert JoinSpec is not None
    assert Index is not None
    # Filtering
    assert GlobalFilter is not None
    assert SingleFilter is not None
    assert FilterType is not None
    # Data access
    assert DataAccessCollection is not None
    # Types
    assert DataType is not None
    assert ParallelizationMode is not None
    # Plugin discovery
    assert PluginLoader is not None
    assert PluginCollector is not None


def test_import_provider_base_classes() -> None:
    """from mloda.provider import ... (23 items for Data Providers)"""
    from mloda.provider import (
        # Base classes
        FeatureGroup,
        ComputeFramework,
        # Versioning
        FeatureGroupVersion,
        # Feature set
        FeatureSet,
        # Input data
        BaseInputData,
        ApiData,
        ApiDataFeatureGroup,
        BaseApiDataSchema,
        ApiDataCollection,
        DataCreator,
        # Match data
        ConnectionMatcherMixin,
        # Artifact
        BaseArtifact,
        # Validators
        BaseValidator,
        FeatureValidator,
        FeatureSetValidator,
        OptionsValidator,
        LinkValidator,
        DataTypeValidator,
        DataTypeMismatchError,
        # Feature chaining
        FeatureChainParser,
        FeatureChainParserMixin,
        # Transformers
        BaseTransformer,
        ComputeFrameworkTransformer,
        # Engines
        BaseFilterEngine,
        BaseMergeEngine,
    )

    # Base classes
    assert FeatureGroup is not None
    assert ComputeFramework is not None
    # Versioning
    assert FeatureGroupVersion is not None
    # Feature set
    assert FeatureSet is not None
    # Input data
    assert BaseInputData is not None
    assert ApiData is not None
    assert ApiDataFeatureGroup is not None
    assert BaseApiDataSchema is not None
    assert ApiDataCollection is not None
    assert DataCreator is not None
    # Match data
    assert ConnectionMatcherMixin is not None
    # Artifact
    assert BaseArtifact is not None
    # Validators
    assert BaseValidator is not None
    assert FeatureValidator is not None
    assert FeatureSetValidator is not None
    assert OptionsValidator is not None
    assert LinkValidator is not None
    assert DataTypeValidator is not None
    assert DataTypeMismatchError is not None
    # Feature chaining
    assert FeatureChainParser is not None
    assert FeatureChainParserMixin is not None
    # Transformers
    assert BaseTransformer is not None
    assert ComputeFrameworkTransformer is not None
    # Engines
    assert BaseFilterEngine is not None
    assert BaseMergeEngine is not None


def test_import_steward_governance() -> None:
    """from mloda.steward import ... (8 items for Data Stewards)"""
    from mloda.steward import (
        # Plugin inspection
        FeatureGroupInfo,
        ComputeFrameworkInfo,
        ExtenderInfo,
        # Documentation
        get_feature_group_docs,
        get_compute_framework_docs,
        get_extender_docs,
        # Function extenders (audit, monitoring, observability)
        Extender,
        ExtenderHook,
    )

    # Plugin inspection
    assert FeatureGroupInfo is not None
    assert ComputeFrameworkInfo is not None
    assert ExtenderInfo is not None
    # Documentation
    assert get_feature_group_docs is not None
    assert get_compute_framework_docs is not None
    assert get_extender_docs is not None
    # Function extenders
    assert Extender is not None
    assert ExtenderHook is not None


def test_all_roles_demo() -> None:
    """Demo: Three roles with explicit modules"""
    # Data User
    from mloda.user import API, Feature, Options

    # Data Provider
    from mloda.provider import FeatureGroup, FeatureGroupVersion

    # Data Steward
    from mloda.steward import get_feature_group_docs

    assert API is not None
    assert Feature is not None
    assert Options is not None
    assert FeatureGroup is not None
    assert FeatureGroupVersion is not None
    assert get_feature_group_docs is not None
