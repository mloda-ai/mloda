"""
Test file for mloda namespace package and imports.

Structure:
- `mloda` - PEP 420 namespace package (no __init__.py)
- `mloda.user` - Full Data User toolkit
- `mloda.provider` - Data Provider base classes
- `mloda.steward` - Data Steward governance
"""

import os


# =============================================================================
# PEP 420 Namespace Package Tests
# =============================================================================


def test_mloda_is_namespace_package() -> None:
    """Verify mloda is a PEP 420 namespace package (no __init__.py at root)."""
    import mloda

    # Namespace packages have __path__ but __file__ is None
    assert hasattr(mloda, "__path__")
    assert mloda.__file__ is None, f"mloda should be namespace package but has __file__: {mloda.__file__}"


def test_mloda_namespace_path_is_iterable() -> None:
    """Verify mloda.__path__ is iterable (namespace package behavior)."""
    import mloda

    assert hasattr(mloda, "__path__")
    paths = list(mloda.__path__)
    assert len(paths) >= 1, "Namespace package should have at least one path"

    # When running in installed testenv, verify it's from site-packages via __path__
    if os.environ.get("MLODA_INSTALLED_TEST"):
        path_str = str(paths)
        assert "site-packages" in path_str, f"Expected installed package, got: {path_str}"


# =============================================================================
# mloda.user Module Tests (Data User toolkit)
# =============================================================================


def test_import_user_full() -> None:
    """from mloda.user import ... (Data User toolkit)"""
    from mloda.user import (
        # mloda
        mlodaAPI,
        mloda,
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

    # mloda
    assert mloda is not None
    assert mlodaAPI is not None
    assert hasattr(mlodaAPI, "run_all")
    assert callable(mloda.run_all)
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


# =============================================================================
# mloda.provider Module Tests (Data Provider base classes)
# =============================================================================


def test_import_provider_base_classes() -> None:
    """from mloda.provider import ... (Data Provider base classes)"""
    from mloda.provider import (
        # Base classes
        FeatureGroup,
        ComputeFramework,
        # Versioning
        BaseFeatureGroupVersion,
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
    assert BaseFeatureGroupVersion is not None
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


# =============================================================================
# mloda.steward Module Tests (Data Steward governance)
# =============================================================================


def test_import_steward_governance() -> None:
    """from mloda.steward import ... (Data Steward governance)"""
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


# =============================================================================
# Cross-Module Integration Tests
# =============================================================================


def test_all_roles_demo() -> None:
    """Demo: Three roles with explicit modules"""
    # Data User
    from mloda.user import mloda, Feature, Options

    # Data Provider
    from mloda.provider import FeatureGroup, BaseFeatureGroupVersion

    # Data Steward
    from mloda.steward import get_feature_group_docs

    assert mloda is not None
    assert Feature is not None
    assert Options is not None
    assert FeatureGroup is not None
    assert BaseFeatureGroupVersion is not None
    assert get_feature_group_docs is not None
