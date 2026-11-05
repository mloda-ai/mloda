from mloda._testing.compute_framework.merge_suite import MergeEngineSuite
from mloda._testing.framework_adapter import PolarsLazyAdapter


class TestPolarsLazyMergeEngine(MergeEngineSuite):
    """Test PolarsLazyMergeEngine using the shared merge suite."""

    adapter_class = PolarsLazyAdapter
