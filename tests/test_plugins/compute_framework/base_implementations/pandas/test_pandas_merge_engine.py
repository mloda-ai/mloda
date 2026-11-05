from mloda._testing.compute_framework.merge_suite import MergeEngineSuite
from mloda._testing.framework_adapter import PandasAdapter


class TestPandasMergeEngine(MergeEngineSuite):
    """Test PandasMergeEngine using the shared merge suite."""

    adapter_class = PandasAdapter
