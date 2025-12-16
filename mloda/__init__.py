from mloda.core.api.request import mlodaAPI as API
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup as FeatureGroup
from mloda.core.abstract_plugins.compute_framework import ComputeFramework as ComputeFramework

# Module-level API alias and function for `import mloda; mloda.API(...)` pattern
run_all = API.run_all

__all__ = [
    "API",
    "run_all",
    "Feature",
    "Options",
    "FeatureGroup",
    "ComputeFramework",
]
