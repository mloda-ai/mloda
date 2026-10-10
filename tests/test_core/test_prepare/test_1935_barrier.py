from typing import Any
import pytest
import threading
from mloda.user import Feature, mloda, PluginCollector, FeatureName, Options
from mloda.provider import FeatureGroup
from mloda.core.abstract_plugins.components.link import Link, JoinSpec
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

barrier = threading.Barrier(2)

class BlockingDict(dict):
    def keys(self):
        try:
            barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError:
            pass
        return super().keys()
class Left1(FeatureGroup):
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'Left1'
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return BlockingDict({'key': [1, 2], 'Left1': ['a', 'b']})

class Left2(FeatureGroup):
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'Left2'
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return BlockingDict({'key': [1, 2], 'Left2': ['x', 'y']})

class RightTarget(FeatureGroup):
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'RightTarget'
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return {'key': [1, 2], 'RightTarget': ['r1', 'r2']}

link1 = Link.right(JoinSpec(Left1, 'key'), JoinSpec(RightTarget, 'key'))
link2 = Link.right(JoinSpec(Left2, 'key'), JoinSpec(RightTarget, 'key'))

class Consumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name='Left1'), Feature(name='Left2'), Feature(name='RightTarget')}
        
    @classmethod
    def depends_on(cls):
        return {link1, link2}
        
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'Consumer'
        
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        assert 'Left1' in data, 'Left1 missing due to race'
        assert 'Left2' in data, 'Left2 missing due to race'
        length = len(next(iter(data.values()))) if data else 2
        data['Consumer'] = [1] * length
        return data

def test_1935_issue_barrier():
    barrier.reset()
    session = mloda.prepare([Feature(name='Consumer')],
                            links={link1, link2},
                            plugin_collector=PluginCollector.enabled_feature_groups({Left1, Left2, RightTarget, Consumer}),
                            compute_frameworks=[PythonDictFramework],
                            parallelization_modes={ParallelizationMode.THREADING})
    
    result = list(session.run(parallelization_modes={ParallelizationMode.THREADING}))
