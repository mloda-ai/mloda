from typing import Any
import pytest
from mloda.user import Feature, mloda, PluginCollector, FeatureName, Options
from mloda.provider import FeatureGroup
from mloda.core.abstract_plugins.components.link import Link, JoinSpec
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.step.join_step import JoinStep
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

class Left1(FeatureGroup):
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'Left1'
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return {'key': [1, 2], 'Left1': ['a', 'b']}

class Left2(FeatureGroup):
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'Left2'
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        return {'key': [1, 2], 'Left2': ['x', 'y']}

class RightTarget(FeatureGroup):
    @classmethod
    def match_feature(cls, feature: Feature) -> bool:
        return feature.name == 'RightTarget'
    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        import time
        time.sleep(0.01) # make it slow enough to overlap
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

def test_1935_issue():
    session = mloda.prepare([Feature(name='Consumer')],
                            links={link1, link2},
                            plugin_collector=PluginCollector.enabled_feature_groups({Left1, Left2, RightTarget, Consumer}),
                            compute_frameworks=[PythonDictFramework],
                            parallelization_modes={ParallelizationMode.THREADING})
    
    join_steps = [s for s in session.engine.execution_planner.execution_plan if isinstance(s, JoinStep)]
    
    # Check if the join steps are ordered by required_uuids
    print('JoinSteps uuids:', [s.uuid for s in join_steps])
    print('JoinSteps required:', [s.required_uuids for s in join_steps])
    print('JoinSteps dest:', [s.destination_framework_uuids for s in join_steps])
    
    unordered = True
    for s in join_steps:
        # if one join step requires the other, they are ordered
        if any(other.uuid in s.required_uuids for other in join_steps if other is not s):
            unordered = False
            break
    
    print('Are they unordered?', unordered)
    
    # Run multiple times to trigger the race condition
    for i in range(10):
        result = list(session.run(parallelization_modes={ParallelizationMode.THREADING}))
        data = {}
        for r in result:
            data.update(r)
        print(f'Run {i} data keys:', data.keys())
