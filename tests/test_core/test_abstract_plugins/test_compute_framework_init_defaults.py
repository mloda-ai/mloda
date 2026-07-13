"""Red-phase tests for ComputeFramework __init__ defaults.

Requirements under test:
1. Default-constructed instances must each get a fresh uuid (currently the
   `uuid4()` default is evaluated once at import time, so all instances share it).
2. The base class should be no-arg constructible with mode=ParallelizationMode.SYNC
   and children_if_root=frozenset() (currently raises TypeError).
3. An explicitly passed uuid is honored.
"""

from uuid import UUID, uuid4

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class InitDefaultsFramework(ComputeFramework):
    pass


class TestComputeFrameworkInitDefaults:
    def test_default_uuid_is_unique_per_instance(self) -> None:
        fw1 = InitDefaultsFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        fw2 = InitDefaultsFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert isinstance(fw1.uuid, UUID)
        assert fw1.uuid != fw2.uuid

    def test_no_arg_construction_uses_sensible_defaults(self) -> None:
        fw = InitDefaultsFramework()
        assert fw.mode == ParallelizationMode.SYNC
        assert fw.children_if_root == frozenset()
        assert isinstance(fw.uuid, UUID)

    def test_explicit_uuid_is_honored(self) -> None:
        explicit = uuid4()
        fw = InitDefaultsFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset(), uuid=explicit)
        assert fw.uuid == explicit
