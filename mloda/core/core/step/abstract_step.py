from abc import ABC, abstractmethod
from typing import Any, Optional, Set, Union, final
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


class Step(ABC):
    def __init__(self, required_uuids: Set[UUID]) -> None:
        self.required_uuids = required_uuids
        self.uuid = uuid4()
        self.step_is_done = False

    @abstractmethod
    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFramework,
        from_cfw: Optional[Union[ComputeFramework, UUID]] = None,
        data: Optional[Any] = None,
    ) -> Optional[Any]:
        """Define what executing this step involves."""
        pass

    @abstractmethod
    def get_uuids(self) -> Set[UUID]:
        """Return result uuids of this step"""
        return set()

    def get_parallelization_mode(self) -> Set[ParallelizationMode]:
        # TODO: This is a placeholder. We will need to add this to feature group later.
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING}

    @final
    def get_result_uuid(self) -> UUID:
        return self.uuid
