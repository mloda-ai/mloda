from abc import ABC, abstractmethod
from typing import Any, Optional, Set, final
from uuid import UUID, uuid4

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.core.cfw_manager import CfwManager
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes


class Step(ABC):
    def __init__(self, required_uuids: Set[UUID]) -> None:
        self.required_uuids = required_uuids
        self.uuid = uuid4()
        self.step_is_done = False

    @abstractmethod
    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFrameWork,
        from_cfw: Optional[ComputeFrameWork] = None,
        data: Optional[Any] = None,
    ) -> Optional[Any]:
        """Define what executing this step involves."""
        pass

    @abstractmethod
    def get_uuids(self) -> Set[UUID]:
        """Return result uuids of this step"""
        return set()

    def get_parallelization_mode(self) -> Set[ParallelizationModes]:
        # TODO: This is a placeholder. We will need to add this to feature group later.
        return {ParallelizationModes.SYNC, ParallelizationModes.THREADING, ParallelizationModes.MULTIPROCESSING}

    @final
    def get_result_uuid(self) -> UUID:
        return self.uuid
