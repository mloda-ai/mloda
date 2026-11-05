from abc import ABC, abstractmethod
from typing import Any, final
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.components.plan_freeze import attribute_write_error


class Step(ABC):
    connection_handle: str | None = None
    connection_explicit: bool = False

    def __init__(self, required_uuids: set[UUID]) -> None:
        self.required_uuids = required_uuids
        self.uuid = uuid4()

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen"):
            raise attribute_write_error(type(self).__name__, name)
        object.__setattr__(self, name, value)

    def __copy__(self) -> "Step":
        duplicate = object.__new__(type(self))
        duplicate.__dict__.update(self.__dict__)
        duplicate.__dict__.pop("_frozen", None)
        return duplicate

    def freeze(self) -> None:
        """Make this step's attributes read-only; copies stay writable."""
        self.__dict__["_frozen"] = True

    @abstractmethod
    def execute(
        self,
        cfw_register: CfwManager,
        cfw: ComputeFramework,
        from_cfw: ComputeFramework | UUID | None = None,
        data: Any | None = None,
    ) -> Any | None:
        """Define what executing this step involves."""
        pass

    @abstractmethod
    def get_uuids(self) -> set[UUID]:
        """Return result uuids of this step"""
        return set()

    def get_wait_uuids(self) -> set[UUID]:
        """Tokens that must finish before this step runs."""
        return set(self.required_uuids)

    def get_parallelization_mode(self) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING}

    @final
    def get_result_uuid(self) -> UUID:
        return self.uuid
