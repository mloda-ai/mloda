from enum import Enum


class ParallelizationMode(Enum):
    SYNC = "sync"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"


def runs_in_worker(register_modes: set[ParallelizationMode], step_modes: set[ParallelizationMode]) -> bool:
    """True when a step is dispatched to a worker process."""
    return ParallelizationMode.MULTIPROCESSING in register_modes & step_modes
