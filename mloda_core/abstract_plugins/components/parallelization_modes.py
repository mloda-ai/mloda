from enum import Enum


class ParallelizationModes(Enum):
    SYNC = "sync"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
