from enum import Enum


class ParallelizationMode(Enum):
    SYNC = "sync"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
