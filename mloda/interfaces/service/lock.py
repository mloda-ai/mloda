"""The process-wide lock every service call into mloda holds, since planning touches shared class state."""

import threading

MLODA_LOCK = threading.RLock()
