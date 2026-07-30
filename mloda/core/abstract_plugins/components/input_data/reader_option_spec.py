"""The ``READER_OPTIONS`` spec type: a deliberately weaker sibling of ``PropertySpec``.

Reader option keys are consumed at MATCH time, inside reader selection, which runs before the
framework materializes any ``PROPERTY_MAPPING`` default, so nothing declared here is ever applied
by the framework. ``runtime_default`` only RECORDS the fallback the reader's own code applies when
the key is absent, and ``framework_set`` marks a key the framework writes rather than the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReaderOptionSpec:
    """Frozen declaration of one reader option key."""

    explanation: str
    runtime_default: Any = None
    framework_set: bool = False
