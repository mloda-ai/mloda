from enum import Enum


class ConnectionRequirement(Enum):
    """How a compute framework obtains the connection it needs to run."""

    NONE = "none"
    SELF_MANAGED = "self_managed"
    REQUIRED = "required"
