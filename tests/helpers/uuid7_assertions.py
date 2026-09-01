"""Shared assertion that a string is a valid RFC 9562 UUIDv7.

Used by the run_id generator tests, the mlodaAPI session.run_id tests, and the HookContext
run_id/carrier wiring tests.
"""

import uuid


def assert_valid_uuid7(value: str) -> None:
    """Assert ``value`` parses as a UUID whose version is 7 and whose variant is RFC 4122."""
    parsed = uuid.UUID(value)
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
