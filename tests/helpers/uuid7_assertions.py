"""Shared assertion that a string is a valid RFC 9562 UUIDv7."""

import uuid


def assert_valid_uuid7(value: str) -> None:
    parsed = uuid.UUID(value)
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
