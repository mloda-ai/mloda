import os
import time
import uuid

_VARIANT_RFC_4122 = 0b10
_VERSION_7 = 0x7

_UNIX_TS_MS_BITS = 48
_VERSION_BITS = 4
_RAND_A_BITS = 12
_VARIANT_BITS = 2
_RAND_B_BITS = 128 - _UNIX_TS_MS_BITS - _VERSION_BITS - _RAND_A_BITS - _VARIANT_BITS

_UNIX_TS_MS_MASK = (1 << _UNIX_TS_MS_BITS) - 1
_RAND_A_MASK = (1 << _RAND_A_BITS) - 1
_RAND_B_MASK = (1 << _RAND_B_BITS) - 1


def generate_run_id() -> str:
    """Mint a UUIDv7 run id (RFC 9562).

    Vendored because ``uuid.uuid7()`` is only available on Python 3.14+, while this repo
    supports >=3.10. Layout: 48-bit unix_ts_ms | 4-bit version (0b0111) | 12-bit rand_a |
    2-bit variant (0b10) | 62-bit rand_b, built from ``time.time_ns()`` and ``os.urandom``.
    """
    unix_ts_ms = (time.time_ns() // 1_000_000) & _UNIX_TS_MS_MASK

    rand_bytes = os.urandom(10)
    rand_a = int.from_bytes(rand_bytes[0:2], "big") & _RAND_A_MASK
    rand_b = int.from_bytes(rand_bytes[2:10], "big") & _RAND_B_MASK

    uuid_int = unix_ts_ms << (128 - _UNIX_TS_MS_BITS)
    uuid_int |= _VERSION_7 << (128 - _UNIX_TS_MS_BITS - _VERSION_BITS)
    uuid_int |= rand_a << (_VARIANT_BITS + _RAND_B_BITS)
    uuid_int |= _VARIANT_RFC_4122 << _RAND_B_BITS
    uuid_int |= rand_b

    return str(uuid.UUID(int=uuid_int))
