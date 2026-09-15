"""Runs a probe script in fresh interpreters, so a test can pin what a cold process decides."""

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

_PROBE_TIMEOUT = 30.0


def run_probes(probe: Path, count: int, seeds: list[int] | None = None) -> list[dict[str, str]]:
    """Run ``probe`` ``count`` times, each in its own interpreter, and parse the one json line it prints.

    ``seeds``, if given, pins ``PYTHONHASHSEED`` to one value per process (one process per seed,
    ``count`` is then ignored) so a hash-order-dependent bug reproduces deterministically instead
    of relying on the luck of a random per-process seed.
    """
    assert probe.is_file(), f"{probe} does not exist"

    envs: list[dict[str, str] | None]
    if seeds is not None:
        envs = [{**os.environ, "PYTHONHASHSEED": str(seed)} for seed in seeds]
    else:
        envs = [None for _ in range(count)]

    processes = [
        # Safe: fixed argv, no shell, no user input.
        subprocess.Popen(  # nosec B603
            [sys.executable, str(probe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for env in envs
    ]

    outputs: list[dict[str, str]] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=_PROBE_TIMEOUT)
            assert process.returncode == 0, f"probe interpreter failed:\n{stderr}"
            lines = [line for line in stdout.splitlines() if line.strip()]
            assert len(lines) == 1, f"probe printed {len(lines)} lines, expected exactly one: {stdout!r}"
            parsed: dict[str, str] = json.loads(lines[0])
            outputs.append(parsed)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    return outputs
