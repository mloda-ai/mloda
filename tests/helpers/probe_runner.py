"""Runs a probe script in fresh interpreters, so a test can pin what a cold process decides."""

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

_PROBE_TIMEOUT = 30.0


def run_probes(probe: Path, count: int) -> list[dict[str, str]]:
    """Run ``probe`` ``count`` times, each in its own interpreter, and parse the one json line it prints."""
    assert probe.is_file(), f"{probe} does not exist"

    processes = [
        # Safe: fixed argv, no shell, no user input.
        subprocess.Popen(  # nosec B603
            [sys.executable, str(probe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(count)
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
