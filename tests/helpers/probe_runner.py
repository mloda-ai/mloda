"""Runs a probe script in fresh interpreters, so a test can pin what a cold process decides."""

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

_PROBE_TIMEOUT = 30.0


def run_probes(probe: Path, count: int) -> list[dict[str, str]]:
    """Run ``probe`` ``count`` times, each in its own interpreter, and parse the one json line it prints."""
    assert probe.is_file(), f"{probe} does not exist"

    outputs: list[dict[str, str]] = []
    for _ in range(count):
        # Safe: fixed argv, no shell, no user input.
        completed = subprocess.run(  # nosec B603
            [sys.executable, str(probe)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
        assert completed.returncode == 0, f"probe interpreter failed:\n{completed.stderr}"
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        assert len(lines) == 1, f"probe printed {len(lines)} lines, expected exactly one: {completed.stdout!r}"
        parsed: dict[str, str] = json.loads(lines[0])
        outputs.append(parsed)
    return outputs
