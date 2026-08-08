"""The tox matrix step must allow more time than the envs actually take.

`build (3.10, -e python310)` failed on `main` while its own gate passed: tox
reported `python310: OK (183.96 ... seconds)` and the step was then killed by a
3-minute cap. A green gate reading as a red PR hides real failures, so the cap
needs headroom over the observed runtime.
"""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CI_YAML = PROJECT_ROOT / ".github" / "workflows" / "ci.yaml"

# The slowest env observed in CI, rounded up from the 183.96s in the report.
SLOWEST_OBSERVED_MINUTES = 4


def _tox_matrix_step() -> dict[str, Any]:
    config: dict[Any, Any] = yaml.safe_load(CI_YAML.read_text(encoding="utf-8"))
    steps = config["jobs"]["build"]["steps"]
    matching = [step for step in steps if str(step.get("run", "")).startswith("tox ${{ matrix.toxenv }}")]
    assert len(matching) == 1, f"expected exactly one tox matrix step, found {len(matching)}"
    return dict(matching[0])


def test_tox_matrix_step_timeout_has_headroom() -> None:
    timeout = _tox_matrix_step()["timeout-minutes"]
    assert timeout > SLOWEST_OBSERVED_MINUTES, (
        f"timeout-minutes={timeout} leaves no headroom over the ~{SLOWEST_OBSERVED_MINUTES}min "
        "the slowest tox env takes; a passing gate would fail the step"
    )


def test_tox_matrix_step_still_has_a_timeout() -> None:
    """Headroom, not removal: a hung step must still be killed rather than run to the job limit."""
    step = _tox_matrix_step()
    assert "timeout-minutes" in step
    assert step["timeout-minutes"] <= 15
