"""Record the orchestrators a session creates, for tests that inspect orchestrator internals."""

from typing import Any

import pytest

from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import mlodaAPI


def record_runners(monkeypatch: pytest.MonkeyPatch) -> list[ExecutionOrchestrator]:
    """Wrap ``mlodaAPI._setup_engine_runner``; the returned list gains one orchestrator per run."""
    runners: list[ExecutionOrchestrator] = []
    original = mlodaAPI._setup_engine_runner

    def recording(self: mlodaAPI, *args: Any, **kwargs: Any) -> ExecutionOrchestrator:
        runner = original(self, *args, **kwargs)
        runners.append(runner)
        return runner

    monkeypatch.setattr(mlodaAPI, "_setup_engine_runner", recording)
    return runners
