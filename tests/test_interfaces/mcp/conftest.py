"""Keeps process-wide log masking from leaking out of tests that run the real serve()."""

import pytest

from mloda.core.abstract_plugins.components import log_masking


@pytest.fixture(autouse=True)
def _restore_process_wide_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log_masking, "_process_wide", False)
