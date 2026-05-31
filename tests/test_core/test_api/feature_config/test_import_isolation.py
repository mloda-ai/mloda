"""
Import-isolation regression tests for the feature_config submodules.

These tests guard against a circular import between the feature_config
loader/models/parser and the ``mloda.user`` facade. Each submodule must be
importable in a fresh interpreter, i.e. before ``mloda.user`` has been
initialized. If a submodule imports ``Feature``/``Options`` back from
``mloda.user`` (instead of their defining modules), importing it first raises
an ImportError from a partially-initialized ``mloda.user``.
"""

import subprocess  # nosec B404
import sys

import pytest


FEATURE_CONFIG_MODULES = [
    "mloda.core.api.feature_config.loader",
    "mloda.core.api.feature_config.models",
    "mloda.core.api.feature_config.parser",
]


@pytest.mark.parametrize("module_name", FEATURE_CONFIG_MODULES)
def test_submodule_imports_in_fresh_interpreter(module_name: str) -> None:
    """Each feature_config submodule must import in a fresh interpreter.

    Spawning a new interpreter and importing only the submodule reproduces the
    "import the submodule before mloda.user" condition that triggered the
    circular import. A non-zero return code means the cycle has returned.
    """
    # Safe: fixed argv (sys.executable + hardcoded module name), no shell, no user input.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Importing {module_name} in a fresh interpreter failed; "
        f"a circular import with mloda.user has likely returned.\n"
        f"stderr:\n{result.stderr}"
    )
