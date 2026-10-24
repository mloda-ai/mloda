"""Pins the three column-wise framework hooks to FeatureChainParserMixin and to every concrete family.

FeatureChainParserMixin declares the hooks as non-abstract classmethods whose body raises
NotImplementedError. That is deliberate: an @abstractmethod on a non-ABC class is silently
unenforced, so it would guarantee nothing. These tests are the enforcement instead. They check
both ends of the contract: the raising default exists on the mixin, and no shipped family
(pandas, PyArrow, Polars, python dict) falls through to it.
"""

from __future__ import annotations

import ast
from abc import ABC, ABCMeta
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from tests.test_plugins.feature_group.experimental.columnwise_hooks_test_mixin import (
    ADD_HOOK,
    CHECK_HOOK,
    DISCOVERY_HOOK,
)

HOOK_NAMES: tuple[str, ...] = (DISCOVERY_HOOK, CHECK_HOOK, ADD_HOOK)

# Anchor the scan root to the repo layout via __file__, not the cwd: a cwd-relative root makes the
# rglob loop empty and the sweep pass vacuously. This file sits four parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group" / "experimental"
assert SCAN_ROOT.exists(), f"scan root not found; check the parents index for the repo root: {SCAN_ROOT}"

# Lower bound only: a new experimental family adds a base.py and must not fail this sweep.
MIN_BASE_MODULES = 12


def sample_frame() -> pd.DataFrame:
    """Return the two-column frame every pandas hook consumer works on."""
    return pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})


class HooklessConsumer(FeatureChainParserMixin):
    """Inherits FeatureChainParserMixin and overrides no hook, so every hook call must raise."""


HOOK_INVOCATIONS: list[tuple[str, Callable[[Any, Any], Any]]] = [
    (DISCOVERY_HOOK, lambda hook, data: hook(data)),
    (CHECK_HOOK, lambda hook, data: hook(data, ["col_a"])),
    (ADD_HOOK, lambda hook, data: hook(data, "hook_result", [1, 2])),
]


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_feature_chain_parser_mixin_declares_hook(hook_name: str) -> None:
    """The mixin owns all three hook declarations, and each one is callable."""
    hook = getattr(FeatureChainParserMixin, hook_name, None)
    assert hook is not None, f"FeatureChainParserMixin does not declare {hook_name}"
    assert callable(hook), f"FeatureChainParserMixin.{hook_name} is not callable"


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_hook_is_not_abstract(hook_name: str) -> None:
    """The hooks must not be decorated @abstractmethod: on a non-ABC class that enforces nothing."""
    hook = getattr(FeatureChainParserMixin, hook_name, None)
    assert hook is not None, f"FeatureChainParserMixin does not declare {hook_name}"
    assert getattr(hook, "__isabstractmethod__", False) is False, f"{hook_name} is marked abstract"


def test_feature_chain_parser_mixin_is_not_an_abc() -> None:
    """The mixin stays a plain class, which is why the hooks raise instead of being abstract."""
    assert not issubclass(FeatureChainParserMixin, ABC)
    assert not isinstance(FeatureChainParserMixin, ABCMeta)


@pytest.mark.parametrize(("hook_name", "invoke"), HOOK_INVOCATIONS, ids=[name for name, _ in HOOK_INVOCATIONS])
def test_unimplemented_hook_raises_not_implemented(hook_name: str, invoke: Callable[[Any, Any], Any]) -> None:
    """A subclass that does not implement a hook fails loudly, naming the class and the hook."""
    hook = getattr(HooklessConsumer, hook_name, None)
    assert hook is not None, f"FeatureChainParserMixin does not declare {hook_name}"

    with pytest.raises(NotImplementedError) as exc_info:
        invoke(hook, sample_frame())

    message = str(exc_info.value)
    assert HooklessConsumer.__name__ in message, f"{hook_name} error omits the class name: {message}"
    assert hook_name in message, f"{hook_name} error omits the hook name: {message}"


def test_no_base_module_declares_a_hook_in_source() -> None:
    """Static sweep: no base.py under the experimental tree re-adds a hook declaration to a class body."""
    offenders: list[str] = []
    visited = 0
    for path in sorted(SCAN_ROOT.rglob("base.py")):
        visited += 1
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            declared = sorted(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in HOOK_NAMES
            )
            if declared:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}::{node.name} declares {declared}")
    assert offenders == [], f"base modules must inherit the hook declarations, found: {offenders}"
    assert visited >= MIN_BASE_MODULES, (
        f"sweep visited only {visited} base.py files under {SCAN_ROOT}: the sweep root is wrong or the glob broke"
    )
