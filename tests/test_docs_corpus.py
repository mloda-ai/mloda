"""Pins the fence-tag vocabulary and is_python_fence classifier that tests/docs_corpus.py must expose."""

import pytest

from tests.docs_corpus import ILLUSTRATIVE_TAG, OUTPUT_TAG, RUNNABLE_TAG, is_python_fence

PYTHON_LOOKING_INFO_STRINGS = (
    "python",
    " python",
    "py",
    "python3",
    "ipython",
    "pycon",
    "python-repl",
    'python title="x"',
    ' python title="x"',
    "Python",
    "PYTHON",
)

NON_PYTHON_INFO_STRINGS = (
    "json",
    "text",
    "pythonic",
    "pytest",
    "",
    "bash",
    "   ",
    "\t",
)


def test_fence_tag_constants() -> None:
    assert RUNNABLE_TAG == "python"
    assert ILLUSTRATIVE_TAG == "py"
    assert OUTPUT_TAG == "text"


@pytest.mark.parametrize("info", PYTHON_LOOKING_INFO_STRINGS)
def test_is_python_fence_matches_python_looking_info(info: str) -> None:
    assert is_python_fence(info), f"{info!r} should be recognised as a python fence"


@pytest.mark.parametrize("info", NON_PYTHON_INFO_STRINGS)
def test_is_python_fence_rejects_non_python_info(info: str) -> None:
    assert not is_python_fence(info), f"{info!r} should not be recognised as a python fence"
