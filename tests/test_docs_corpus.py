"""Pins the fence-tag vocabulary and is_python_fence classifier that tests/docs_corpus.py must expose."""

import pytest

from tests.docs_corpus import (
    DOCS_ROOT,
    ILLUSTRATIVE_TAG,
    OUTPUT_TAG,
    REPO_ROOT,
    RUNNABLE_TAG,
    doc_files,
    is_python_fence,
)

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


def test_doc_files_default_returns_docs_root_files() -> None:
    files = doc_files()
    assert files
    assert files == sorted(DOCS_ROOT.rglob("*.md"))


def test_doc_files_accepts_explicit_root() -> None:
    other_root = DOCS_ROOT / "in_depth"
    files = doc_files(other_root)
    assert files
    assert files == sorted(other_root.rglob("*.md"))
    assert files != doc_files()


def test_doc_id_returns_repo_relative_posix_path() -> None:
    from tests.docs_corpus import doc_id

    fpath = doc_files()[0]
    result = doc_id(fpath)
    assert result == fpath.relative_to(REPO_ROOT).as_posix()
    assert not result.startswith("/")
    assert "\\" not in result
