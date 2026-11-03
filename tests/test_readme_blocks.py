"""Every README ```python block runs standalone in a fresh interpreter and prints its ```text block.

Pure parsing plus one subprocess per block, so it lives outside tests/test_documentation,
which the default tox env ignores.
"""

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from tests.docs_corpus import REPO_ROOT
from tests.readme_first_run import _block_failure, _clean_env, python_blocks

README_BLOCKS = python_blocks((REPO_ROOT / "README.md").read_text(encoding="utf-8"))


class TestPythonBlocks:
    """python_blocks pairs each ```python block with the ```text block directly after it."""

    def test_a_python_block_followed_by_a_text_block_is_paired(self) -> None:
        text = "# Doc\n\n```python\nprint(1)\n```\n\n```text\n1\n```\n"

        assert python_blocks(text) == [("print(1)\n", "1\n")]

    def test_a_python_block_without_a_text_block_pairs_with_none(self) -> None:
        text = "# Doc\n\n```python\nprint(1)\n```\n\nProse.\n"

        assert python_blocks(text) == [("print(1)\n", None)]

    def test_a_bash_block_in_between_breaks_the_pairing(self) -> None:
        text = "```python\nprint(1)\n```\n\n```bash\nls\n```\n\n```text\n1\n```\n"

        assert python_blocks(text) == [("print(1)\n", None)]

    def test_an_illustrative_py_block_is_ignored(self) -> None:
        text = "```py\nfragment\n```\n\n```text\nnot paired\n```\n\n```python\nprint(2)\n```\n\n```text\n2\n```\n"

        assert python_blocks(text) == [("print(2)\n", "2\n")]

    def test_blocks_keep_document_order(self) -> None:
        text = "```python\nfirst\n```\n\n```text\nout\n```\n\n```python\nsecond\n```\n"

        assert python_blocks(text) == [("first\n", "out\n"), ("second\n", None)]


def test_readme_has_python_blocks() -> None:
    assert README_BLOCKS, "README.md has no ```python block, so nothing below is checked"


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("index", "code", "expected"),
    [(index, code, expected) for index, (code, expected) in enumerate(README_BLOCKS, 1)],
    ids=[f"block-{index}" for index in range(1, len(README_BLOCKS) + 1)],
)
def test_readme_python_block_runs_standalone(tmp_path: Path, index: int, code: str, expected: str | None) -> None:
    script = tmp_path / "block.py"
    script.write_text(code, encoding="utf-8")
    # PYTHONWARNINGS: optional backends in the full dev env may emit deprecation warnings that are not
    # this test's concern; the bare env (tox -e bare) is the authoritative quiet-run check.
    # Safe: argv is the running interpreter and a script written above, no shell.
    run = subprocess.run(  # nosec B603
        [sys.executable, "-I", str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**_clean_env(), "PYTHONWARNINGS": "ignore"},
        timeout=60,
    )

    failure = _block_failure(index, code, expected, run)
    assert failure is None, failure
