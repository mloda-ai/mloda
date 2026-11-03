"""Every README ```python block runs standalone in a fresh interpreter and prints its ```text block.

Pure parsing plus one subprocess per block, so it lives outside tests/test_documentation,
which the default tox env ignores.
"""

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from tests.docs_corpus import REPO_ROOT
from tests.readme_first_run import ReadmeBlock, _block_failure, _clean_env, python_blocks

README_BLOCKS = python_blocks((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
PANDAS_INSTALL = '```bash\npip install "mloda[pandas]"\n```\n'
PRINT_ONE = "```python\nprint(1)\n```\n"


class TestPythonBlocks:
    """python_blocks pairs each ```python block with the ```text block directly after it."""

    def test_a_python_block_followed_by_a_text_block_is_paired(self) -> None:
        text = "# Doc\n\n```python\nprint(1)\n```\n\n```text\n1\n```\n"

        assert python_blocks(text) == [ReadmeBlock("print(1)\n", "1\n", ())]

    def test_a_python_block_without_a_text_block_pairs_with_none(self) -> None:
        text = "# Doc\n\n```python\nprint(1)\n```\n\nProse.\n"

        assert python_blocks(text) == [ReadmeBlock("print(1)\n", None, ())]

    def test_a_bash_block_in_between_breaks_the_pairing(self) -> None:
        text = "```python\nprint(1)\n```\n\n```bash\nls\n```\n\n```text\n1\n```\n"

        assert python_blocks(text) == [ReadmeBlock("print(1)\n", None, ())]

    def test_an_illustrative_py_block_is_ignored(self) -> None:
        text = "```py\nfragment\n```\n\n```text\nnot paired\n```\n\n```python\nprint(2)\n```\n\n```text\n2\n```\n"

        assert python_blocks(text) == [ReadmeBlock("print(2)\n", "2\n", ())]

    def test_blocks_keep_document_order(self) -> None:
        text = "```python\nfirst\n```\n\n```text\nout\n```\n\n```python\nsecond\n```\n"

        assert python_blocks(text) == [ReadmeBlock("first\n", "out\n", ()), ReadmeBlock("second\n", None, ())]


class TestExtrasRule:
    """A python block needs the extras of the code block right before it, if that is a bash pip install of mloda[...]."""

    def test_a_bare_pip_install_gives_no_extras(self) -> None:
        text = "```bash\npip install mloda\n```\n\n" + PRINT_ONE

        assert [block.extras for block in python_blocks(text)] == [()]

    def test_a_pandas_install_then_prose_gives_pandas(self) -> None:
        text = PANDAS_INSTALL + "\nThen change one line.\n\n" + PRINT_ONE

        assert [block.extras for block in python_blocks(text)] == [("pandas",)]

    def test_a_comma_separated_list_gives_every_extra_in_order(self) -> None:
        text = '```bash\npip install "mloda[pandas,polars]"\n```\n\n' + PRINT_ONE

        assert [block.extras for block in python_blocks(text)] == [("pandas", "polars")]

    def test_only_the_first_install_in_the_bash_block_counts(self) -> None:
        text = '```bash\npip install "mloda[pandas]"   # or mloda[polars], mloda[pyarrow]\n```\n\n' + PRINT_ONE

        assert [block.extras for block in python_blocks(text)] == [("pandas",)]

    def test_a_python_block_in_between_resets_to_no_extras(self) -> None:
        text = PANDAS_INSTALL + "\n" + PRINT_ONE + "\n```python\nprint(2)\n```\n"

        assert [block.extras for block in python_blocks(text)] == [("pandas",), ()]

    @pytest.mark.parametrize("tag", ["text", "markdown"])
    def test_a_non_bash_block_in_between_resets_to_no_extras(self, tag: str) -> None:
        text = PANDAS_INSTALL + f"\n```{tag}\nsomething\n```\n\n" + PRINT_ONE

        assert [block.extras for block in python_blocks(text)] == [()]

    def test_a_bash_block_without_a_pip_install_gives_no_extras(self) -> None:
        text = "```bash\nuvx mloda@latest demo\n```\n\n" + PRINT_ONE

        assert [block.extras for block in python_blocks(text)] == [()]

    def test_extras_do_not_affect_the_output_pairing(self) -> None:
        text = PANDAS_INSTALL + "\n" + PRINT_ONE + "\n```text\n1\n```\n"

        assert python_blocks(text) == [ReadmeBlock("print(1)\n", "1\n", ("pandas",))]


def test_readme_has_python_blocks() -> None:
    assert README_BLOCKS, "README.md has no ```python block, so nothing below is checked"


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("index", "code", "expected"),
    [(index, block.code, block.expected) for index, block in enumerate(README_BLOCKS, 1)],
    ids=[f"block-{index}" for index in range(1, len(README_BLOCKS) + 1)],
)
def test_readme_python_block_runs_standalone(tmp_path: Path, index: int, code: str, expected: str | None) -> None:
    script = tmp_path / "block.py"
    script.write_text(code, encoding="utf-8")
    # The dev venv has every backend, so extras are not a concern here; the bare env (tox -e bare) is the
    # authoritative check that each block runs with exactly the extras its README section installs.
    # PYTHONWARNINGS: optional backends in the full dev env may emit deprecation warnings that are not
    # this test's concern; the bare env is also the authoritative quiet-run check.
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
