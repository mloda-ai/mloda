"""Collection contract of the doc fence vocabulary.

Only the subprocess-backed contract lives here; the static string checks moved
to tests/test_docs_fences.py so the default tox env runs them.
"""

import textwrap
from pathlib import Path

import pytest
from mktestdocs import grab_code_blocks

from tests.test_docs_fences import ILLUSTRATIVE_TAG, RUNNABLE_TAG
from tests.test_documentation.test_documentation import run_md_file_isolated

FAILING_SNIPPET = 'raise RuntimeError("doc fence collection contract")'

RUNNABLE_MD = textwrap.dedent(
    f"""\
    # Runnable

    ```{RUNNABLE_TAG}
    {FAILING_SNIPPET}
    ```
    """
)

ILLUSTRATIVE_MD = textwrap.dedent(
    f"""\
    # Illustrative

    ```{ILLUSTRATIVE_TAG}
    {FAILING_SNIPPET}
    ```
    """
)


@pytest.mark.timeout(120)
def test_collection_contract_of_python_and_py_fences(tmp_path: Path) -> None:
    """```python is executed by the doc runner, ```py is not collected at all."""
    runnable = tmp_path / "runnable.md"
    runnable.write_text(RUNNABLE_MD, encoding="utf-8")
    with pytest.raises(AssertionError):
        run_md_file_isolated(runnable)

    assert grab_code_blocks(ILLUSTRATIVE_MD, lang="python") == [], (
        f"```{ILLUSTRATIVE_TAG} reached the collector, so it would be executed"
    )
