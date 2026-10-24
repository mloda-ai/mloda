"""Collection contract of the doc fence vocabulary.

Only the subprocess-backed contract lives here; the static string checks moved
to tests/test_docs_fences.py so the default tox env runs them.
"""

import textwrap
from pathlib import Path

import pytest
from mktestdocs import grab_code_blocks

from tests.docs_corpus import ILLUSTRATIVE_TAG, RUNNABLE_TAG
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


def _doc_with(source: str) -> str:
    return f"# Doc\n\n```{RUNNABLE_TAG}\n{source}\n```\n"


@pytest.mark.timeout(120)
def test_runner_calls_doc_defined_test_functions(tmp_path: Path) -> None:
    """A failing doc test_* fails the run naming doc and function; a passing one does not."""
    passing = tmp_path / "passing.md"
    passing.write_text(_doc_with("def test_ok(): assert True"), encoding="utf-8")
    run_md_file_isolated(passing)

    failing = tmp_path / "failing.md"
    failing.write_text(
        _doc_with('def test_ok(): pass\ndef test_fails(): raise AssertionError("doc test ran")'),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError) as exc:
        run_md_file_isolated(failing)
    message = str(exc.value)
    assert str(failing) in message
    assert "test_fails" in message
    assert "doc test ran" in message


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("def test_needs_arg(x): pass", "test_needs_arg"),
        ("async def test_async(): pass", "test_async"),
        ("def test_gen():\n    yield", "test_gen"),
        ("class TestThing:\n    def test_m(self): pass", "TestThing"),
    ],
)
def test_runner_rejects_uncallable_doc_test_shapes(tmp_path: Path, source: str, name: str) -> None:
    doc = tmp_path / "shape.md"
    doc.write_text(_doc_with(source), encoding="utf-8")
    with pytest.raises(AssertionError) as exc:
        run_md_file_isolated(doc)
    message = str(exc.value)
    assert str(doc) in message
    assert name in message
