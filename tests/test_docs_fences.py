"""Fence vocabulary guard for docs/docs markdown.

Pure string parsing with no runtime dependency, so it lives outside
tests/test_documentation, which the default tox env ignores.
"""

import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import cast

import pytest
from mktestdocs import grab_code_blocks

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCS_ROOT = REPO_ROOT / "docs" / "docs"

RUNNABLE_TAG = "python"
ILLUSTRATIVE_TAG = "py"
OUTPUT_TAG = "text"

PYTHON_LOOKING = re.compile(r"i?py(thon)?[0-9]*(-repl)?|pycon", re.IGNORECASE)

VOCABULARY_HINT = (
    f"Use ```{RUNNABLE_TAG} when the block runs, ```{ILLUSTRATIVE_TAG} when it is an "
    f"illustrative fragment that must not run, ```{OUTPUT_TAG} when it is output."
)

# Repo-relative doc path -> permitted number of ```py blocks. An absent file permits zero.
ILLUSTRATIVE_BLOCK_ALLOWLIST: dict[str, int] = {
    "docs/docs/chapter1/api-request.md": 1,
    "docs/docs/chapter1/compute-frameworks.md": 4,
    "docs/docs/in_depth/access-feature-data.md": 5,
    "docs/docs/in_depth/artifacts.md": 1,
    "docs/docs/in_depth/compute-framework-integration.md": 12,
    "docs/docs/in_depth/data-access-patterns.md": 7,
    "docs/docs/in_depth/data-type-enforcement.md": 1,
    "docs/docs/in_depth/discover-plugins.md": 1,
    "docs/docs/in_depth/feature-chain-parser.md": 16,
    "docs/docs/in_depth/feature-config.md": 2,
    "docs/docs/in_depth/feature-group-matching.md": 3,
    "docs/docs/in_depth/feature-group-testing.md": 5,
    "docs/docs/in_depth/filter_data.md": 1,
    "docs/docs/in_depth/framework-connection-object.md": 9,
    "docs/docs/in_depth/framework-transformers.md": 4,
    "docs/docs/in_depth/join_data.md": 4,
    "docs/docs/in_depth/mloda-api.md": 5,
    "docs/docs/in_depth/multiple_result_columns.md": 5,
    "docs/docs/in_depth/named-data-access-handles.md": 4,
    "docs/docs/in_depth/plugin-loader.md": 1,
    "docs/docs/in_depth/plugin_registry.md": 1,
    "docs/docs/in_depth/property-mapping.md": 4,
    "docs/docs/in_depth/streaming.md": 6,
    "docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md": 3,
}

MAX_REPORTED_VIOLATIONS = 60


FENCE_OPENING = re.compile(r"(?P<marker>`{3,}|~{3,})(?P<info>.*)$")


class Fence:
    """One opening code fence in a markdown file."""

    def __init__(self, path: Path, lineno: int, text: str, marker: str, info: str) -> None:
        self.path = path
        self.lineno = lineno
        self.text = text
        self.marker = marker
        self.info = info

    @property
    def tag(self) -> str:
        parts = self.info.strip().split()
        return parts[0] if parts else ""

    @property
    def has_leading_space(self) -> bool:
        return bool(self.info) and self.info[0].isspace() and bool(self.info.strip())

    @property
    def has_trailing_space(self) -> bool:
        return self.info != self.info.rstrip()

    @property
    def is_tilde(self) -> bool:
        return self.marker.startswith("~")

    @property
    def is_overlong(self) -> bool:
        return not self.is_tilde and len(self.marker) > 3

    @property
    def location(self) -> str:
        return f"{self.path}:{self.lineno}"


def _scan_fences(path: Path) -> tuple[list[Fence], Fence | None]:
    """Return the opening fences of a markdown file and the one left unclosed, if any."""
    # CommonMark closing rules; mktestdocs instead toggles on any line containing three
    # backticks, so a doc showing a fence inside a fence would desynchronize the two.
    openings: list[Fence] = []
    open_fence: Fence | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE_OPENING.match(line.lstrip())
        if match is None:
            continue
        marker, info = match.group("marker"), match.group("info")
        if open_fence is not None:
            closes = marker[0] == open_fence.marker[0] and len(marker) >= len(open_fence.marker)
            if closes and not info.strip():
                open_fence = None
            continue
        open_fence = Fence(path=path, lineno=lineno, text=line.lstrip(), marker=marker, info=info)
        openings.append(open_fence)
    return openings, open_fence


def _iter_fence_openings(path: Path) -> Iterator[Fence]:
    """Yield the opening fences of a markdown file, skipping fence bodies."""
    openings, _ = _scan_fences(path)
    return iter(openings)


def _doc_files() -> list[Path]:
    files = sorted(DOCS_ROOT.rglob("*.md"))
    assert files, f"no markdown files under {DOCS_ROOT}, the fence checks would pass vacuously"
    return files


def _suggested_tag(tag: str) -> str:
    if tag.lower() == "pycon":
        return OUTPUT_TAG
    return RUNNABLE_TAG


def _violations_in_file(path: Path) -> list[str]:
    """Report the fence spellings of one markdown file that never reach the doc runner."""
    messages: list[str] = []
    openings, unterminated = _scan_fences(path)
    for fence in openings:
        tag = fence.tag
        python_looking = bool(PYTHON_LOOKING.fullmatch(tag))
        if fence.has_leading_space:
            unspaced = f"```{fence.info.strip()}"
            messages.append(f"{fence.location}: spaced fence {fence.text!r} -> write {unspaced!r}. {VOCABULARY_HINT}")
            continue
        if tag and fence.info.strip() != tag:
            messages.append(
                f"{fence.location}: fence {fence.text!r} carries attributes after the tag, so it does not "
                f"render as a code block -> write a bare ```{tag} with the title in a comment. {VOCABULARY_HINT}"
            )
            continue
        if not python_looking:
            continue
        if fence.is_tilde:
            messages.append(
                f"{fence.location}: fence {fence.text!r} highlights as Python but is never executed, "
                f"only a backtick fence is collected -> write ```{_suggested_tag(tag)}. {VOCABULARY_HINT}"
            )
            continue
        if fence.is_overlong:
            messages.append(
                f"{fence.location}: fence {fence.text!r} looks like Python but is never executed, "
                f"only a three-backtick fence is executed -> write ```{_suggested_tag(tag)}. {VOCABULARY_HINT}"
            )
            continue
        if fence.has_trailing_space:
            messages.append(
                f"{fence.location}: fence {fence.text!r} has trailing whitespace after the tag and is never "
                f"executed -> write ```{tag} without it. {VOCABULARY_HINT}"
            )
            continue
        if tag in {RUNNABLE_TAG, ILLUSTRATIVE_TAG}:
            continue
        messages.append(
            f"{fence.location}: fence {fence.text!r} looks like Python but is never executed, "
            f"write ```{_suggested_tag(tag)}. {VOCABULARY_HINT}"
        )
    if unterminated is not None:
        messages.append(
            f"{unterminated.location}: fence {unterminated.text!r} is never closed, so nothing inside it is "
            f"collected -> close it. {VOCABULARY_HINT}"
        )
    return messages


def _fence_violations() -> list[str]:
    return [message for path in _doc_files() for message in _violations_in_file(path)]


def _stale_allowlist_keys(allowlist: Mapping[str, int]) -> list[str]:
    """Report the allowlist keys that no longer name an existing doc file."""
    return [key for key in sorted(allowlist) if not (REPO_ROOT / key).exists()]


def _format(violations: list[str]) -> str:
    shown = violations[:MAX_REPORTED_VIOLATIONS]
    lines = list(shown)
    omitted = len(violations) - len(shown)
    if omitted:
        lines.append(f"... and {omitted} more omitted.")
    return "\n".join(lines)


def test_doc_fences_use_the_documented_vocabulary() -> None:
    violations = _fence_violations()
    assert not violations, f"{len(violations)} doc fence violation(s):\n{_format(violations)}"


def test_illustrative_py_blocks_match_allowlist() -> None:
    errors: list[str] = []
    for path in _doc_files():
        actual = sum(1 for fence in _iter_fence_openings(path) if fence.info.strip() == ILLUSTRATIVE_TAG)
        allowed = ILLUSTRATIVE_BLOCK_ALLOWLIST.get(str(path.relative_to(REPO_ROOT)), 0)
        if actual > allowed:
            errors.append(
                f"{path}: {actual} ```{ILLUSTRATIVE_TAG} block(s) but only {allowed} allowed. "
                f"Promote the block to ```{RUNNABLE_TAG} (or ```{OUTPUT_TAG} if it is output), "
                f"or raise its ILLUSTRATIVE_BLOCK_ALLOWLIST entry."
            )
        elif actual < allowed:
            errors.append(
                f"{path}: {actual} ```{ILLUSTRATIVE_TAG} block(s) but {allowed} allowed. "
                f"Lower the ILLUSTRATIVE_BLOCK_ALLOWLIST entry to {actual}."
            )
    assert not errors, "```py allowlist drift:\n" + "\n".join(errors)


# Helpers the guard is expected to expose so it can be driven over synthetic input.
VIOLATIONS_HELPER = "_violations_in_file"
STALE_KEYS_HELPER = "_stale_allowlist_keys"


def _violations_for(path: Path) -> list[str]:
    """Report the fence violations of a single markdown file."""
    helper = globals().get(VIOLATIONS_HELPER)
    assert callable(helper), (
        f"{__name__} must expose {VIOLATIONS_HELPER}(path) -> list[str] so the fence scanner "
        f"can be exercised on synthetic markdown instead of only on the whole docs tree"
    )
    return cast(Callable[[Path], list[str]], helper)(path)


def _stale_keys(allowlist: Mapping[str, int]) -> list[str]:
    """Report the allowlist keys that no longer name an existing doc file."""
    helper = globals().get(STALE_KEYS_HELPER)
    assert callable(helper), (
        f"{__name__} must expose {STALE_KEYS_HELPER}(allowlist) -> list[str] so a key whose file "
        f"was deleted or renamed is reported instead of rotting unvisited"
    )
    return cast(Callable[[Mapping[str, int]], list[str]], helper)(allowlist)


SNIPPET = 'print("collected")'

TRAILING_SPACE_OPEN = "```python "  # one trailing space, which the guard's strip() throws away
FOUR_BACKTICK_OPEN = "````python"
TILDE_OPEN = "~~~python"


def _markdown(opening: str, closing: str | None) -> str:
    """Build a one-block markdown document, leaving the block open when closing is None."""
    lines = ["# Synthetic", "", opening, SNIPPET]
    if closing is not None:
        lines.append(closing)
    return "\n".join(lines) + "\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestUncollectedPythonFenceSpellings:
    """Fence spellings that read as runnable Python but that mktestdocs never collects."""

    def test_a_plain_python_fence_is_collected(self) -> None:
        """Control: the ground-truth assertions of this class are not vacuous."""
        assert grab_code_blocks(_markdown("```python", "```"), lang="python") == [f"{SNIPPET}\n"]

    def test_a_trailing_space_after_the_tag_is_reported(self, tmp_path: Path) -> None:
        text = _markdown(TRAILING_SPACE_OPEN, "```")
        assert grab_code_blocks(text, lang="python") == [], (
            f"ground truth changed: mktestdocs now collects {TRAILING_SPACE_OPEN!r}"
        )
        path = _write(tmp_path, "trailing_space.md", text)
        violations = _violations_for(path)
        assert violations, (
            f"{TRAILING_SPACE_OPEN!r} looks runnable and is never executed, "
            f"because mktestdocs compares the info string against 'python' without rstrip"
        )

    def test_a_four_backtick_fence_is_reported(self, tmp_path: Path) -> None:
        text = _markdown(FOUR_BACKTICK_OPEN, "````")
        assert grab_code_blocks(text, lang="python") == [], (
            f"ground truth changed: mktestdocs now collects {FOUR_BACKTICK_OPEN!r}"
        )
        path = _write(tmp_path, "four_backticks.md", text)
        violations = _violations_for(path)
        assert violations, (
            f"{FOUR_BACKTICK_OPEN!r} looks runnable and is never executed, "
            f"because mktestdocs slices exactly three backticks and reads the tag as '`python'"
        )

    def test_a_tilde_fence_is_reported(self, tmp_path: Path) -> None:
        text = _markdown(TILDE_OPEN, "~~~")
        assert grab_code_blocks(text, lang="python") == [], (
            f"ground truth changed: mktestdocs now collects {TILDE_OPEN!r}"
        )
        path = _write(tmp_path, "tilde.md", text)
        violations = _violations_for(path)
        assert violations, (
            f"{TILDE_OPEN!r} highlights as Python and is never executed, "
            f"because mktestdocs only recognises backtick fences"
        )

    def test_an_unterminated_python_fence_is_reported(self, tmp_path: Path) -> None:
        text = _markdown("```python", None)
        assert grab_code_blocks(text, lang="python") == [], (
            "ground truth changed: mktestdocs now collects a block with no closing fence"
        )
        path = _write(tmp_path, "unterminated.md", text)
        violations = _violations_for(path)
        assert violations, (
            "an unclosed ```python block at end of file is never executed, "
            "because mktestdocs only emits a block once it sees the closing fence"
        )

    @pytest.mark.parametrize("tag", ["ipython", "python-repl"])
    def test_a_python_highlighting_alias_is_reported(self, tmp_path: Path, tag: str) -> None:
        text = _markdown(f"```{tag}", "```")
        assert grab_code_blocks(text, lang="python") == [], f"ground truth changed: mktestdocs now collects ```{tag}"
        path = _write(tmp_path, f"{tag.replace('-', '_')}.md", text)
        violations = _violations_for(path)
        assert violations, f"```{tag} highlights as Python and is never executed, yet PYTHON_LOOKING misses it"


class TestAllowlistKeysStayCurrent:
    """An allowlist key whose file is gone is never visited by the file loop."""

    def test_every_allowlist_key_names_an_existing_doc(self) -> None:
        missing = _stale_keys(ILLUSTRATIVE_BLOCK_ALLOWLIST)
        assert not missing, (
            f"ILLUSTRATIVE_BLOCK_ALLOWLIST key(s) name files that do not exist: {missing}. "
            f"Drop the entry, or fix the path if the doc was renamed."
        )

    def test_the_helper_reports_a_key_whose_file_is_gone(self) -> None:
        stale = _stale_keys({**ILLUSTRATIVE_BLOCK_ALLOWLIST, "docs/docs/deleted-page.md": 1})
        assert stale == ["docs/docs/deleted-page.md"], f"a deleted doc stayed unreported, got {stale}"


class TestDocCorpusIsNotCwdDependent:
    """A CWD-relative DOCS_ROOT lets both static checks pass over an empty corpus."""

    def test_doc_files_are_found_from_any_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from_repo_root = _doc_files()
        assert from_repo_root, "the doc corpus is empty even from the repo root"

        monkeypatch.chdir(tmp_path)
        from_elsewhere = _doc_files()
        assert from_elsewhere, (
            "DOCS_ROOT resolves against the process CWD, so from any other directory the fence "
            "vocabulary and allowlist checks pass over an empty corpus"
        )
        assert [path.name for path in from_elsewhere] == [path.name for path in from_repo_root]
        assert all(path.exists() for path in from_elsewhere)
