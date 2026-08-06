"""Published docs quoting a near-miss label sit outside the doc-test guard, which runs ```python fences only.

Renaming a ``_STAGE_LABELS`` value reddens the renderer's hardcoded label table, which names no doc page, so
updating that table turns the suite green with the pages still quoting the old label. These guards read the
label back off the page, check it against the live table, and name the file and line that went stale.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.resolution_failure_renderer import _STAGE_LABELS, render_resolution_failure
from mloda.core.prepare.resolution_types import Elimination, EliminationStage, EvaluationResult


# Anchored to this file, not to the cwd: a relative glob run from anywhere but
# the repo root yields nothing and these guards pass vacuously (issue #937).
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs" / "docs"
FENCE_PATTERN = re.compile(r"^\s*```")


def _docs_pages() -> list[Path]:
    """Every published page, never an empty list.

    ``parametrize`` runs at import time, where an empty list collapses to zero
    cases instead of failing, so an empty glob must raise rather than return.
    """
    pages = sorted(DOCS_ROOT.rglob("*.md"))
    if not pages:
        raise RuntimeError(f"no documentation pages found under {DOCS_ROOT} - this guard would check nothing")
    return pages


# Derived, never restated: a renamed label moves this set with it, and a label no page quotes costs nothing.
ALLOWED_LABELS: frozenset[str] = frozenset(_STAGE_LABELS.values())

# The near-miss bullet '  - <ClassName> (<label>): <reason>'. The '): ' excludes the multiple-candidates
# render ('  - FG (module.path)'); the capital starts fg.__name__ and keeps fenced 'Args:' lines out.
NEAR_MISS_BULLET_PATTERN = re.compile(r"^  - (?P<candidate>[A-Z]\w*) \((?P<label>[^)]+)\): (?P<reason>.+)$")

# FeatureGroup is the candidate itself: the renderer reads only __name__/__module__, so no subclass leaks.
SCANNER_FEATURE = "near_miss_scanner_probe_feature"
SCANNER_STAGE: EliminationStage = "framework_pin"
SCANNER_REASON = "eliminated so that one near-miss line renders"


def _fenced_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield (1-based line number, line) for every line inside a ``` fenced block."""
    inside = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE_PATTERN.match(line):
            inside = not inside
            continue
        if inside:
            yield number, line


def _near_miss_bullets(text: str) -> Iterator[tuple[int, re.Match[str]]]:
    """Yield (1-based line number, match) for every near-miss bullet inside a ``` fenced block."""
    for number, line in _fenced_lines(text):
        match = NEAR_MISS_BULLET_PATTERN.match(line)
        if match is not None:
            yield number, match


@pytest.mark.parametrize("fpath", _docs_pages(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_rendered_near_miss_bullets_carry_a_current_stage_label(fpath: Path) -> None:
    """Every label a page reproduces in a near-miss bullet is still a value of ``_STAGE_LABELS``."""
    stale = [
        f"{fpath}:{number} renders near-miss label '{match.group('label')}'"
        for number, match in _near_miss_bullets(fpath.read_text(encoding="utf-8"))
        if match.group("label") not in ALLOWED_LABELS
    ]

    assert not stale, (
        f"Doc page(s) render a near-miss label no stage carries (current labels: {sorted(ALLOWED_LABELS)}):\n"
        + "\n".join(stale)
    )


def test_the_scanner_pattern_matches_a_rendered_near_miss_bullet() -> None:
    """A changed bullet format would otherwise degrade the scan above into a no-op."""
    elimination = Elimination(stage=SCANNER_STAGE, reason=SCANNER_REASON)
    result = EvaluationResult(identified={}, eliminations={FeatureGroup: elimination})

    # Premise: this failure reaches the near-miss block, the only place a stage label renders.
    assert result.failure_kind == "none"

    message = render_resolution_failure(result, Feature(SCANNER_FEATURE))
    assert message is not None

    # Located by its reason, never by the prefix under test, so a changed prefix fails the match below.
    rendered = [line for line in message.splitlines() if SCANNER_REASON in line]
    assert len(rendered) == 1, f"Expected exactly one rendered near-miss line, got {rendered}"

    match = NEAR_MISS_BULLET_PATTERN.match(rendered[0])
    assert match is not None, f"NEAR_MISS_BULLET_PATTERN no longer matches {rendered[0]!r}, so the scan is a no-op."
    assert match.group("candidate") == FeatureGroup.__name__
    assert match.group("label") == _STAGE_LABELS[SCANNER_STAGE]
    assert match.group("reason") == SCANNER_REASON


# The troubleshooting page carries the full label table; the table guard below and DOC_BULLET_STAGES share it.
LABEL_TABLE_PAGE = "in_depth/troubleshooting/feature-group-resolution-errors.md"

# Pages reproducing a rendered near-miss bullet, pinned to their stage. This is the floor under the scan
# above, which passes when it finds nothing, and the stage pin its set-membership check cannot be.
DOC_BULLET_STAGES: dict[str, EliminationStage] = {
    "in_depth/compute-framework-integration.md": "framework_pin",
    "in_depth/data-access-patterns.md": "input_data",
    "in_depth/property-mapping.md": "value_rejection",
    LABEL_TABLE_PAGE: "domain",
}


@pytest.mark.parametrize(("relative_path", "stage"), sorted(DOC_BULLET_STAGES.items()), ids=sorted(DOC_BULLET_STAGES))
def test_a_registered_page_still_renders_the_bullet_of_its_own_stage(
    relative_path: str, stage: EliminationStage
) -> None:
    page = DOCS_ROOT / relative_path
    assert page.is_file(), f"{page} was moved or renamed; update DOC_BULLET_STAGES"

    labels = {match.group("label") for _, match in _near_miss_bullets(page.read_text(encoding="utf-8"))}

    # Containment, not equality: a page may grow a second example, and unknown labels fail the scan above.
    assert _STAGE_LABELS[stage] in labels, (
        f"{page} no longer renders a '{_STAGE_LABELS[stage]}' bullet the scanner reads (found: {sorted(labels)})."
    )


# data-access-patterns.md is in this table AND in DOC_BULLET_STAGES on purpose: its prose mention sits far
# from its rendered bullet, so the bullet's failure names a line that leaves the prose mention stale.
PROSE_LABEL_PAGES: tuple[tuple[str, EliminationStage], ...] = (
    ("in_depth/data-access-patterns.md", "input_data"),
    ("in_depth/feature-chain-parser.md", "matcher_error"),
    ("in_depth/feature-group-matching.md", "matcher_error"),
)


class TestProseLabelMentions:
    """Pages naming a label in prose, which the bullet scanner never reads."""

    @pytest.mark.parametrize(
        ("relative_path", "stage"),
        PROSE_LABEL_PAGES,
        ids=[f"{relative_path}-{stage}" for relative_path, stage in PROSE_LABEL_PAGES],
    )
    def test_the_page_still_spells_the_current_label(self, relative_path: str, stage: EliminationStage) -> None:
        page = DOCS_ROOT / relative_path
        assert page.is_file(), f"{page} was moved or renamed; update PROSE_LABEL_PAGES"

        # The bare label, or the parenthesized form the bullet renders.
        label = _STAGE_LABELS[stage]
        spellings = (f"`{label}`", f"`({label})`")
        text = page.read_text(encoding="utf-8")
        assert any(spelling in text for spelling in spellings), (
            f"{page} no longer names the '{stage}' near-miss label in prose as any of {list(spellings)}."
        )


# A markdown table row whose first column is a backticked value; only the label table backticks its first column.
LABEL_TABLE_ROW_PATTERN = re.compile(r"^\|\s*`(?P<label>[^`]+)`\s*\|")


def test_the_label_table_rows_are_exactly_the_distinct_stage_labels() -> None:
    """Equality both ways: a deleted or renamed row fails, and so does a row naming a label no stage carries."""
    page = DOCS_ROOT / LABEL_TABLE_PAGE
    assert page.is_file(), f"{page} was moved or renamed; update LABEL_TABLE_PAGE"

    rows = {
        match.group("label")
        for line in page.read_text(encoding="utf-8").splitlines()
        if (match := LABEL_TABLE_ROW_PATTERN.match(line)) is not None
    }

    missing = ALLOWED_LABELS - rows
    assert not missing, f"{page} label table lost the row(s) for label(s) {sorted(missing)}."

    stale = rows - ALLOWED_LABELS
    assert not stale, f"{page} label table names label(s) no stage carries: {sorted(stale)}."
