"""Published docs quoting a near-miss label sit outside the doc-test guard, which runs ```python fences only.

A page reproducing a rendered near-miss bullet, or naming a label in prose, is never executed, so renaming an
entry of ``_STAGE_LABELS`` leaves the suite green and the docs wrong. These guards read the label back off the
page and check it against the live table.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.resolution_failure_renderer import _STAGE_LABELS, render_resolution_failure
from mloda.core.prepare.resolution_types import Elimination, EliminationStage, EvaluationResult


DOCS_ROOT = Path("docs/docs")

# Derived from the production table, never restated: a renamed label moves this set with it, and a label no
# page happens to quote costs nothing here.
ALLOWED_LABELS: frozenset[str] = frozenset(_STAGE_LABELS.values())

FENCE_PATTERN = re.compile(r"^\s*```")

# The near-miss bullet of a resolution failure: '  - <ClassName> (<label>): <reason>'. The '): ' is
# load-bearing. The multiple-candidates render is a deliberate near-collision: it is '  - FG (module.path)',
# optionally with ' [domain: ...]', and carries no ': ' straight after the paren, so requiring '): ' keeps
# those lines (docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md) out of the scan.
# The candidate is capitalized because the renderer emits fg.__name__: a bare \w+ also matches a fenced
# 'Args:' parameter line ('  - event_from (datetime): Start of the range'), which would fail this module
# on a page that has nothing to do with resolution.
NEAR_MISS_BULLET_PATTERN = re.compile(r"^  - (?P<candidate>[A-Z]\w*) \((?P<label>[^)]+)\): (?P<reason>.+)$")

# Guard 2's probe. FeatureGroup itself is the eliminated candidate: the renderer only reads __name__ and
# __module__ off it, so no local subclass is declared and nothing leaks into FeatureGroup.__subclasses__().
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


@pytest.mark.parametrize("fpath", sorted(DOCS_ROOT.rglob("*.md")), ids=str)
def test_rendered_near_miss_bullets_carry_a_current_stage_label(fpath: Path) -> None:
    """Every label a doc page reproduces inside a near-miss bullet is still a value of ``_STAGE_LABELS``."""
    stale = [
        f"{fpath}:{number} renders near-miss label '{match.group('label')}'"
        for number, match in _near_miss_bullets(fpath.read_text(encoding="utf-8"))
        if match.group("label") not in ALLOWED_LABELS
    ]

    assert not stale, (
        "Doc page(s) render a near-miss label that no elimination stage carries today "
        f"(current labels: {sorted(ALLOWED_LABELS)}):\n" + "\n".join(stale)
    )


def test_the_scanner_pattern_matches_a_rendered_near_miss_bullet() -> None:
    """Pin the scanner to the renderer, so a changed bullet format cannot degrade the scan into a no-op."""
    result = EvaluationResult(
        identified={},
        eliminations={FeatureGroup: Elimination(stage=SCANNER_STAGE, reason=SCANNER_REASON)},
    )

    # Premise: this failure reaches the near-miss block, the only place a stage label renders.
    assert result.failure_kind == "none"

    message = render_resolution_failure(result, Feature(SCANNER_FEATURE))
    assert message is not None

    # Located by its reason, never by the prefix under test, so a changed prefix fails the match below.
    rendered = [line for line in message.splitlines() if SCANNER_REASON in line]
    assert len(rendered) == 1, f"Expected exactly one rendered near-miss line, got {rendered}"

    match = NEAR_MISS_BULLET_PATTERN.match(rendered[0])
    assert match is not None, (
        f"NEAR_MISS_BULLET_PATTERN no longer matches the rendered near-miss bullet {rendered[0]!r}, "
        "so the doc scanner above silently reads nothing."
    )
    assert match.group("candidate") == FeatureGroup.__name__
    assert match.group("label") == _STAGE_LABELS[SCANNER_STAGE]
    assert match.group("reason") == SCANNER_REASON


# Pages reproducing a rendered near-miss bullet, pinned to the stage they render. This table does two jobs
# the tree-wide scan cannot. It is the FLOOR: guard 1 passes green when it finds nothing, so an editorial
# change that hides the fence from the toggle (indenting it into an ``!!! note``, moving it under a numbered
# step, switching to ``~~~``, blockquoting it, nesting it in a 4-backtick block, or any earlier odd ``` count
# on the page) fails here instead of scanning nothing. It is also the STAGE PIN: guard 1 tests set
# membership, so it sees neither two labels swapped nor a rename of just one of the two stages sharing the
# "compute framework" label. The tree-wide scan stays: it is what covers pages nobody registered.
DOC_BULLET_STAGES: dict[str, EliminationStage] = {
    "in_depth/compute-framework-integration.md": "framework_pin",
    "in_depth/data-access-patterns.md": "input_data",
    "in_depth/property-mapping.md": "value_rejection",
}


@pytest.mark.parametrize(
    ("relative_path", "stage"),
    sorted(DOC_BULLET_STAGES.items()),
    ids=sorted(DOC_BULLET_STAGES),
)
def test_a_registered_page_still_renders_the_bullet_of_its_own_stage(
    relative_path: str, stage: EliminationStage
) -> None:
    """The page still carries a bullet the scanner can read, and it is still this stage's bullet."""
    page = DOCS_ROOT / relative_path
    assert page.is_file(), f"{page} was moved or renamed; update DOC_BULLET_STAGES"

    labels = {match.group("label") for _, match in _near_miss_bullets(page.read_text(encoding="utf-8"))}

    # Containment, not equality: a page that later grows a second near-miss example must not be forced into
    # this table, and guard 1 already rejects any label it carries that no stage renders.
    assert _STAGE_LABELS[stage] in labels, (
        f"{page} no longer renders a '{_STAGE_LABELS[stage]}' near-miss bullet the scanner can read "
        f"(labels found on the page: {sorted(labels)}). Either the '{stage}' label was renamed, or the "
        "fence stopped matching and the tree-wide scan now reads this page for nothing."
    )


# Every place a page names a label in PROSE, whether or not the same page also renders a bullet. A pair
# per mention, not a mapping, so a page naming two labels in prose stays expressible.
# in_depth/data-access-patterns.md is in this table AND in DOC_BULLET_STAGES on purpose: it names the label
# a second time in prose (line 102), far from its rendered bullet (line 138). A rename is detected by the
# bullet, but that failure names the bullet's line only, which leaves the prose mention stale. The other two
# bullet pages stay out because they name no label outside their fence.
PROSE_LABEL_PAGES: tuple[tuple[str, EliminationStage], ...] = (
    ("in_depth/data-access-patterns.md", "input_data"),
    ("in_depth/feature-chain-parser.md", "matcher_error"),
    ("in_depth/feature-group-matching.md", "matcher_error"),
)


def _accepted_prose_spellings(label: str) -> tuple[str, str]:
    """The two spellings prose may use: the bare label, or the parenthesized form the bullet renders."""
    return (f"`{label}`", f"`({label})`")


class TestProseLabelMentions:
    """Pages naming a label in prose, which the bullet scanner never reads."""

    @pytest.mark.parametrize(
        ("relative_path", "stage"),
        PROSE_LABEL_PAGES,
        ids=[f"{relative_path}-{stage}" for relative_path, stage in PROSE_LABEL_PAGES],
    )
    def test_the_page_still_spells_the_current_label(self, relative_path: str, stage: EliminationStage) -> None:
        """A prose mention is caught by nothing else when its label is renamed."""
        page = DOCS_ROOT / relative_path
        assert page.is_file(), f"{page} was moved or renamed; update PROSE_LABEL_PAGES"

        spellings = _accepted_prose_spellings(_STAGE_LABELS[stage])
        text = page.read_text(encoding="utf-8")
        assert any(spelling in text for spelling in spellings), (
            f"{page} no longer names the '{stage}' near-miss label in prose as any of {list(spellings)}. "
            "No rendered bullet covers this mention, so a rename leaves the sentence stale."
        )
