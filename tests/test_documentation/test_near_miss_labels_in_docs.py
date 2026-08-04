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
NEAR_MISS_BULLET_PATTERN = re.compile(r"^  - (?P<candidate>\w+) \((?P<label>[^)]+)\): (?P<reason>.+)$")

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


@pytest.mark.parametrize("fpath", sorted(DOCS_ROOT.rglob("*.md")), ids=str)
def test_rendered_near_miss_bullets_carry_a_current_stage_label(fpath: Path) -> None:
    """Every label a doc page reproduces inside a near-miss bullet is still a value of ``_STAGE_LABELS``."""
    stale: list[str] = []
    for number, line in _fenced_lines(fpath.read_text(encoding="utf-8")):
        match = NEAR_MISS_BULLET_PATTERN.match(line)
        if match is not None and match.group("label") not in ALLOWED_LABELS:
            stale.append(f"{fpath}:{number} renders near-miss label '{match.group('label')}'")

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


# Pages that name a label in PROSE only. The three pages reproducing a rendered bullet
# (in_depth/compute-framework-integration.md, in_depth/property-mapping.md, in_depth/data-access-patterns.md)
# are deliberately absent: the bullet scanner above already reads their label off the page and checks it
# against the live table, so listing them here would pin the same text twice and freeze which page shows
# which stage.
PROSE_LABEL_PAGES: dict[str, EliminationStage] = {
    "in_depth/feature-chain-parser.md": "matcher_error",
    "in_depth/feature-group-matching.md": "matcher_error",
}


class TestProseOnlyLabelMentions:
    """Pages naming a label in prose only; the bullet pages stay out because guard 1 reads their label."""

    @pytest.mark.parametrize(
        ("relative_path", "stage"),
        sorted(PROSE_LABEL_PAGES.items()),
        ids=sorted(PROSE_LABEL_PAGES),
    )
    def test_the_page_still_spells_the_current_label(self, relative_path: str, stage: EliminationStage) -> None:
        """A page carrying no rendered bullet is caught by nothing else when its label is renamed."""
        page = DOCS_ROOT / relative_path
        assert page.is_file(), f"{page} was moved or renamed; update PROSE_LABEL_PAGES"

        label = _STAGE_LABELS[stage]
        assert f"`{label}`" in page.read_text(encoding="utf-8"), (
            f"{page} no longer mentions the '{stage}' near-miss label as `{label}`. It reproduces no "
            "rendered near-miss line, so this table is the only guard on it."
        )
