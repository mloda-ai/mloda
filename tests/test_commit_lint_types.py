"""Guards against drift between .releaserc.yaml's commit-analyzer releaseRules and the commit
type alternation embedded in the commit-lint workflow's regex pattern."""

import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASERC_YAML = PROJECT_ROOT / ".releaserc.yaml"
COMMIT_LINT_YAML = PROJECT_ROOT / ".github" / "workflows" / "commit-lint.yaml"

# revert is release-triggering via commit-analyzer's built-in default rules, not an explicit
# entry in .releaserc.yaml's releaseRules, so the workflow deliberately lists it while
# .releaserc.yaml deliberately does not.
DOCUMENTED_WORKFLOW_ONLY_TYPES = {"revert"}

# Matches the `(Revert "|(TYPES)(\([^)]+\))?!?: ).+$` fragment of the pattern= line embedded in
# the workflow's bash run: script and captures the pipe-separated TYPES alternation.
TYPE_ALTERNATION = re.compile(r'\(Revert "\|\((?P<types>[a-z|]+)\)\(\\\(\[\^\)\]\+\\\)\)\?!\?: \)\.\+\$')


def _releaserc_release_rule_types() -> set[str]:
    config: dict[str, Any] = yaml.safe_load(RELEASERC_YAML.read_text(encoding="utf-8"))
    plugins: list[Any] = config["plugins"]
    commit_analyzer_entry = plugins[0]
    assert commit_analyzer_entry[0] == "@semantic-release/commit-analyzer", (
        "commit-analyzer must be the first entry in .releaserc.yaml's plugins list; this test reads "
        f"plugins[0] directly. Found: {commit_analyzer_entry[0]!r}"
    )
    release_rules: list[dict[str, Any]] = commit_analyzer_entry[1]["releaseRules"]
    return {rule["type"] for rule in release_rules}


def _workflow_pattern_line() -> str:
    lines = [line for line in COMMIT_LINT_YAML.read_text(encoding="utf-8").splitlines() if "pattern=" in line]
    assert lines, f"no 'pattern=' line found in {COMMIT_LINT_YAML}; the commit-lint workflow structure changed"
    return lines[0]


def _workflow_alternation_types() -> set[str]:
    line = _workflow_pattern_line()
    match = TYPE_ALTERNATION.search(line)
    assert match is not None, (
        f"could not find the '(Revert \"|(...))' type alternation in the pattern line: {line!r}. "
        "The regex shape in commit-lint.yaml changed; update TYPE_ALTERNATION in this test to match it."
    )
    return set(match.group("types").split("|"))


def test_releaserc_types_are_all_recognized_by_workflow() -> None:
    releaserc_types = _releaserc_release_rule_types()
    workflow_types = _workflow_alternation_types()
    missing = releaserc_types - workflow_types
    assert not missing, (
        f"{sorted(missing)} appear in .releaserc.yaml's releaseRules but not in the commit-lint workflow's "
        "type alternation. commit-analyzer treats these as release-triggering, so commit-lint.yaml would "
        "reject a valid, release-triggering commit type. Add the missing type(s) to the pattern= "
        "alternation in .github/workflows/commit-lint.yaml."
    )


def test_workflow_types_match_releaserc_plus_documented_revert_exception() -> None:
    releaserc_types = _releaserc_release_rule_types()
    workflow_types = _workflow_alternation_types()
    assert workflow_types - DOCUMENTED_WORKFLOW_ONLY_TYPES == releaserc_types, (
        f"workflow alternation types {sorted(workflow_types)} minus the documented exception "
        f"{sorted(DOCUMENTED_WORKFLOW_ONLY_TYPES)} must equal .releaserc.yaml's releaseRules types "
        f"{sorted(releaserc_types)}. 'revert' is the only allowed extra: commit-analyzer's built-in "
        "default rules make it release-triggering without an explicit releaseRules entry. Any other "
        "extra type in the workflow is likely a typo that silently widens the commit-lint gate."
    )
