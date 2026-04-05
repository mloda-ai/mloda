import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestContributingFile:
    """Validate that CONTRIBUTING.md has substantive content for contributors."""

    def test_contributing_file_exists(self) -> None:
        assert (PROJECT_ROOT / "CONTRIBUTING.md").is_file(), "CONTRIBUTING.md must exist at project root"

    def test_contributing_has_dev_setup_section(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        content_lower = content.lower()
        assert "development" in content_lower or "setup" in content_lower or "getting started" in content_lower, (
            "CONTRIBUTING.md must include a development setup section"
        )

    def test_contributing_mentions_uv(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "uv" in content, "CONTRIBUTING.md must mention uv as the dependency manager"

    def test_contributing_mentions_tox(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "tox" in content, "CONTRIBUTING.md must mention tox as the test runner"

    def test_contributing_mentions_python_version(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "3.10" in content, "CONTRIBUTING.md must specify the minimum supported Python version (3.10)"

    def test_contributing_has_code_style_section(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        content_lower = content.lower()
        assert "code style" in content_lower or "coding standard" in content_lower or "style" in content_lower, (
            "CONTRIBUTING.md must include a code style section"
        )

    def test_contributing_mentions_ruff(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "ruff" in content, "CONTRIBUTING.md must mention ruff as the linter/formatter"

    def test_contributing_mentions_mypy(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "mypy" in content, "CONTRIBUTING.md must mention mypy for type checking"

    def test_contributing_has_pr_workflow_section(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        content_lower = content.lower()
        assert "pull request" in content_lower or "pr" in content_lower, (
            "CONTRIBUTING.md must describe the pull request workflow"
        )

    def test_contributing_mentions_license(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "Apache" in content, "CONTRIBUTING.md must mention the Apache 2.0 license"

    def test_contributing_has_plugin_development_section(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        content_lower = content.lower()
        assert "plugin" in content_lower, "CONTRIBUTING.md must cover plugin development"

    def test_contributing_mentions_registry_guides(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "mloda-registry" in content, "CONTRIBUTING.md must reference the mloda-registry guides"
        assert "guides" in content.lower(), "CONTRIBUTING.md must mention the plugin development guides"

    def test_contributing_mentions_plugin_template(self) -> None:
        content = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()
        assert "mloda-plugin-template" in content, "CONTRIBUTING.md must reference the mloda-plugin-template"


class TestLicenseFile:
    def test_license_file_exists(self) -> None:
        assert (PROJECT_ROOT / "LICENSE").is_file(), "LICENSE file must exist at project root"

    def test_no_uppercase_license_extension(self) -> None:
        assert not (PROJECT_ROOT / "LICENSE.TXT").exists(), "LICENSE.TXT should not exist; use LICENSE instead"

    def test_license_is_apache2(self) -> None:
        content = (PROJECT_ROOT / "LICENSE").read_text()
        assert "Apache License, Version 2.0" in content


def _load_extras() -> dict[str, list[str]]:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    extras: dict[str, list[str]] = data["project"]["optional-dependencies"]
    return extras


def _normalize_extra_name(name: str) -> str:
    return name.replace("_", "-")


class TestExtrasConsistency:
    INTENTIONALLY_EXCLUDED_FROM_ALL = {"all", "spark"}

    def test_all_extras_uses_self_references(self) -> None:
        extras = _load_extras()
        for entry in extras["all"]:
            assert entry.startswith("mloda[") and entry.endswith("]"), (
                f"[all] must only contain self-references (mloda[...]), found raw dependency: {entry}"
            )

    def test_all_extras_covers_every_group(self) -> None:
        extras = _load_extras()
        referenced = {entry.split("[")[1].rstrip("]") for entry in extras["all"]}
        expected = set(extras.keys()) - self.INTENTIONALLY_EXCLUDED_FROM_ALL

        referenced_normalized = {_normalize_extra_name(r) for r in referenced}
        expected_normalized = {_normalize_extra_name(g) for g in expected}

        missing = expected_normalized - referenced_normalized
        assert not missing, f"[all] is missing references to extras groups: {missing}"

    def test_scikit_learn_version_consistent(self) -> None:
        extras = _load_extras()
        sklearn_specs: dict[str, str] = {}
        for group_name, deps in extras.items():
            for dep in deps:
                if dep.startswith("mloda["):
                    continue
                req = Requirement(dep)
                if req.name == "scikit-learn":
                    sklearn_specs[group_name] = str(req.specifier)

        unique_specs = set(sklearn_specs.values())
        assert len(unique_specs) == 1, (
            f"scikit-learn has inconsistent version constraints across extras groups: {sklearn_specs}"
        )
