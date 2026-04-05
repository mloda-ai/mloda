import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
