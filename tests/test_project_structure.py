import configparser
import fnmatch
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestContributingFile:
    """Validate that CONTRIBUTING.md has substantive content for contributors."""

    def test_contributing_file_exists(self) -> None:
        assert (PROJECT_ROOT / "CONTRIBUTING.md").is_file(), "CONTRIBUTING.md must exist at project root"

    def test_contributing_has_dev_setup_section(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        content_lower = content.lower()
        assert "development" in content_lower or "setup" in content_lower or "getting started" in content_lower, (
            "CONTRIBUTING.md must include a development setup section"
        )

    def test_contributing_mentions_uv(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "uv" in content, "CONTRIBUTING.md must mention uv as the dependency manager"

    def test_contributing_mentions_tox(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "tox" in content, "CONTRIBUTING.md must mention tox as the test runner"

    def test_contributing_mentions_python_version(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "Python 3.11 or higher" in content, "CONTRIBUTING.md must state the minimum Python version (3.11)"
        assert "Python 3.10" not in content and "3.10," not in content, "CONTRIBUTING.md mentions Python 3.10"

    def test_contributing_has_code_style_section(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        content_lower = content.lower()
        assert "code style" in content_lower or "coding standard" in content_lower or "style" in content_lower, (
            "CONTRIBUTING.md must include a code style section"
        )

    def test_contributing_mentions_ruff(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "ruff" in content, "CONTRIBUTING.md must mention ruff as the linter/formatter"

    def test_contributing_mentions_mypy(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "mypy" in content, "CONTRIBUTING.md must mention mypy for type checking"

    def test_contributing_has_pr_workflow_section(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        content_lower = content.lower()
        assert "pull request" in content_lower or "pr" in content_lower, (
            "CONTRIBUTING.md must describe the pull request workflow"
        )

    def test_contributing_mentions_license(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "Apache" in content, "CONTRIBUTING.md must mention the Apache 2.0 license"

    def test_contributing_has_plugin_development_section(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        content_lower = content.lower()
        assert "plugin" in content_lower, "CONTRIBUTING.md must cover plugin development"

    def test_contributing_mentions_registry_guides(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "mloda-registry" in content, "CONTRIBUTING.md must reference the mloda-registry guides"
        assert "guides" in content.lower(), "CONTRIBUTING.md must mention the plugin development guides"

    def test_contributing_mentions_plugin_template(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        assert "mloda-plugin-template" in content, "CONTRIBUTING.md must reference the mloda-plugin-template"

    def test_contributing_describes_fork_workflow(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        content_lower = content.lower()
        assert "fork" in content_lower, "CONTRIBUTING.md must describe the fork workflow for external contributors"

    def test_contributing_clarifies_pytest_is_not_sufficient(self) -> None:
        content = _read_text(PROJECT_ROOT / "CONTRIBUTING.md")
        content_lower = content.lower()
        assert "not a substitute" in content_lower or "not sufficient" in content_lower, (
            "CONTRIBUTING.md must clarify that running pytest alone is not a substitute for tox"
        )


class TestLicenseFile:
    def test_license_file_exists(self) -> None:
        assert (PROJECT_ROOT / "LICENSE").is_file(), "LICENSE file must exist at project root"

    def test_no_uppercase_license_extension(self) -> None:
        assert not (PROJECT_ROOT / "LICENSE.TXT").exists(), "LICENSE.TXT should not exist; use LICENSE instead"

    def test_license_is_apache2(self) -> None:
        content = _read_text(PROJECT_ROOT / "LICENSE")
        assert "Apache License, Version 2.0" in content

    def test_license_is_spdx_expression(self) -> None:
        """License is an SPDX string and no License classifier is present."""
        project = _load_project()
        assert project["license"] == "Apache-2.0"
        assert not [c for c in project.get("classifiers", []) if c.startswith("License ::")]


def _load_project() -> dict[str, Any]:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    project: dict[str, Any] = data["project"]
    return project


def _load_extras() -> dict[str, list[str]]:
    extras: dict[str, list[str]] = _load_project()["optional-dependencies"]
    return extras


def _normalize_extra_name(name: str) -> str:
    return name.replace("_", "-")


class TestExtrasConsistency:
    # nopyarrow_test is a test-only extra that must omit pyarrow (used by the
    # nopyarrow tox env); like spark, it is intentionally not part of [all].
    INTENTIONALLY_EXCLUDED_FROM_ALL = {"all", "spark", "nopyarrow_test"}

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


class TestRuffConfig:
    """Validate that ruff lint rules enforce modern Python typing conventions."""

    def test_modern_typing_rules_configured(self) -> None:
        """UP006, UP007, UP045, UP036 (outdated version blocks) and UP017 (datetime.UTC) must be enforced."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        extend_select = data.get("tool", {}).get("ruff", {}).get("lint", {}).get("extend-select", [])
        assert "UP006" in extend_select, "ruff must enforce UP006 (use builtin generics instead of typing generics)"
        assert "UP007" in extend_select, "ruff must enforce UP007 (use X | Y instead of Union)"
        assert "UP045" in extend_select, "ruff must enforce UP045 (use X | None instead of Optional)"
        assert "UP036" in extend_select, "ruff must enforce UP036 (remove outdated version blocks)"
        assert "UP017" in extend_select, "ruff must enforce UP017 (use datetime.UTC instead of timezone.utc)"

    def test_no_redundant_typing_generics_in_source(self) -> None:
        """Source files must not import redundant typing generics that UP006/UP007 replace."""
        redundant_names = {"Dict", "FrozenSet", "List", "Set", "Tuple", "Type"}
        source_dirs = [PROJECT_ROOT / "mloda", PROJECT_ROOT / "mloda_plugins"]
        violations: list[str] = []
        for source_dir in source_dirs:
            for py_file in source_dir.rglob("*.py"):
                for i, line in enumerate(_read_text(py_file).splitlines(), start=1):
                    if not line.startswith("from typing import"):
                        continue
                    imported = {name.strip() for name in line.split("import")[1].split(",")}
                    found = imported & redundant_names
                    if found:
                        violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i} imports {found}")
        assert not violations, "Redundant typing imports found:\n" + "\n".join(violations)

    def test_no_optional_typing_usage_in_source(self) -> None:
        """Source and test files must not use Optional-style annotations that UP045 replaces."""
        needle = "Optional" + "["
        source_dirs = [PROJECT_ROOT / "mloda", PROJECT_ROOT / "mloda_plugins", PROJECT_ROOT / "tests"]
        violations: list[str] = []
        for source_dir in source_dirs:
            for py_file in source_dir.rglob("*.py"):
                for i, line in enumerate(_read_text(py_file).splitlines(), start=1):
                    if needle in line:
                        violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{i}")
        assert not violations, f"Found {len(violations)} {needle}...] usages (showing up to 20):\n" + "\n".join(
            violations[:20]
        )


class TestPackagingConfig:
    """Validate that pyproject.toml is the single source of packaging truth."""

    def test_no_setup_py(self) -> None:
        assert not (PROJECT_ROOT / "setup.py").exists(), (
            "setup.py must not exist; pyproject.toml is the single source of packaging configuration"
        )

    def test_no_setup_cfg(self) -> None:
        assert not (PROJECT_ROOT / "setup.cfg").exists(), (
            "setup.cfg must not exist; pyproject.toml is the single source of packaging configuration"
        )

    def test_pyproject_has_build_system(self) -> None:
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        assert "build-system" in data, "pyproject.toml must have a [build-system] section"
        assert "requires" in data["build-system"], "pyproject.toml [build-system] must specify 'requires'"
        assert "build-backend" in data["build-system"], "pyproject.toml [build-system] must specify 'build-backend'"

    def test_python_classifiers_match_requires_python(self) -> None:
        """Python 3.X classifiers equal the minors admitted by requires-python."""
        project = _load_project()
        spec = SpecifierSet(project["requires-python"])
        prefix = "Programming Language :: Python :: 3."
        expected = {f"{prefix}{m}" for m in range(31) if spec.contains(f"3.{m}")}
        actual = {c for c in project.get("classifiers", []) if re.fullmatch(re.escape(prefix) + r"\d+", c)}
        assert actual == expected
        assert "3.10" not in spec, "requires-python must exclude 3.10"
        assert "3.11" in spec, "requires-python must admit 3.11"


class TestToxConfig:
    """Validate the tox settings the CI and release workflows depend on."""

    TOX_INSTALL = re.compile(r"\binstall\b.*(?<![\w-])tox(?![\w-])")
    TOX_PIN = re.compile(r"uv tool install tox==\S+ --with tox-uv==\S+$")

    @staticmethod
    def _tox_parser() -> configparser.ConfigParser:
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
        assert parser.read(PROJECT_ROOT / "tox.ini", encoding="utf-8"), "tox.ini not found"
        return parser

    @staticmethod
    def _workflow(name: str) -> dict[str, Any]:
        config: dict[str, Any] = yaml.safe_load(_read_text(PROJECT_ROOT / ".github" / "workflows" / name))
        return config

    @classmethod
    def _workflow_jobs(cls, name: str) -> dict[str, Any]:
        jobs: dict[str, Any] = cls._workflow(name)["jobs"]
        return jobs

    def test_tox_opts_out_of_venv_redirect(self) -> None:
        """tox >= 4.64 otherwise writes a .venv redirect file that makes the release job's `uv lock` fail."""
        parser = self._tox_parser()
        assert not parser.getboolean("tox", "venv_redirect", fallback=True), (
            "tox.ini must set `venv_redirect = false` under [tox]"
        )

    def test_workflows_pin_the_same_tox(self) -> None:
        pins: dict[str, set[str]] = {}
        for workflow in sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.y*ml")):
            for line in _read_text(workflow).splitlines():
                command = line.strip()
                if command.startswith("#") or not self.TOX_INSTALL.search(command):
                    continue
                match = self.TOX_PIN.search(command)
                assert match, (
                    f"{workflow.name} must install `uv tool install tox==X --with tox-uv==Y`, found: {command}"
                )
                pins.setdefault(workflow.name, set()).add(match.group(0))
        assert {"ci.yaml", "release.yaml"} <= pins.keys(), f"ci.yaml and release.yaml must install tox: {pins}"
        assert len(set().union(*pins.values())) == 1, f"workflows must install the same tox and tox-uv: {pins}"

    def test_tox_splits_tests_from_lint(self) -> None:
        """Plain `tox` runs python311 and lint; lint tools run only in the lint env."""
        parser = self._tox_parser()
        envlist = parser.get("tox", "envlist")
        envs = set(re.findall(r"[\w-]+", envlist))
        assert {"python311", "lint"} <= envs, f"envlist lacks python311/lint: {envlist}"
        assert "python310" not in envs, f"envlist must not include python310: {envlist}"
        assert parser.has_section("testenv:lint"), "tox.ini needs a [testenv:lint] section"
        lint = parser.get("testenv:lint", "commands", fallback="")
        for tool in ("ruff format", "ruff check", "pip-licenses", "mypy", "bandit"):
            assert tool in lint, f"[testenv:lint] commands must run {tool}"
        base = parser.get("testenv", "commands")
        assert "pytest" in base, "[testenv] commands must run pytest"
        for tool in ("ruff", "mypy", "bandit", "pip-licenses"):
            assert not re.search(rf"(?<![\w-]){tool}(?![\w-])", base), f"[testenv] commands must not run {tool}"
        assert "pytest" not in lint, "[testenv:lint] commands must not run pytest"
        bandit = [line.strip() for line in lint.splitlines() if line.strip().startswith("bandit")]
        assert bandit == [
            "bandit -c pyproject.toml -r -q -x ./tests .",
            "bandit -c pyproject.toml -r -q -s B105,B106 tests",
        ], f"bandit must run all rules outside tests and skip only B105/B106 in tests: {bandit}"
        overridden = [
            key
            for key in ("extras", "deps", "skip_install", "runner", "package", "basepython")
            if parser.has_option("testenv:lint", key)
        ]
        assert not overridden, f"[testenv:lint] must inherit the full [testenv] install, overrides: {overridden}"

    def test_bandit_exclude_dirs_skip_only_artifact_dirs(self) -> None:
        """Bandit exclude entries must match only artifact dirs, never shipped or test files."""
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            entries = tomllib.load(handle)["tool"]["bandit"]["exclude_dirs"]

        def excluded(path: str) -> bool:
            return any(entry in path or fnmatch.fnmatch(path, entry) for entry in entries)

        skipped: list[str] = []
        for top in ("mloda", "mloda_plugins"):
            for file in (PROJECT_ROOT / top).rglob("*.py"):
                path = "./" + file.relative_to(PROJECT_ROOT).as_posix()
                if excluded(path):
                    skipped.append(path)
        for file in (PROJECT_ROOT / "tests").rglob("*.py"):
            path = file.relative_to(PROJECT_ROOT).as_posix()
            if excluded(path):
                skipped.append(path)
        assert not skipped, f"bandit would skip shipped or test files: {skipped}"

        artifacts = [
            "./build/lib/mloda/x.py",
            "./dist/x.py",
            "./.venv/lib/x.py",
            "./venv/lib/x.py",
            "./.tox/lint/x.py",
            "./attribution/attributions.py",
            "./mloda.egg-info/x.py",
            "./.cache/mloda-main/x.py",
            "./.claude/worktrees/w/x.py",
            "./.vscode/x.py",
            "./lib/site-packages/x.py",
        ]
        kept = [path for path in artifacts if not excluded(path)]
        assert not kept, f"bandit must exclude artifact paths: {kept}"

    def test_mcp_env_fails_when_its_smoke_test_is_skipped(self) -> None:
        """The mcp env runs one stdio smoke test; a failed `mcp` import must not skip it into a green run."""
        parser = self._tox_parser()
        sections = [name for name in parser.sections() if name.replace(" ", "") == "testenv:mcp"]
        assert len(sections) == 1, f"tox.ini needs exactly one [testenv:mcp] section: {parser.sections()}"
        setenv: dict[str, str] = {}
        for line in parser.get(sections[0], "setenv", fallback="").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                setenv[key.strip()] = value.strip()
        defaults = {key: re.sub(r"^\{env:.*:([^:}]*)\}$", r"\1", value) for key, value in setenv.items()}
        assert defaults.get("CHECK_SKIP_COUNT") == "1", f"[testenv:mcp] must default CHECK_SKIP_COUNT to 1: {setenv}"
        assert defaults.get("EXPECTED_SKIP_COUNT") == "0", (
            f"[testenv:mcp] must default EXPECTED_SKIP_COUNT to 0: {setenv}"
        )

    def test_ci_runs_pandas2_env(self) -> None:
        """Lockfile everywhere except pandas/numpy, which are capped to pandas 2.x / numpy < 2.4."""
        parser = self._tox_parser()
        sections = [name for name in parser.sections() if name.replace(" ", "") == "testenv:pandas2"]
        assert len(sections) == 1, f"tox.ini needs exactly one [testenv:pandas2] section: {parser.sections()}"
        section = sections[0]
        overridden = [key for key in ("runner", "deps", "package", "extras") if parser.has_option(section, key)]
        assert not overridden, f"[testenv:pandas2] must inherit the lockfile install from [testenv]: {overridden}"
        lines = [ln.strip() for ln in parser.get(section, "commands_pre", fallback="").splitlines() if ln.strip()]
        installs = [
            i
            for i, ln in enumerate(lines)
            if ln.startswith("uv pip install") and "pandas<3" in ln and "numpy<2.4" in ln
        ]
        assert installs, f"commands_pre needs a `uv pip install` line capping pandas<3 and numpy<2.4: {lines}"
        guards = [
            ln
            for ln in lines[installs[0] + 1 :]
            if all(word in ln for word in ("pandas", "numpy", "assert", "__version__"))
        ]
        assert guards, f"commands_pre needs an assert on pandas/numpy __version__ after the install line: {lines}"
        matrix = self._workflow_jobs("ci.yaml")["build"]["strategy"]["matrix"]["include"]
        assert any(str(e.get("python-version")) == "3.11" and e.get("toxenv") == "-e pandas2" for e in matrix), (
            f"ci.yaml build matrix needs an include entry for python 3.11 with toxenv '-e pandas2': {matrix}"
        )

    def test_ci_runs_lint_env(self) -> None:
        jobs = self._workflow_jobs("ci.yaml")
        assert "lint" in jobs, "ci.yaml needs a lint job"
        tox_lint = re.compile(r"tox\s+-e\s+lint\b")
        assert any(tox_lint.search(str(s.get("run", ""))) for s in jobs["lint"].get("steps", [])), (
            "lint job needs a step running `tox -e lint`"
        )
        versions = {str(v) for v in jobs["lint"].get("strategy", {}).get("matrix", {}).get("python-version", [])}
        assert versions == {"3.11", "3.14"}, f"lint job must matrix python-version over 3.11 and 3.14: {versions}"
        assert not any(tox_lint.search(str(s.get("run", ""))) for s in jobs["build"].get("steps", [])), (
            "build job must not run `tox -e lint`"
        )

    def test_ci_build_job_derives_workers_from_nproc(self) -> None:
        steps = self._workflow_jobs("ci.yaml")["build"]["steps"]
        write = re.compile(r"PYTEST_WORKERS=\$\(\(.*nproc.*\b3\b.*\)\).*GITHUB_ENV")
        assert any(write.search(line) for s in steps for line in str(s.get("run", "")).splitlines()), (
            "build job must write nproc, capped at 3, into PYTEST_WORKERS via $GITHUB_ENV"
        )
        hardcoded = [s for s in steps if "PYTEST_WORKERS" in (s.get("env") or {})]
        assert not hardcoded, "no build step may hardcode PYTEST_WORKERS in env"

    def test_release_license_write_runs_lint_env(self) -> None:
        steps = [s for job in self._workflow_jobs("release.yaml").values() for s in job.get("steps", [])]
        writers = [s for s in steps if "TOX_WRITE_THIRD_PARTY_LICENSES" in (s.get("env") or {})]
        assert writers, "release.yaml needs a step setting TOX_WRITE_THIRD_PARTY_LICENSES"
        for step in writers:
            run = str(step.get("run", ""))
            envs = {env for arg in re.findall(r"-e\s+(\S+)", run) for env in arg.split(",")}
            assert "lint" in envs, f"license step must run tox -e with lint: {run}"

    def test_ci_cancels_only_superseded_pr_runs(self) -> None:
        concurrency = self._workflow("ci.yaml").get("concurrency", {})
        assert concurrency.get("cancel-in-progress") is True, "ci.yaml must cancel in-progress runs"
        group = str(concurrency.get("group", ""))
        assert "pull_request" in group and "run_id" in group, (
            f"concurrency group must be unique per push run (run_id) and shared per PR: {group}"
        )
