"""The version-independent half of the tox gate must cost one CI job, not one per interpreter."""

import configparser
import re
from pathlib import Path
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOX_INI = PROJECT_ROOT / "tox.ini"
CI_YAML = PROJECT_ROOT / ".github" / "workflows" / "ci.yaml"

# Checks whose verdict cannot differ between interpreters: the CI matrix should pay for them once.
VERSION_INDEPENDENT_CHECKS = ("ruff format", "ruff check", "pip-licenses", "bandit")
# Checks that resolve differently per interpreter and must therefore run in every python3XX env.
VERSION_DEPENDENT_CHECKS = ("pytest", "mypy")

# A tox factor prefix, e.g. `python310: ruff check .`. Command bodies contain colons inside quoted shell
# strings, so only a bare identifier plus colon at the very start of a line counts.
FACTOR_PREFIX = re.compile(r"^(?P<factor>[A-Za-z0-9_][A-Za-z0-9_.,!-]*):\s*(?P<command>\S.*)$")

NO_GATE_FACTOR = (
    "The version-independent [testenv] commands do not share one tox factor prefix, so no single env owns "
    "the static gate (see test_version_independent_checks_share_one_factor)."
)


def _tox_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(TOX_INI.read_text(encoding="utf-8"))
    return config


def _envlist() -> list[str]:
    """tox's default envs; the value carries a trailing `#` comment listing the ones CI adds."""
    raw = _tox_config()["tox"]["envlist"].split("#", 1)[0]
    return [env for env in re.split(r"[,\s]+", raw) if env]


def _testenv_commands() -> list[str]:
    """The `commands` lines of the default [testenv], without blank and comment lines."""
    lines = [line.strip() for line in _tox_config()["testenv"]["commands"].splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _factor_of(command: str) -> str | None:
    match = FACTOR_PREFIX.match(command)
    return match.group("factor") if match else None


def _body_of(command: str) -> str:
    match = FACTOR_PREFIX.match(command)
    return match.group("command") if match else command


def _commands_running(check: str) -> list[str]:
    return [command for command in _testenv_commands() if check in _body_of(command)]


def _version_independent_commands() -> list[str]:
    """One command can run several checks (the pip-licenses lines), so walk the list once."""
    return [
        command
        for command in _testenv_commands()
        if any(check in _body_of(command) for check in VERSION_INDEPENDENT_CHECKS)
    ]


def _gate_factor() -> str | None:
    """The one factor guarding the version-independent commands, or None if they disagree or are ungated."""
    factors = {_factor_of(command) for command in _version_independent_commands()}
    return factors.pop() if len(factors) == 1 else None


def _short(command: str) -> str:
    return command if len(command) <= 70 else f"{command[:70]}..."


def _build_matrix() -> dict[str, Any]:
    config: dict[Any, Any] = yaml.safe_load(CI_YAML.read_text(encoding="utf-8"))
    matrix: dict[str, Any] = config["jobs"]["build"]["strategy"]["matrix"]
    return matrix


def _matrix_python_versions() -> list[str]:
    return [str(version) for version in _build_matrix()["python-version"]]


def _env_name(toxenv: str) -> str:
    """`-e python310` -> `python310`."""
    return toxenv.removeprefix("-e").strip()


def _matrix_entries() -> list[tuple[str, str]]:
    """(python version, tox env) for every job the build matrix expands to: base product plus includes."""
    matrix = _build_matrix()
    base: list[str] = [_env_name(str(toxenv)) for toxenv in matrix.get("toxenv", [])]
    entries: list[tuple[str, str]] = [(version, env) for version in _matrix_python_versions() for env in base]
    entries += [
        (str(item["python-version"]), _env_name(str(item["toxenv"])))
        for item in matrix.get("include", [])
        if "python-version" in item and "toxenv" in item
    ]
    return entries


@pytest.mark.parametrize("check", VERSION_INDEPENDENT_CHECKS)
def test_version_independent_check_is_factor_gated(check: str) -> None:
    commands = _commands_running(check)
    assert commands, f"No [testenv] command runs {check!r} any more. Retarget this guard or drop the check."
    unguarded = [command for command in commands if _factor_of(command) is None]
    assert not unguarded, (
        f"[testenv] command {_short(unguarded[0])!r} runs {check!r} in every python3XX env, but its verdict "
        "cannot differ between interpreters. Prefix it with a tox factor (e.g. `python310: ...`) so the CI "
        "matrix runs it once."
    )


@pytest.mark.parametrize("check", VERSION_DEPENDENT_CHECKS)
def test_version_dependent_check_runs_in_every_env(check: str) -> None:
    commands = _commands_running(check)
    assert commands, f"No [testenv] command runs {check!r} any more. Retarget this guard or drop the check."
    gated = [command for command in commands if _factor_of(command) is not None]
    assert not gated, (
        f"[testenv] command {_short(gated[0])!r} is gated on factor {_factor_of(gated[0])!r}, but {check!r} is "
        "version-dependent: it must keep running on every interpreter in the matrix. Only version-independent "
        "checks may carry a factor prefix."
    )


def test_version_independent_checks_share_one_factor() -> None:
    factors = {_factor_of(command) for command in _version_independent_commands()}
    assert len(factors) == 1 and None not in factors, (
        "The version-independent checks must all carry the same tox factor prefix, otherwise the static gate is "
        f"split across matrix jobs. Factors found (None = ungated): {sorted(str(factor) for factor in factors)}."
    )


def test_gate_factor_is_a_default_env() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    envlist = _envlist()
    assert factor in envlist, (
        f"The static gate is bound to factor {factor!r}, which is not in [tox] envlist {envlist}. Plain `tox` and "
        "the release workflow's `tox -e <default env>` would then skip ruff, bandit and the license checks."
    )


def test_gate_env_runs_in_exactly_one_matrix_entry() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    versions = [version for version, env in _matrix_entries() if env == factor]
    assert len(versions) == 1, (
        f"The CI build matrix runs tox env {factor!r} in {len(versions)} jobs {versions}, so the static gate "
        "executes that many times per CI run. Exactly one matrix entry must carry it."
    )


def test_matrix_keeps_one_versioned_env_per_python_version() -> None:
    entries = _matrix_entries()
    for version in _matrix_python_versions():
        env = "python" + version.replace(".", "")
        assert [v for v, e in entries if e == env] == [version], (
            f"The CI build matrix must run tox env {env!r} exactly once, on Python {version}. Binding the static "
            "gate to one env must not shrink the per-interpreter pytest and mypy coverage."
        )
