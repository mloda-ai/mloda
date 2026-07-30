"""The version-independent half of the tox gate must cost one CI job, not one per interpreter."""

import configparser
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOX_INI = PROJECT_ROOT / "tox.ini"
CI_YAML = PROJECT_ROOT / ".github" / "workflows" / "ci.yaml"
RELEASE_YAML = PROJECT_ROOT / ".github" / "workflows" / "release.yaml"

# Checks whose verdict cannot differ between interpreters: the CI matrix should pay for them once. The two
# pip-licenses writers belong here because nothing consumes their output per interpreter: the files are read
# from the default env only (a contributor's `tox`, and the release workflow's `tox -e <gate env>`).
VERSION_INDEPENDENT_CHECKS = ("ruff format", "ruff check", "bandit", "THIRD_PARTY_LICENSES", "ATTRIBUTION")
# Checks that resolve differently per interpreter and must therefore run in every python3XX env. The
# pip-licenses allowlist (`--allow-only`) inspects the *installed* distributions, and `runner =
# uv-venv-lock-runner` installs a different set per interpreter (pandas 2.3.3 on 3.10 vs 3.0.3 on >=3.11,
# psutil only on >=3.11, tomli only on 3.10), so gating it would leave 3.11-3.14 license-unchecked.
VERSION_DEPENDENT_CHECKS = ("pytest", "mypy", "--allow-only")

# tox splits a factor prefix off a `commands` line at the first `:` followed by whitespace or the line end,
# wherever it sits on the line (tox/config/loader/ini/factor.py::expand_factors). Negation (`!python310:`)
# and brace groups (`{python311,python312}:`) are part of that prefix, so any whitespace-free text in front
# of such a colon gates the command and must not be mistaken for the command itself.
FACTOR_SEPARATOR = re.compile(r":(?=\s|$)")

# The gate factor doubles as an env name in the CI matrix and the release workflow, so the guards below only
# reason correctly about a single bare env: no negation, no brace group, no comma list.
PLAIN_ENV_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")

# tox drops (`-`) or inverts (`!`) the exit code of a command whose first token is that character.
EXIT_CODE_OPT_OUTS = ("-", "!")

# Defining any of these in `[testenv:<env>]` replaces the base `[testenv]` list wholesale.
COMMAND_KEYS = ("commands", "commands_pre", "commands_post")

TOX_ENV_INVOCATION = re.compile(r"\btox\b[^\n]*?\s-e[=\s]+(?P<env>[\w.-]+)")

NO_GATE_FACTOR = (
    "The version-independent [testenv] commands do not share one tox factor prefix, so no single env owns "
    "the static gate (see test_version_independent_checks_share_one_factor)."
)


def _tox_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(TOX_INI.read_text(encoding="utf-8"))
    return config


def _envlist() -> list[str]:
    """tox's default envs. `env_list` is tox 4's spelling; either key may carry per-line `#` comments."""
    section = _tox_config()["tox"]
    raw = next((section[key] for key in ("envlist", "env_list") if key in section), "")
    uncommented = " ".join(line.split("#", 1)[0] for line in raw.splitlines())
    return [env for env in re.split(r"[,\s]+", uncommented) if env]


def _testenv_commands() -> list[str]:
    """The `commands` lines of the default [testenv], without blank and comment lines."""
    lines = [line.strip() for line in _tox_config()["testenv"]["commands"].splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _split_command(command: str) -> tuple[str | None, str]:
    """(factor prefix, command body), read the way tox reads a `commands` line."""
    marker = FACTOR_SEPARATOR.search(command)
    if marker is None:
        return None, command
    candidate = command[: marker.start()].strip()
    if not candidate or any(char.isspace() for char in candidate):
        return None, command
    return candidate, command[marker.end() :].strip()


def _factor_of(command: str) -> str | None:
    return _split_command(command)[0]


def _body_of(command: str) -> str:
    return _split_command(command)[1]


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


def _env_section_keys(env: str) -> list[str]:
    """The option names of `[testenv:<env>]` (the section name may be spelled with spaces), else []."""
    config = _tox_config()
    for name in config.sections():
        head, separator, tail = name.partition(":")
        if separator and head.strip() == "testenv" and tail.strip() == env:
            return list(config[name])
    return []


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


def _run_scripts(workflow: Path) -> Iterator[str]:
    """Every `run:` script of a workflow, wherever it sits in the job/step tree."""

    def walk(node: Any) -> Iterator[str]:
        if isinstance(node, dict):
            script = node.get("run")
            if isinstance(script, str):
                yield script
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    yield from walk(yaml.safe_load(workflow.read_text(encoding="utf-8")))


def _release_tox_envs() -> list[str]:
    envs = {
        match.group("env") for script in _run_scripts(RELEASE_YAML) for match in TOX_ENV_INVOCATION.finditer(script)
    }
    return sorted(envs)


@pytest.mark.parametrize("check", VERSION_INDEPENDENT_CHECKS)
def test_version_independent_check_is_factor_gated(check: str) -> None:
    commands = _commands_running(check)
    assert commands, f"No [testenv] command in {TOX_INI} mentions {check!r} any more. Retarget or drop this guard."
    unguarded = [command for command in commands if _factor_of(command) is None]
    assert not unguarded, (
        f"{TOX_INI}: [testenv] command {_short(unguarded[0])!r} runs {check!r} in every python3XX env, but its "
        "verdict cannot differ between interpreters. Prefix it with a tox factor (e.g. `python310: ...`) so the "
        "CI matrix runs it once."
    )


@pytest.mark.parametrize("check", VERSION_DEPENDENT_CHECKS)
def test_version_dependent_check_runs_in_every_env(check: str) -> None:
    commands = _commands_running(check)
    assert commands, f"No [testenv] command in {TOX_INI} mentions {check!r} any more. Retarget or drop this guard."
    gated = [command for command in commands if _factor_of(command) is not None]
    assert not gated, (
        f"{TOX_INI}: [testenv] command {_short(gated[0])!r} is gated on factor {_factor_of(gated[0])!r}, but "
        f"{check!r} is version-dependent: `runner = uv-venv-lock-runner` installs a different distribution set "
        "per interpreter, so it must keep running on every env in the matrix. Drop the factor prefix."
    )


def test_version_independent_checks_share_one_factor() -> None:
    factors = {_factor_of(command) for command in _version_independent_commands()}
    assert len(factors) == 1 and None not in factors, (
        f"{TOX_INI}: the version-independent checks must all carry the same tox factor prefix, otherwise the "
        "static gate is split across matrix jobs. Factors found (None = ungated): "
        f"{sorted(str(factor) for factor in factors)}."
    )


def test_gate_factor_is_a_plain_env_name() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    assert PLAIN_ENV_NAME.fullmatch(factor), (
        f"{TOX_INI}: the static gate is bound to factor expression {factor!r}. A negation (`!env:`), a brace "
        "group (`{env1,env2}:`) or a comma list selects a varying number of envs, so the gate would run zero or "
        "many times. Use one bare env name."
    )


def test_gate_factor_is_a_default_env() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    envlist = _envlist()
    assert factor in envlist, (
        f"{TOX_INI}: the static gate is bound to factor {factor!r}, which is not in [tox] envlist {envlist}. Plain "
        "`tox` and the release workflow's `tox -e <default env>` would then skip ruff, bandit and the license "
        f"writers. Add {factor!r} to envlist."
    )


def test_no_env_section_redefines_the_gate_commands() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    keys = _env_section_keys(factor)
    overrides = [key for key in COMMAND_KEYS if key in keys]
    assert not overrides, (
        f"{TOX_INI}: section [testenv:{factor}] defines {overrides}, which replaces the [testenv] command list "
        f"wholesale. Ruff, bandit and the license writers would vanish from plain `tox`, from `tox -e {factor}` "
        f"and from the release workflow, with every other guard still green. Keep [testenv:{factor}] to settings "
        "such as basepython and leave the commands in [testenv]."
    )


def test_gated_commands_do_not_ignore_their_exit_code() -> None:
    offenders = [
        command for command in _version_independent_commands() if _body_of(command).startswith(EXIT_CODE_OPT_OUTS)
    ]
    assert not offenders, (
        f"{TOX_INI}: [testenv] command {_short(offenders[0])!r} starts with `-` or `!`, which makes tox ignore or "
        "invert its exit code. The one job that owns the static gate would then report success no matter what "
        "ruff, bandit or pip-licenses find. Drop the leading token."
    )


def test_release_workflow_runs_the_gate_env() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    envs = _release_tox_envs()
    assert envs == [factor], (
        f"{RELEASE_YAML} invokes tox env(s) {envs}, while {TOX_INI} binds the static gate to factor {factor!r}. "
        "That release step generates attribution/THIRD_PARTY_LICENSES.md from the gate env and fails the release "
        f"if the file is missing, so it must run `tox -e {factor}`."
    )


def test_gate_env_runs_in_exactly_one_matrix_entry() -> None:
    factor = _gate_factor()
    assert factor is not None, NO_GATE_FACTOR
    versions = [version for version, env in _matrix_entries() if env == factor]
    assert len(versions) == 1, (
        f"{CI_YAML}: the build matrix runs tox env {factor!r} in {len(versions)} jobs {versions}, so the static "
        "gate executes that many times per CI run. Exactly one matrix entry must carry it."
    )


def test_matrix_keeps_one_versioned_env_per_python_version() -> None:
    entries = _matrix_entries()
    for version in _matrix_python_versions():
        env = "python" + version.replace(".", "")
        assert [v for v, e in entries if e == env] == [version], (
            f"{CI_YAML}: the build matrix must run tox env {env!r} exactly once, on Python {version}. Binding the "
            "static gate to one env must not shrink the per-interpreter pytest, mypy and license-allowlist "
            "coverage."
        )
