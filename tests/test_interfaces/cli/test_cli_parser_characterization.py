"""Characterization of the ``mloda`` argparse setup: help text, parsed namespaces and usage-error exit codes.

Pins current behavior so a refactor of ``_parser()`` can be proven behavior-identical. Python 3.13 changed
usage wrapping, and the quoting of invalid choices varies by patch release; both are handled explicitly.
"""

import io
import sys
from collections.abc import Sequence
from typing import Any

import pytest

from mloda.core.version import get_mloda_version
from mloda.interfaces.cli.main import _parser, main

ROOT_HELP = """\
usage: mloda [-h] [--version] {explain,run,mcp} ...

Explain, run or serve mloda feature requests.

positional arguments:
  {explain,run,mcp}
    explain          Plan a feature request and print the plan.
    run              Plan and execute a feature request and print its preview.
    mcp              Serve the feature catalog, explain and run to MCP clients.

options:
  -h, --help         show this help message and exit
  --version          show program's version number and exit
"""

EXPLAIN_USAGE = """\
usage: mloda explain [-h] [-o KEY=VALUE] [-c KEY=VALUE] [--config CONFIG | --no-config]
                     TARGET [TARGET ...]
"""

EXPLAIN_HELP = (
    EXPLAIN_USAGE
    + """
positional arguments:
  TARGET                Feature names, or one request file (a path, a .json name, or - for stdin).

options:
  -h, --help            show this help message and exit
  -o, --option KEY=VALUE
                        Group option for every feature; KEY:=JSON parses JSON, KEY=@FILE reads a
                        file.
  -c, --context-option KEY=VALUE
                        Context option for every feature; same value forms as --option.
  --config CONFIG       Settings file (default: ./mloda.toml if present).
  --no-config           Ignore any settings file.
"""
)

if sys.version_info >= (3, 13):
    RUN_USAGE = """\
usage: mloda run [-h] [--max-rows MAX_ROWS] [--format {json,jsonl,csv}] [--full] [-o KEY=VALUE]
                 [-c KEY=VALUE] [--config CONFIG | --no-config]
                 TARGET [TARGET ...]
"""
else:
    RUN_USAGE = """\
usage: mloda run [-h] [--max-rows MAX_ROWS] [--format {json,jsonl,csv}] [--full] [-o KEY=VALUE]
                 [-c KEY=VALUE] [--config CONFIG | --no-config]
                 TARGET [TARGET ...]
"""

RUN_HELP = (
    RUN_USAGE
    + """
positional arguments:
  TARGET                Feature names, or one request file (a path, a .json name, or - for stdin).

options:
  -h, --help            show this help message and exit
  --max-rows MAX_ROWS   Refuse sources and results over this, and uncountable reads.
  --format {json,jsonl,csv}
                        Output format of the result.
  --full                Return every row and column, not the classified preview.
  -o, --option KEY=VALUE
                        Group option for every feature; KEY:=JSON parses JSON, KEY=@FILE reads a
                        file.
  -c, --context-option KEY=VALUE
                        Context option for every feature; same value forms as --option.
  --config CONFIG       Settings file (default: ./mloda.toml if present).
  --no-config           Ignore any settings file.
"""
)

if sys.version_info < (3, 13):
    # Before 3.13 argparse repeats the metavar after every flag.
    EXPLAIN_HELP = EXPLAIN_HELP.replace("-o, --option KEY=VALUE", "-o KEY=VALUE, --option KEY=VALUE").replace(
        "-c, --context-option KEY=VALUE", "-c KEY=VALUE, --context-option KEY=VALUE"
    )
    RUN_HELP = RUN_HELP.replace("-o, --option KEY=VALUE", "-o KEY=VALUE, --option KEY=VALUE").replace(
        "-c, --context-option KEY=VALUE", "-c KEY=VALUE, --context-option KEY=VALUE"
    )

MCP_HELP = """\
usage: mloda mcp [-h] {serve} ...

positional arguments:
  {serve}
    serve     Serve over stdio or http; settings from the config file, then MLODA_MCP_* env vars.

options:
  -h, --help  show this help message and exit
"""

MCP_SERVE_HELP = """\
usage: mloda mcp serve [-h] [--config CONFIG | --no-config] [--transport {stdio,http}]
                       [--host HOST] [--port PORT]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Settings file (default: ./mloda.toml if present).
  --no-config           Ignore any settings file.
  --transport {stdio,http}
                        Transport to serve over (default: stdio).
  --host HOST           Address to bind for the http transport (default: 127.0.0.1).
  --port PORT           Port to bind for the http transport (default: 8000).
"""

MCP_SERVE_DEFAULTS: dict[str, Any] = {
    "command": "mcp",
    "mcp_command": "serve",
    "config": None,
    "transport": "stdio",
    "host": "127.0.0.1",
    "port": 8000,
}

ROOT_USAGE = "usage: mloda [-h] [--version] {explain,run,mcp} ...\n"


@pytest.fixture(autouse=True)
def _deterministic_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PYTHON_COLORS", raising=False)


def _cli(argv: Sequence[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), env={}, stdin=io.StringIO(""), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--help"], ROOT_HELP),
        (["explain", "--help"], EXPLAIN_HELP),
        (["run", "--help"], RUN_HELP),
        (["mcp", "--help"], MCP_HELP),
        (["mcp", "serve", "--help"], MCP_SERVE_HELP),
    ],
)
def test_help_output_is_pinned(argv: list[str], expected: str) -> None:
    assert _cli(argv) == (0, expected, "")


def test_version_flag_prints_the_installed_version() -> None:
    assert _cli(["--version"]) == (0, f"mloda {get_mloda_version()}\n", "")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["explain", "req.json"],
            {
                "command": "explain",
                "targets": ["req.json"],
                "option": None,
                "context_option": None,
                "config": None,
                "no_config": False,
            },
        ),
        (
            ["explain", "-", "--config", "c.toml"],
            {
                "command": "explain",
                "targets": ["-"],
                "option": None,
                "context_option": None,
                "config": "c.toml",
                "no_config": False,
            },
        ),
        (
            ["run", "req.json", "--max-rows", "5", "--format", "csv", "--no-config"],
            {
                "command": "run",
                "targets": ["req.json"],
                "option": None,
                "context_option": None,
                "max_rows": 5,
                "format": "csv",
                "full": False,
                "config": None,
                "no_config": True,
            },
        ),
        (
            ["run", "req.json"],
            {
                "command": "run",
                "targets": ["req.json"],
                "option": None,
                "context_option": None,
                "max_rows": None,
                "format": "json",
                "full": False,
                "config": None,
                "no_config": False,
            },
        ),
        (
            ["run", "req.json", "--full"],
            {
                "command": "run",
                "targets": ["req.json"],
                "option": None,
                "context_option": None,
                "max_rows": None,
                "format": "json",
                "full": True,
                "config": None,
                "no_config": False,
            },
        ),
        (
            ["explain", "income", "segment", "-o", "a=1", "--option", "b:=2", "-c", "c=@f.txt"],
            {
                "command": "explain",
                "targets": ["income", "segment"],
                "option": ["a=1", "b:=2"],
                "context_option": ["c=@f.txt"],
                "config": None,
                "no_config": False,
            },
        ),
        (
            ["run", "income", "--context-option", "c=1", "--full"],
            {
                "command": "run",
                "targets": ["income"],
                "option": None,
                "context_option": ["c=1"],
                "max_rows": None,
                "format": "json",
                "full": True,
                "config": None,
                "no_config": False,
            },
        ),
        (["mcp", "serve"], {**MCP_SERVE_DEFAULTS, "no_config": False}),
        (["mcp", "serve", "--no-config"], {**MCP_SERVE_DEFAULTS, "no_config": True}),
        (
            ["mcp", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "9000"],  # nosec B104
            {**MCP_SERVE_DEFAULTS, "no_config": False, "transport": "http", "host": "0.0.0.0", "port": 9000},  # nosec B104
        ),
    ],
)
def test_parsed_namespace_is_pinned(argv: list[str], expected: dict[str, Any]) -> None:
    assert vars(_parser().parse_args(argv)) == expected


@pytest.mark.parametrize(
    ("argv", "expected_err"),
    [
        ([], ROOT_USAGE + "mloda: error: the following arguments are required: command\n"),
        (
            ["bogus"],
            ROOT_USAGE + "mloda: error: argument command: invalid choice: bogus (choose from explain, run, mcp)\n",
        ),
        (
            ["run", "req.json", "--format", "xml"],
            RUN_USAGE + "mloda run: error: argument --format: invalid choice: xml (choose from json, jsonl, csv)\n",
        ),
        (["run"], RUN_USAGE + "mloda run: error: the following arguments are required: TARGET\n"),
        (
            ["explain", "a", "--config", "x", "--no-config"],
            EXPLAIN_USAGE + "mloda explain: error: argument --no-config: not allowed with argument --config\n",
        ),
    ],
)
def test_usage_error_exits_2_on_stderr_only(argv: list[str], expected_err: str) -> None:
    code, out, err = _cli(argv)
    assert (code, out) == (2, "")
    # Whether invalid choices are quoted differs across Python patch releases, so quotes are not pinned.
    assert err.replace("'", "") == expected_err
