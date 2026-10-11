"""The ``mloda`` command line: explain and run feature requests through the service layer.

Results are printed as the services return them; exit code 2 means an error result or nothing could run.
"""

import argparse
import csv
import getpass
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import ConfigError, Settings, load_settings
from mloda.interfaces.service.errors import is_error
from mloda.interfaces.service.planner import Planner
from mloda.interfaces.service.runner import Runner
from mloda.user import PluginCollector

DEFAULT_CONFIG = "mloda.toml"
SIGNING_KEY_ENV = "MLODA_MCP_SIGNING_KEY"
FORMATS = ("json", "jsonl", "csv")
FAILED = 2


def build_services(
    settings: Settings,
    *,
    env: Mapping[str, str],
    plugin_collector: PluginCollector | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[Catalog, Planner, Runner]:
    """Build the catalog, planner and runner for ``settings``, signing plans with the env key if set."""
    key = env.get(SIGNING_KEY_ENV)
    catalog = Catalog.build(settings, plugin_collector)
    planner = Planner(
        settings,
        catalog,
        signing_key=key.encode("utf-8") if key else None,
        principal=getpass.getuser(),
        plugin_collector=plugin_collector,
        clock=time.time if clock is None else clock,
    )
    return catalog, planner, Runner(planner)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mloda", description="Explain and run mloda feature requests.")
    commands = parser.add_subparsers(dest="command", required=True)
    explain = commands.add_parser("explain", help="Plan a feature request and print the plan.")
    run = commands.add_parser("run", help="Plan and execute a feature request and print its preview.")
    for command in (explain, run):
        command.add_argument("request", help="Path to a JSON feature request, or - for stdin.")
        config = command.add_mutually_exclusive_group()
        config.add_argument("--config", help=f"Settings file (default: ./{DEFAULT_CONFIG} if present).")
        config.add_argument("--no-config", action="store_true", help="Ignore any settings file.")
    run.add_argument("--max-rows", type=int, help="Refuse sources with more rows than this.")
    run.add_argument("--format", choices=FORMATS, default="json", help="Output format of the result.")
    return parser


def _parse(argv: Sequence[str] | None, stdout: TextIO, stderr: TextIO) -> argparse.Namespace | int:
    with redirect_stdout(stdout), redirect_stderr(stderr):
        # argparse reports --help and usage errors only by raising SystemExit.
        try:
            return _parser().parse_args(argv)
        except SystemExit as exited:
            return exited.code if isinstance(exited.code, int) else FAILED


def _config_path(args: argparse.Namespace) -> str | None:
    if args.no_config:
        return None
    if args.config is not None:
        return str(args.config)
    return DEFAULT_CONFIG if Path(DEFAULT_CONFIG).is_file() else None


def _settings(args: argparse.Namespace, env: Mapping[str, str]) -> Settings | str:
    overrides: dict[str, Any] = {}
    if args.command == "run":
        overrides["mode"] = "execute"
        if args.max_rows is not None:
            overrides["max_rows"] = args.max_rows
    # load_settings signals an unusable configuration only by raising ConfigError.
    try:
        return load_settings(_config_path(args), env, overrides)
    except ConfigError as error:
        return str(error)


def _read_request(source: str, stdin: TextIO) -> tuple[Any, str | None]:
    """The parsed request and None, or None and an error message that never quotes the request text."""
    if source == "-":
        text = stdin.read()
    else:
        path = Path(source)
        if not path.is_file() or not os.access(path, os.R_OK):
            return None, f"request file not found or not readable: {source}"
        text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        return None, f"request is not valid JSON ({error.msg} at line {error.lineno} column {error.colno})"


def _write_result(result: Mapping[str, Any], output_format: str, stdout: TextIO) -> None:
    tables: list[dict[str, Any]] = result.get("tables", [])
    if is_error(result) or output_format == "json":
        stdout.write(json.dumps(result, indent=2) + "\n")
    elif output_format == "jsonl":
        for table in tables:
            for row in table["preview"]["rows"]:
                stdout.write(json.dumps(row) + "\n")
    else:
        writer = csv.writer(stdout, lineterminator="\n")
        for table in tables:
            header = [column["name"] for column in table["columns"] if column["in_preview"]]
            writer.writerow(header)
            writer.writerows([row.get(name) for name in header] for row in table["preview"]["rows"])


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    plugin_collector: PluginCollector | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    """Run the ``mloda`` command line and return its exit code."""
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = _parse(argv, out, err)
    if isinstance(args, int):
        return args
    settings = _settings(args, env)
    if isinstance(settings, str):
        err.write(f"mloda: error: {settings}\n")
        return FAILED
    request, problem = _read_request(args.request, sys.stdin if stdin is None else stdin)
    if problem is not None:
        err.write(f"mloda: error: {problem}\n")
        return FAILED
    _, planner, runner = build_services(settings, env=env, plugin_collector=plugin_collector, clock=clock)
    result = planner.explain(request)
    output_format = "json"
    if args.command == "run" and not is_error(result):
        result = runner.run(result["plan_id"])
        output_format = args.format
    tables = result.get("tables", [])
    if output_format == "csv" and len(tables) > 1:
        err.write(f"mloda: error: csv output needs a single table, got {len(tables)}; use json or jsonl\n")
        return FAILED
    _write_result(result, output_format, out)
    return FAILED if is_error(result) else 0
