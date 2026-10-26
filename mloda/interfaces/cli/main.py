"""The ``mloda`` command line: explain and run feature requests through the service layer, or serve them over MCP.

Results are printed as the services return them; exit code 2 means an error result or nothing could run,
exit code 3 a completed run that reported failed rows.
"""

import argparse
import csv
import getpass
import ipaddress
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
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
FAILED_ROWS = 3


def build_services(
    settings: Settings,
    *,
    env: Mapping[str, str],
    plugin_collector: PluginCollector | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[Catalog, Planner, Runner]:
    """Build the catalog, planner and runner for ``settings``, signing plans with the env key if set."""
    key = env.get(SIGNING_KEY_ENV)
    principal = getpass.getuser()
    catalog = Catalog.build(settings, plugin_collector, principal=principal)
    now = time.time if clock is None else clock
    planner = Planner(
        settings,
        catalog,
        signing_key=key.encode("utf-8") if key else None,
        principal=principal,
        plugin_collector=plugin_collector,
        clock=now,
    )
    return catalog, planner, Runner(planner, clock=now)


TARGETS = (
    ("targets",),
    {
        "nargs": "+",
        "metavar": "TARGET",
        "help": "Feature names, or one request file (a path, a .json name, or - for stdin).",
    },
)
OPTION = (
    ("-o", "--option"),
    {
        "action": "append",
        "metavar": "KEY=VALUE",
        "help": "Group option for every feature; KEY:=JSON parses JSON, KEY=@FILE reads a file.",
    },
)
CONTEXT_OPTION = (
    ("-c", "--context-option"),
    {
        "action": "append",
        "metavar": "KEY=VALUE",
        "help": "Context option for every feature; same value forms as --option.",
    },
)
COMMANDS: tuple[tuple[str, str, tuple[tuple[tuple[str, ...], dict[str, Any]], ...]], ...] = (
    ("explain", "Plan a feature request and print the plan.", (TARGETS, OPTION, CONTEXT_OPTION)),
    (
        "run",
        "Plan and execute a feature request and print its preview.",
        (
            TARGETS,
            (("--max-rows",), {"type": int, "help": "Refuse sources and results over this, and uncountable reads."}),
            (("--format",), {"choices": FORMATS, "default": "json", "help": "Output format of the result."}),
            (
                ("--full",),
                {"action": "store_true", "help": "Return every row and column, not the classified preview."},
            ),
            OPTION,
            CONTEXT_OPTION,
        ),
    ),
)


def _add_config_options(command: argparse.ArgumentParser) -> None:
    config = command.add_mutually_exclusive_group()
    config.add_argument("--config", help=f"Settings file (default: ./{DEFAULT_CONFIG} if present).")
    config.add_argument("--no-config", action="store_true", help="Ignore any settings file.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mloda", description="Explain, run or serve mloda feature requests.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text, arguments in COMMANDS:
        command = commands.add_parser(name, help=help_text)
        for flags, options in arguments:
            command.add_argument(*flags, **options)
        _add_config_options(command)
    mcp = commands.add_parser("mcp", help="Serve the feature catalog, explain and run to MCP clients.")
    serve = mcp.add_subparsers(dest="mcp_command", required=True).add_parser(
        "serve", help="Serve over stdio or http; settings from the config file, then MLODA_MCP_* env vars."
    )
    _add_config_options(serve)
    serve.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio", help="Transport to serve over (default: stdio)."
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="Address to bind for the http transport (default: 127.0.0.1)."
    )
    serve.add_argument("--port", type=int, default=8000, help="Port to bind for the http transport (default: 8000).")
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


def _serve_mcp(args: argparse.Namespace, env: Mapping[str, str], err: TextIO) -> int:
    """Serve over MCP; the module is looked up at call time so ``mcp`` stays an optional extra."""
    from mloda.interfaces.mcp import main as mcp_main

    if not mcp_main.mcp_available():
        err.write(f"mloda: error: {mcp_main.INSTALL_HINT}\n")
        return FAILED
    settings = _settings(args, env)
    if isinstance(settings, str):
        err.write(f"mloda: error: {settings}\n")
        return FAILED
    if args.transport == "stdio":
        return mcp_main.serve(settings, env=env)
    if not _is_loopback(args.host):
        err.write(
            f"mloda: warning: serving over http on {args.host} with no authentication; the network can reach it.\n"
        )
    return mcp_main.serve(settings, env=env, transport="http", host=args.host, port=args.port)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    # ip_address signals a non-IP host only by raising.
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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


def _is_request_file(targets: Sequence[str]) -> bool:
    return len(targets) == 1 and (targets[0] == "-" or targets[0].endswith(".json") or Path(targets[0]).is_file())


def _parse_pair(pair: str) -> tuple[str, Any, str | None]:
    """Key, value and None, or an error message that never quotes the value."""
    key, separator, text = pair.partition("=")
    as_json = key.endswith(":")
    key = key.removesuffix(":")
    if not separator or not key:
        return "", None, "an option must be written KEY=VALUE"
    if as_json:
        # json.loads signals bad JSON only by raising.
        try:
            return key, json.loads(text), None
        except (ValueError, RecursionError):
            return key, None, f"the value of option '{key}' is not valid JSON"
    if text.startswith("@"):
        path = Path(text[1:])
        if not path.is_file() or not os.access(path, os.R_OK):
            return key, None, f"the file of option '{key}' was not found or is not readable"
        return key, path.read_text(encoding="utf-8", errors="replace"), None
    return key, text, None


def _parse_pairs(pairs: Sequence[str] | None) -> tuple[dict[str, Any], str | None]:
    parsed: dict[str, Any] = {}
    for pair in pairs or ():
        key, value, problem = _parse_pair(pair)
        if problem is not None:
            return {}, problem
        parsed[key] = value
    return parsed, None


def _build_feature_request(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    group, problem = _parse_pairs(args.option)
    if problem is not None:
        return {}, problem
    context, problem = _parse_pairs(args.context_option)
    if problem is not None:
        return {}, problem
    extra = {**({"group_options": group} if group else {}), **({"context_options": context} if context else {})}
    return {"features": [{"name": name, **extra} for name in args.targets]}, None


def _load_request(args: argparse.Namespace, stdin: TextIO) -> tuple[Any, str | None]:
    if not _is_request_file(args.targets):
        return _build_feature_request(args)
    if args.option or args.context_option:
        return None, "--option and --context-option apply to feature names, not to a request file"
    return _read_request(args.targets[0], stdin)


def _write_failed_rows_summary(result: Mapping[str, Any], stderr: TextIO) -> None:
    for entry in result.get("failed_rows", ()):
        rows = ",".join(str(row) for row in entry["rows"])
        stderr.write(
            f"mloda: failed rows: column {entry['column']}, reason {entry['reason']}, "
            f"row_count {entry['row_count']}, rows {rows}\n"
        )


def _write_result(
    result: Mapping[str, Any], output_format: str, stdout: TextIO, full: bool = False, stderr: TextIO | None = None
) -> None:
    tables: list[dict[str, Any]] = result.get("tables", [])
    if is_error(result) or output_format == "json":
        stdout.write(json.dumps(result, indent=2) + "\n")
    elif output_format == "jsonl":
        if stderr is not None:
            _write_failed_rows_summary(result, stderr)
        for table in tables:
            for row in table["rows"] if full else table["preview"]["rows"]:
                stdout.write(json.dumps(row) + "\n")
    else:
        if stderr is not None:
            _write_failed_rows_summary(result, stderr)
        writer = csv.writer(stdout, lineterminator="\n")
        for table in tables:
            header = [column["name"] for column in table["columns"] if full or column["in_preview"]]
            writer.writerow(header)
            rows = table["rows"] if full else table["preview"]["rows"]
            writer.writerows([row.get(name) for name in header] for row in rows)


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
    if args.command == "mcp":
        return _serve_mcp(args, env, err)
    settings = _settings(args, env)
    if isinstance(settings, str):
        err.write(f"mloda: error: {settings}\n")
        return FAILED
    settings = replace(settings, trusted_caller=True)
    request, problem = _load_request(args, sys.stdin if stdin is None else stdin)
    if problem is not None:
        err.write(f"mloda: error: {problem}\n")
        return FAILED
    _, planner, runner = build_services(settings, env=env, plugin_collector=plugin_collector, clock=clock)
    result = planner.explain(request)
    output_format = "json"
    if args.command == "run" and not is_error(result):
        result = runner.run(result["plan_id"], full=args.full)
        output_format = args.format
    tables = result.get("tables", [])
    if output_format == "csv" and len(tables) > 1:
        err.write(f"mloda: error: csv output needs a single table, got {len(tables)}; use json or jsonl\n")
        return FAILED
    _write_result(result, output_format, out, full=args.command == "run" and args.full, stderr=err)
    if is_error(result):
        return FAILED
    return FAILED_ROWS if result.get("failed_rows") else 0
