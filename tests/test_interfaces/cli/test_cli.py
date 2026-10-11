"""The ``mloda`` CLI: explain and run feature requests through the same services the MCP server uses.

``mloda explain`` prints exactly what the service layer returns, so agent sessions can be reproduced locally.
"""

import ast
import base64
import csv
import getpass
import io
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.interfaces.cli.main import build_services, main
from mloda.interfaces.service.catalog import Catalog
from mloda.interfaces.service.config import Settings, load_settings
from mloda.interfaces.service.errors import is_error
from mloda.interfaces.service.planner import Planner
from mloda.interfaces.service.runner import Runner
from mloda.user import PluginCollector
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# The CLI resolves real FeatureGroups and reads source files; a cold cache under xdist load can exceed 10s.
pytestmark = pytest.mark.timeout(30)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_PATH = REPO_ROOT / "mloda" / "interfaces" / "cli" / "main.py"
CLOCK_START = 1_800_000_000.0
ENV: dict[str, str] = {"MLODA_MCP_SIGNING_KEY": "k" * 32}
REQUEST: dict[str, Any] = {"features": ["customer_id", "income", "segment"]}
CUSTOMER_IDS = [f"C9Q4{index:03d}" for index in range(1, 6)]
PREVIEW_ROWS = [
    {"income": 10, "segment": "a"},
    {"income": 20, "segment": "b"},
    {"income": 30, "segment": "a"},
    {"income": 40, "segment": "b"},
    {"income": 50, "segment": "a"},
]
SECRET = "SECRET_mcp_cli_7f3a"  # nosec B105
CONFIG = """
[mcp]
mode = "plan"

[mcp.classification]
customer_id = "personal"
income = "internal"
segment = "public"
order_id = "public"

[mcp.sources.sales]
files = ["data/sales.csv"]
"""
ORDERS_SOURCE = """
[mcp.sources.orders]
files = ["data/orders.csv"]
"""


def _clock() -> float:
    return CLOCK_START


def _base() -> PluginCollector:
    return PluginCollector.enabled_feature_groups({ReadFileFeature})


@dataclass(frozen=True)
class _Outcome:
    code: int
    out: str
    err: str

    def json(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.out)
        return result


def _cli(argv: Sequence[str], *, env: Mapping[str, str] = ENV, stdin_text: str = "") -> _Outcome:
    out, err = io.StringIO(), io.StringIO()
    try:
        code = main(
            list(argv),
            env=env,
            stdin=io.StringIO(stdin_text),
            stdout=out,
            stderr=err,
            plugin_collector=_base(),
            clock=_clock,
        )
    except SystemExit as exited:
        code = exited.code if isinstance(exited.code, int) else 1
    return _Outcome(code, out.getvalue(), err.getvalue())


def _write_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    rows = [f"{CUSTOMER_IDS[i - 1]},{10 * i},{'a' if i % 2 else 'b'}" for i in range(1, 6)]
    (data / "sales.csv").write_text("customer_id,income,segment\n" + "\n".join(rows) + "\n")
    (data / "orders.csv").write_text("order_id,amount\n1,3\n2,4\n3,5\n")


def _config(tmp_path: Path, text: str = CONFIG, name: str = "mloda.toml") -> Path:
    _write_data(tmp_path)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _request_file(tmp_path: Path, request: Any = REQUEST) -> Path:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def _services(config: Path | None, overrides: Mapping[str, Any] | None = None) -> tuple[Catalog, Planner, Runner]:
    settings = load_settings(config, ENV, overrides)
    services: tuple[Catalog, Planner, Runner] = build_services(
        settings, env=ENV, plugin_collector=_base(), clock=_clock
    )
    return services


def _expected_explain(config: Path | None, request: Any = REQUEST) -> dict[str, Any]:
    _, planner, _ = _services(config)
    return planner.explain(request)


def _expected_run(config: Path, request: Any = REQUEST) -> dict[str, Any]:
    _, planner, runner = _services(config, {"mode": "execute"})
    explained = planner.explain(request)
    assert not is_error(explained), explained
    return runner.run(explained["plan_id"])


def _without_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(result))
    stripped.pop("audit_id", None)
    if isinstance(stripped.get("error"), dict):
        stripped["error"].pop("audit_id", None)
    return dict(stripped)


def _error_code(outcome: _Outcome) -> str:
    result = outcome.json()
    assert is_error(result), result
    code: str = result["error"]["code"]
    return code


def _plan_payload(plan_id: str) -> dict[str, Any]:
    segment = plan_id.split(".")[1]
    payload: dict[str, Any] = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    return payload


def _assert_no_paths(text: str, tmp_path: Path) -> None:
    assert str(tmp_path) not in text
    assert "sales.csv" not in text


def _assert_no_withheld_values(outcome: _Outcome) -> None:
    for customer_id in CUSTOMER_IDS:
        assert customer_id not in outcome.out
        assert customer_id not in outcome.err


class TestExplainParity:
    def test_explain_file_prints_exactly_the_service_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", str(_request_file(tmp_path)), "--config", str(config)])

        assert outcome.code == 0, outcome.err
        assert outcome.json() == _expected_explain(config)
        assert "plan_id" in outcome.json()

    def test_explain_stdin_prints_exactly_the_service_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", "-", "--config", str(config)], stdin_text=json.dumps(REQUEST))

        assert outcome.code == 0, outcome.err
        assert outcome.json() == _expected_explain(config)

    def test_error_result_matches_the_service_apart_from_audit_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)
        request = {"features": ["customer_id", "incme"]}

        outcome = _cli(["explain", str(_request_file(tmp_path, request)), "--config", str(config)])

        assert outcome.code == 2
        assert _without_audit(outcome.json()) == _without_audit(_expected_explain(config, request))
        assert _error_code(outcome) == "validation_failed"

    def test_output_is_one_json_document_without_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", str(_request_file(tmp_path)), "--config", str(config)])

        assert isinstance(outcome.json(), dict)
        _assert_no_paths(outcome.out, tmp_path)


class TestExplainErrors:
    @pytest.mark.parametrize(
        ("request_value", "code"),
        [
            ({"features": ["incme"]}, "validation_failed"),
            ([1, 2], "validation_failed"),
            ({"features": [{"name": "income", "options": {RESERVED_READER_OPTION_KEY: SECRET}}]}, "policy_denied"),
        ],
    )
    def test_error_result_goes_to_stdout_with_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_value: Any, code: str
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", str(_request_file(tmp_path, request_value)), "--config", str(config)])

        assert outcome.code == 2
        assert _error_code(outcome) == code
        assert SECRET not in outcome.out + outcome.err

    def test_env_settings_apply_on_top_of_the_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(
            ["explain", str(_request_file(tmp_path)), "--config", str(config)],
            env={**ENV, "MLODA_MCP_MAX_ROWS": "3"},
        )

        assert outcome.code == 2
        assert _error_code(outcome) == "limit_exceeded"


class TestRequestInput:
    def test_missing_request_file_is_a_stderr_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", str(tmp_path / "nope.json"), "--config", str(config)])

        assert outcome.code == 2
        assert outcome.out == ""
        assert outcome.err.strip()

    def test_invalid_json_file_is_a_stderr_error_without_its_contents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)
        path = tmp_path / "request.json"
        path.write_text('{"features": ["' + SECRET + '"', encoding="utf-8")

        outcome = _cli(["explain", str(path), "--config", str(config)])

        assert outcome.code == 2
        assert outcome.out == ""
        assert outcome.err.strip()
        assert SECRET not in outcome.err

    def test_invalid_json_on_stdin_is_a_stderr_error_without_its_contents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", "-", "--config", str(config)], stdin_text=SECRET + " {")

        assert outcome.code == 2
        assert outcome.out == ""
        assert outcome.err.strip()
        assert SECRET not in outcome.err


class TestConfigResolution:
    def test_mloda_toml_in_the_working_directory_is_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _request_file(tmp_path)
        monkeypatch.chdir(tmp_path)

        outcome = _cli(["explain", "request.json"])

        assert outcome.code == 0, outcome.err
        assert outcome.json() == _expected_explain(config)

    def test_without_a_config_the_defaults_apply(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_data(tmp_path)
        request = _request_file(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", str(request)])

        assert outcome.code == 2
        assert _without_audit(outcome.json()) == _without_audit(_expected_explain(None))

    def test_no_config_ignores_mloda_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _config(tmp_path)
        request = _request_file(tmp_path)
        monkeypatch.chdir(tmp_path)

        outcome = _cli(["explain", str(request), "--no-config"])

        assert outcome.code == 2
        assert _without_audit(outcome.json()) == _without_audit(_expected_explain(None))

    def test_no_config_ignores_a_broken_mloda_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _config(tmp_path, "[mcp]\nnot_a_key = 1\n")
        request = _request_file(tmp_path)
        monkeypatch.chdir(tmp_path)

        outcome = _cli(["explain", str(request), "--no-config"])

        assert outcome.code == 2
        assert _error_code(outcome) == "validation_failed"

    def test_broken_mloda_toml_in_the_working_directory_is_a_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _config(tmp_path, "[mcp]\nnot_a_key = 1\n")
        request = _request_file(tmp_path)
        monkeypatch.chdir(tmp_path)

        outcome = _cli(["explain", str(request)])

        assert outcome.code == 2
        assert outcome.out == ""
        assert "not_a_key" in outcome.err

    @pytest.mark.parametrize("command", ["explain", "run"])
    @pytest.mark.parametrize("problem", ["missing", "bad_toml", "unknown_key"])
    def test_config_error_is_stderr_only_with_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, problem: str
    ) -> None:
        texts = {"bad_toml": "[mcp\nmode = ", "unknown_key": "[mcp]\nmood = 'plan'\n"}
        config = tmp_path / "missing.toml" if problem == "missing" else _config(tmp_path, texts[problem], "cfg.toml")
        request = _request_file(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli([command, str(request), "--config", str(config)])

        assert outcome.code == 2
        assert outcome.out == ""
        assert outcome.err.strip()

    def test_config_and_no_config_are_exclusive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["explain", str(_request_file(tmp_path)), "--config", str(config), "--no-config"])

        assert outcome.code == 2
        assert outcome.out == ""


class TestRun:
    def test_run_prints_the_service_run_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config)])

        assert outcome.code == 0, outcome.err
        result = outcome.json()
        assert result["status"] == "completed"
        assert result["audit_id"].startswith("run_")
        assert _without_audit(result) == _without_audit(_expected_run(config))

    def test_run_executes_even_when_the_config_is_in_plan_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config), "--format", "json"])

        assert outcome.code == 0, outcome.err
        (table,) = outcome.json()["tables"]
        assert table["row_count"] == 5
        assert table["preview"]["rows"] == PREVIEW_ROWS

    def test_run_reads_the_request_from_stdin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", "-", "--config", str(config)], stdin_text=json.dumps(REQUEST))

        assert outcome.code == 0, outcome.err
        assert outcome.json()["status"] == "completed"

    @pytest.mark.parametrize("output_format", ["json", "jsonl", "csv"])
    def test_max_rows_over_the_source_is_limit_exceeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)
        argv = ["run", str(_request_file(tmp_path)), "--config", str(config), "--max-rows", "3"]

        outcome = _cli([*argv, "--format", output_format])

        assert outcome.code == 2
        assert _error_code(outcome) == "limit_exceeded"
        details = outcome.json()["error"]["details"]
        assert {"source": "sales", "reason": "over_limit", "rows": 5, "max_rows": 3} in details
        _assert_no_withheld_values(outcome)

    def test_max_rows_within_the_source_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config), "--max-rows", "10"])

        assert outcome.code == 0, outcome.err
        assert outcome.json()["status"] == "completed"

    def test_explain_error_is_printed_and_nothing_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)
        request = _request_file(tmp_path, {"features": ["incme"]})

        outcome = _cli(["run", str(request), "--config", str(config)])

        assert outcome.code == 2
        assert _error_code(outcome) == "validation_failed"
        assert "tables" not in outcome.json()

    @pytest.mark.parametrize("extra", [["--max-rows", "abc"], ["--format", "xml"]])
    def test_bad_run_option_exits_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config), *extra])

        assert outcome.code == 2
        assert outcome.out == ""


class TestRunFormats:
    def test_jsonl_prints_one_preview_row_per_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config), "--format", "jsonl"])

        assert outcome.code == 0, outcome.err
        assert [json.loads(line) for line in outcome.out.splitlines()] == PREVIEW_ROWS

    def test_jsonl_prints_rows_of_every_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path, CONFIG + ORDERS_SOURCE)
        _elsewhere(tmp_path, monkeypatch)
        request = _request_file(tmp_path, {"features": ["income", "order_id"]})

        outcome = _cli(["run", str(request), "--config", str(config), "--format", "jsonl"])

        assert outcome.code == 0, outcome.err
        rows = [json.loads(line) for line in outcome.out.splitlines()]
        expected = [{"income": 10 * i} for i in range(1, 6)] + [{"order_id": i} for i in range(1, 4)]
        assert sorted(rows, key=json.dumps) == sorted(expected, key=json.dumps)

    def test_csv_prints_header_and_preview_rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config), "--format", "csv"])

        assert outcome.code == 0, outcome.err
        rows = list(csv.reader(io.StringIO(outcome.out)))
        assert rows == [["income", "segment"]] + [[str(row["income"]), str(row["segment"])] for row in PREVIEW_ROWS]

    def test_csv_with_more_than_one_table_is_a_stderr_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, CONFIG + ORDERS_SOURCE)
        _elsewhere(tmp_path, monkeypatch)
        request = _request_file(tmp_path, {"features": ["income", "order_id"]})

        outcome = _cli(["run", str(request), "--config", str(config), "--format", "csv"])

        assert outcome.code == 2
        assert outcome.out == ""
        assert outcome.err.strip()

    @pytest.mark.parametrize("output_format", ["json", "jsonl", "csv"])
    def test_withheld_values_never_appear(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["run", str(_request_file(tmp_path)), "--config", str(config), "--format", output_format])

        assert outcome.code == 0, outcome.err
        _assert_no_withheld_values(outcome)
        _assert_no_paths(outcome.out, tmp_path)


class TestBuildServices:
    def _settings(self, tmp_path: Path, mode: str = "plan") -> Settings:
        return load_settings(_config(tmp_path), ENV, {"mode": mode})

    def test_returns_catalog_planner_and_runner_over_the_settings(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)

        catalog, planner, runner = build_services(settings, env=ENV, plugin_collector=_base(), clock=_clock)

        assert isinstance(catalog, Catalog)
        assert isinstance(planner, Planner)
        assert isinstance(runner, Runner)
        assert planner.settings == settings

    def test_signing_key_from_env_makes_plan_ids_portable(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        _, first, _ = build_services(settings, env=ENV, plugin_collector=_base(), clock=_clock)
        _, second, _ = build_services(settings, env=ENV, plugin_collector=_base(), clock=_clock)

        plan_id = first.explain(REQUEST)["plan_id"]

        assert not is_error(second.verify_plan_id(plan_id))

    def test_without_a_signing_key_each_build_has_its_own_key(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        _, first, _ = build_services(settings, env={}, plugin_collector=_base(), clock=_clock)
        _, second, _ = build_services(settings, env={}, plugin_collector=_base(), clock=_clock)

        plan_id = first.explain(REQUEST)["plan_id"]

        assert not is_error(first.verify_plan_id(plan_id))
        assert second.verify_plan_id(plan_id)["error"]["code"] == "plan_invalid"

    def test_principal_is_the_local_user(self, tmp_path: Path) -> None:
        _, planner, _ = build_services(self._settings(tmp_path), env=ENV, plugin_collector=_base(), clock=_clock)

        plan_id = planner.explain(REQUEST)["plan_id"]

        assert _plan_payload(plan_id)["sub"] == getpass.getuser()

    def test_clock_is_passed_to_the_planner(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        _, planner, _ = build_services(settings, env=ENV, plugin_collector=_base(), clock=_clock)

        plan_id = planner.explain(REQUEST)["plan_id"]

        assert _plan_payload(plan_id)["exp"] == int(CLOCK_START) + settings.plan_ttl_seconds

    def test_runner_runs_plans_of_its_planner(self, tmp_path: Path) -> None:
        _, planner, runner = build_services(
            self._settings(tmp_path, "execute"), env=ENV, plugin_collector=_base(), clock=_clock
        )

        result = runner.run(planner.explain(REQUEST)["plan_id"])

        assert result["status"] == "completed"
        assert result["tables"][0]["preview"]["rows"] == PREVIEW_ROWS


class TestUsage:
    def _printed(self, outcome: _Outcome, capsys: pytest.CaptureFixture[str]) -> str:
        captured = capsys.readouterr()
        return outcome.out + outcome.err + captured.out + captured.err

    def test_help_prints_usage_and_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        outcome = _cli(["--help"])

        assert outcome.code == 0
        printed = self._printed(outcome, capsys)
        assert "usage" in printed.lower()
        assert "explain" in printed
        assert "run" in printed

    def test_subcommand_help_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        outcome = _cli(["run", "--help"])

        assert outcome.code == 0
        printed = self._printed(outcome, capsys)
        assert "--max-rows" in printed
        assert "--format" in printed

    def test_no_subcommand_prints_usage_and_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        outcome = _cli([])

        assert outcome.code == 2
        assert "usage" in self._printed(outcome, capsys).lower()


class TestPackaging:
    def test_console_script_is_declared(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            data = tomllib.load(handle)

        assert data["project"]["scripts"]["mloda"] == "mloda.interfaces.cli.main:main"

    def test_cli_module_does_not_import_mcp_at_import_time(self) -> None:
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))

        modules: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
                modules.append(node.module)

        assert not [name for name in modules if name == "mcp" or name.startswith("mcp.")]
