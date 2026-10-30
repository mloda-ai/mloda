"""``mloda-mcp`` and ``mloda mcp serve``: resolve settings like the CLI, then hand them to ``serve``.

Without the ``mcp`` extra both entry points print the install hint and exit 2.
"""

import ast
import importlib.util
import io
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest

import mloda.interfaces.cli.main as cli_main
import mloda.interfaces.mcp.main as mcp_main
from mloda.interfaces.service.config import Settings, load_settings
from tests.helpers.docs_scope import shipped_feature_group_registry

# serve builds the catalog over every loaded FeatureGroup; a cold cache under xdist load can exceed 10s.
pytestmark = pytest.mark.timeout(30)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_PATH = REPO_ROOT / "mloda" / "interfaces" / "mcp" / "main.py"
ENV: dict[str, str] = {"MLODA_MCP_SIGNING_KEY": "k" * 32}
INSTALL_HINT = "pip install 'mloda[mcp]'"
CONFIG = """
[mcp]
mode = "plan"

[mcp.sources.sales]
files = ["data/sales.csv"]
"""
BROKEN = {"bad_toml": "[mcp\nmode = ", "unknown_key": "[mcp]\nmood = 'plan'\n"}


@dataclass(frozen=True)
class _Outcome:
    code: int
    out: str
    err: str


@dataclass(frozen=True)
class _Served:
    settings: Settings
    env: dict[str, str]


def _fake_serve(monkeypatch: pytest.MonkeyPatch, code: int = 0) -> list[_Served]:
    """Replace serve with a recorder and report the extra as installed, so no server starts."""
    calls: list[_Served] = []

    def fake_serve(settings: Settings, *, env: Mapping[str, str]) -> int:
        calls.append(_Served(settings, dict(env)))
        return code

    monkeypatch.setattr(mcp_main, "serve", fake_serve)
    monkeypatch.setattr(mcp_main, "mcp_available", lambda: True)
    return calls


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[_Served]:
    return _fake_serve(monkeypatch)


def _exit_code(exited: SystemExit) -> int:
    return exited.code if isinstance(exited.code, int) else 1


def _mcp(argv: Sequence[str], *, env: Mapping[str, str] | None = ENV) -> _Outcome:
    err = io.StringIO()
    try:
        code = mcp_main.main(list(argv), env=env, stderr=err)
    except SystemExit as exited:
        code = _exit_code(exited)
    return _Outcome(code, "", err.getvalue())


def _cli(argv: Sequence[str], *, env: Mapping[str, str] = ENV) -> _Outcome:
    out, err = io.StringIO(), io.StringIO()
    try:
        code = cli_main.main(list(argv), env=env, stdin=io.StringIO(), stdout=out, stderr=err)
    except SystemExit as exited:
        code = _exit_code(exited)
    return _Outcome(code, out.getvalue(), err.getvalue())


def _config(tmp_path: Path, text: str = CONFIG, name: str = "mloda.toml") -> Path:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "sales.csv").write_text("customer_id,income\nC1,10\n")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def _top_level_imports(path: Path) -> list[str]:
    modules: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            modules.append(node.module)
    return modules


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    operands = [test.left, *test.comparators]
    names = [operand.id for operand in operands if isinstance(operand, ast.Name)]
    constants = [operand.value for operand in operands if isinstance(operand, ast.Constant)]
    return names == ["__name__"] and constants == ["__main__"] and isinstance(test.ops[0], ast.Eq)


class TestMcpMainSettings:
    def test_config_file_settings_and_env_are_served(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp(["--config", str(config)])

        assert outcome.code == 0, outcome.err
        (call,) = served
        assert call.settings == load_settings(config, ENV)
        assert call.env == ENV
        assert set(call.settings.sources) == {"sales"}

    def test_env_settings_apply_on_top_of_the_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp(["--config", str(config)], env={**ENV, "MLODA_MCP_MODE": "execute"})

        assert outcome.code == 0, outcome.err
        assert served[0].settings.mode == "execute"

    def test_mloda_toml_in_the_working_directory_is_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        config = _config(tmp_path)
        monkeypatch.chdir(tmp_path)

        outcome = _mcp([])

        assert outcome.code == 0, outcome.err
        assert served[0].settings == load_settings(config, ENV)

    def test_without_a_config_the_defaults_apply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp([])

        assert outcome.code == 0, outcome.err
        assert served[0].settings == load_settings(None, ENV)

    def test_no_config_ignores_mloda_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _config(tmp_path, "[mcp]\nnot_a_key = 1\n")
        monkeypatch.chdir(tmp_path)

        outcome = _mcp(["--no-config"])

        assert outcome.code == 0, outcome.err
        assert served[0].settings == load_settings(None, ENV)

    def test_default_env_is_the_process_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _elsewhere(tmp_path, monkeypatch)
        monkeypatch.setenv("MLODA_MCP_MODE", "execute")

        outcome = _mcp(["--no-config"], env=None)

        assert outcome.code == 0, outcome.err
        assert served[0].settings.mode == "execute"
        assert served[0].env["MLODA_MCP_MODE"] == "execute"

    def test_returns_the_exit_code_of_serve(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _elsewhere(tmp_path, monkeypatch)
        calls = _fake_serve(monkeypatch, code=3)

        assert _mcp(["--no-config"]).code == 3
        assert len(calls) == 1


class TestMcpMainErrors:
    @pytest.mark.parametrize("problem", ["missing", "bad_toml", "unknown_key"])
    def test_config_error_is_a_stderr_message_with_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served], problem: str
    ) -> None:
        config = tmp_path / "missing.toml" if problem == "missing" else _config(tmp_path, BROKEN[problem], "cfg.toml")
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp(["--config", str(config)])

        assert outcome.code == 2
        assert outcome.err.strip()
        assert served == []

    def test_config_and_no_config_are_exclusive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp(["--config", str(config), "--no-config"])

        assert outcome.code == 2
        assert served == []

    def test_missing_mcp_extra_prints_the_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _elsewhere(tmp_path, monkeypatch)
        monkeypatch.setattr(mcp_main, "mcp_available", lambda: False)

        outcome = _mcp(["--no-config"])

        assert outcome.code == 2
        assert INSTALL_HINT in outcome.err
        assert served == []


class TestMcpAvailable:
    def _patch_find_spec(self, monkeypatch: pytest.MonkeyPatch, found: bool) -> list[str]:
        asked: list[str] = []
        original = importlib.util.find_spec

        def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
            if name == "mcp":
                asked.append(name)
                return ModuleSpec("mcp", None) if found else None
            return original(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        return asked

    def test_false_when_mcp_cannot_be_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked = self._patch_find_spec(monkeypatch, found=False)

        assert mcp_main.mcp_available() is False
        assert asked == ["mcp"]

    def test_true_when_mcp_can_be_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked = self._patch_find_spec(monkeypatch, found=True)

        assert mcp_main.mcp_available() is True
        assert asked == ["mcp"]


class TestServe:
    def test_serve_runs_the_server_over_stdio(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mcpserver = pytest.importorskip("mcp.server.mcpserver")
        runs: list[tuple[Any, str]] = []

        def fake_run(self: Any, transport: str = "stdio", **kwargs: Any) -> None:
            runs.append((self, transport))

        monkeypatch.setattr(mcpserver.MCPServer, "run", fake_run)
        registry = shipped_feature_group_registry()
        monkeypatch.setattr("mloda.core.api.plugin_docs.feature_group_subclasses", registry)
        settings = load_settings(_config(tmp_path), ENV)

        assert mcp_main.serve(settings, env=ENV) == 0

        ((server, transport),) = runs
        assert transport == "stdio"
        assert isinstance(server, mcpserver.MCPServer)
        assert "mloda_explain_request" in (server.instructions or "")


class TestMcpModule:
    def test_main_module_does_not_import_mcp_at_import_time(self) -> None:
        modules = _top_level_imports(MAIN_PATH)

        assert not [name for name in modules if name == "mcp" or name.startswith("mcp.")]
        assert "mloda.interfaces.mcp.server" not in modules

    def test_main_module_runs_as_a_script(self) -> None:
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))

        assert any(_is_main_guard(node) for node in tree.body)

    def test_help_prints_usage_and_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        outcome = _mcp(["--help"])

        assert outcome.code == 0
        printed = capsys.readouterr().out + outcome.err
        assert "usage" in printed.lower()
        assert "--config" in printed
        assert "--no-config" in printed

    def test_console_script_is_declared(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            data = tomllib.load(handle)

        assert data["project"]["scripts"]["mloda-mcp"] == "mloda.interfaces.mcp.main:main"


class TestCliMcpServe:
    def test_config_file_settings_are_served(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["mcp", "serve", "--config", str(config)])

        assert outcome.code == 0, outcome.err
        (call,) = served
        assert call.settings == load_settings(config, ENV)
        assert call.settings.mode == "plan"
        assert call.env == ENV

    def test_env_mode_applies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]) -> None:
        config = _config(tmp_path)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["mcp", "serve", "--config", str(config)], env={**ENV, "MLODA_MCP_MODE": "execute"})

        assert outcome.code == 0, outcome.err
        assert served[0].settings.mode == "execute"

    def test_mloda_toml_in_the_working_directory_is_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        config = _config(tmp_path)
        monkeypatch.chdir(tmp_path)

        outcome = _cli(["mcp", "serve"])

        assert outcome.code == 0, outcome.err
        assert served[0].settings == load_settings(config, ENV)

    def test_no_config_ignores_mloda_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _config(tmp_path, "[mcp]\nnot_a_key = 1\n")
        monkeypatch.chdir(tmp_path)

        outcome = _cli(["mcp", "serve", "--no-config"])

        assert outcome.code == 0, outcome.err
        assert served[0].settings == load_settings(None, ENV)

    def test_returns_the_exit_code_of_serve(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _elsewhere(tmp_path, monkeypatch)
        calls = _fake_serve(monkeypatch, code=3)

        assert _cli(["mcp", "serve", "--no-config"]).code == 3
        assert len(calls) == 1

    @pytest.mark.parametrize("problem", ["missing", "bad_toml", "unknown_key"])
    def test_config_error_is_stderr_only_with_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served], problem: str
    ) -> None:
        config = tmp_path / "missing.toml" if problem == "missing" else _config(tmp_path, BROKEN[problem], "cfg.toml")
        _elsewhere(tmp_path, monkeypatch)

        outcome = _cli(["mcp", "serve", "--config", str(config)])

        assert outcome.code == 2
        assert outcome.out == ""
        assert outcome.err.strip()
        assert served == []

    def test_missing_mcp_extra_prints_the_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _elsewhere(tmp_path, monkeypatch)
        monkeypatch.setattr(mcp_main, "mcp_available", lambda: False)

        outcome = _cli(["mcp", "serve", "--no-config"])

        assert outcome.code == 2
        assert INSTALL_HINT in outcome.err
        assert served == []

    def test_help_lists_the_mcp_command(self) -> None:
        outcome = _cli(["--help"])

        assert outcome.code == 0
        assert "mcp" in outcome.out

    def test_mcp_serve_help_exits_0(self) -> None:
        outcome = _cli(["mcp", "serve", "--help"])

        assert outcome.code == 0
        assert "--config" in outcome.out
        assert "--no-config" in outcome.out


@dataclass(frozen=True)
class _Uvicorn:
    app: Any
    host: str
    port: int


def _fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> list[_Uvicorn]:
    """Capture uvicorn.run and replace create_app with a sentinel, so nothing is built or bound."""
    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    runs: list[_Uvicorn] = []
    app = object()

    def fake_run(application: Any, *, host: str, port: int, **kwargs: Any) -> None:
        runs.append(_Uvicorn(application, host, port))

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr("mloda.interfaces.mcp.server.create_app", lambda settings, *, env, host="127.0.0.1": app)
    monkeypatch.setattr(mcp_main, "mcp_available", lambda: True)
    runs.append(_Uvicorn(app, "", 0))
    return runs


class TestHttpTransport:
    @pytest.mark.parametrize("entry", ["mcp_main", "cli"])
    def test_parser_accepts_transport_host_and_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
    ) -> None:
        runs = _fake_uvicorn(monkeypatch)
        _elsewhere(tmp_path, monkeypatch)
        flags = ["--no-config", "--transport", "http", "--host", "127.0.0.1", "--port", "9123"]

        outcome = _mcp(flags) if entry == "mcp_main" else _cli(["mcp", "serve", *flags])

        assert outcome.code == 0, outcome.err
        assert runs[1:] == [_Uvicorn(runs[0].app, "127.0.0.1", 9123)]

    def test_defaults_are_loopback_port_8000(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        runs = _fake_uvicorn(monkeypatch)
        _elsewhere(tmp_path, monkeypatch)

        assert _mcp(["--no-config", "--transport", "http"]).code == 0
        assert runs[1:] == [_Uvicorn(runs[0].app, "127.0.0.1", 8000)]

    def test_serve_http_runs_the_created_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runs = _fake_uvicorn(monkeypatch)

        code = mcp_main.serve(load_settings(None, ENV), env=ENV, transport="http", host="127.0.0.1", port=8123)

        assert code == 0
        assert runs[1:] == [_Uvicorn(runs[0].app, "127.0.0.1", 8123)]

    def test_stdio_is_the_default_and_never_starts_uvicorn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: list[_Served]
    ) -> None:
        _elsewhere(tmp_path, monkeypatch)
        runs = _fake_uvicorn(monkeypatch)
        monkeypatch.setattr(mcp_main, "serve", lambda settings, *, env: 0)

        assert _mcp(["--no-config"]).code == 0
        assert runs[1:] == []

    def test_unknown_transport_is_a_usage_error(self, served: list[_Served]) -> None:
        assert _mcp(["--no-config", "--transport", "ws"]).code == 2
        assert served == []

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5"])  # nosec B104
    def test_non_loopback_host_warns_about_missing_authentication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
    ) -> None:
        runs = _fake_uvicorn(monkeypatch)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp(["--no-config", "--transport", "http", "--host", host])

        assert outcome.code == 0, outcome.err
        assert "no authentication" in outcome.err
        assert runs[1].host == host

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_host_does_not_warn(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str) -> None:
        _fake_uvicorn(monkeypatch)
        _elsewhere(tmp_path, monkeypatch)

        outcome = _mcp(["--no-config", "--transport", "http", "--host", host])

        assert outcome.code == 0, outcome.err
        assert "no authentication" not in outcome.err
