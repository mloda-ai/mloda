"""Stdio smoke tests: the real ``mloda-mcp`` process serves plan tools and, in execute mode, runs a plan.

Spawns a subprocess, so it is marked ``mcp``: deselected by the default tox run, run by ``tox -e mcp``.
"""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.types import TextContent  # noqa: E402

import mloda.interfaces.service.config as config_module  # noqa: E402

pytestmark = [pytest.mark.mcp, pytest.mark.timeout(60), pytest.mark.anyio]

PLAN_TOOLS = {"mloda_search_features", "mloda_describe_feature", "mloda_explain_request"}
CONFIG = """
[mcp]
mode = "plan"

[mcp.sources.sales]
files = ["data/sales.csv"]
"""
EXECUTE_CONFIG = """
[mcp]
mode = "execute"

[mcp.classification]
customer_id = "public"
income = "internal"

[mcp.sources.sales]
files = ["data/sales.csv"]
"""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _import_root() -> str:
    """The directory the test process imports mloda from, so the subprocess runs the same code."""
    return str(Path(config_module.__file__).resolve().parents[3])


def _config(tmp_path: Path, text: str = CONFIG) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "sales.csv").write_text("customer_id,income\nC1,10\nC2,20\n")
    path = tmp_path / "mloda.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _parameters(tmp_path: Path, text: str = CONFIG) -> StdioServerParameters:
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    python_path = os.pathsep.join(part for part in (_import_root(), os.environ.get("PYTHONPATH", "")) if part)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mloda.interfaces.mcp.main", "--config", str(_config(tmp_path, text))],
        env={"MLODA_MCP_SIGNING_KEY": "k" * 32, "PYTHONPATH": python_path},
        cwd=str(cwd),
    )


async def test_stdio_server_lists_plan_tools_and_finds_the_source(tmp_path: Path) -> None:
    async with Client(_parameters(tmp_path)) as client:
        tools = await client.list_tools()
        result = await client.call_tool("mloda_search_features", {"query": "", "kinds": ["source"]})

    assert {tool.name for tool in tools.tools} == PLAN_TOOLS
    assert result.is_error is False
    structured = result.structured_content
    assert isinstance(structured, dict)
    assert [item["alias"] for item in structured["items"]] == ["sales"]
    texts = [item.text for item in result.content if isinstance(item, TextContent)]
    for text in [*texts, json.dumps(structured)]:
        assert str(tmp_path) not in text
        assert "sales.csv" not in text


async def test_stdio_server_in_execute_mode_explains_and_runs_a_plan(tmp_path: Path) -> None:
    async with Client(_parameters(tmp_path, EXECUTE_CONFIG)) as client:
        tools = await client.list_tools()
        explained = await client.call_tool(
            "mloda_explain_request", {"request": {"features": ["customer_id", "income"]}}
        )
        assert explained.is_error is False
        assert isinstance(explained.structured_content, dict)
        plan_id = explained.structured_content["plan_id"]
        ran = await client.call_tool("mloda_run_request", {"plan_id": plan_id})

    assert {tool.name for tool in tools.tools} == PLAN_TOOLS | {"mloda_run_request"}
    assert ran.is_error is False
    result = ran.structured_content
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    (table,) = result["tables"]
    assert table["row_count"] == 2
    assert table["columns"] == [
        {"name": "customer_id", "classification": "public", "in_preview": True},
        {"name": "income", "classification": "internal", "in_preview": True},
    ]
    assert table["preview"] == {
        "untrusted": True,
        "truncated": False,
        "rows": [{"customer_id": "C1", "income": 10}, {"customer_id": "C2", "income": 20}],
    }
    for call in (explained, ran):
        texts = [item.text for item in call.content if isinstance(item, TextContent)]
        for text in [*texts, json.dumps(call.structured_content)]:
            assert str(tmp_path) not in text
            assert "sales.csv" not in text
