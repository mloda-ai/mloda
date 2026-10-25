"""MCP server over the shared services: tools, schema resource, result envelope, error boundary and audit log.

Every call goes through the SDK's in-process client; results must be exactly what the services return.
"""

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import jsonschema  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.types import CallToolResult, TextContent, TextResourceContents, Tool  # noqa: E402

from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY  # noqa: E402
from mloda.interfaces.cli.main import build_services  # noqa: E402
from mloda.interfaces.mcp.server import SCHEMA_URI, build_server  # noqa: E402
from mloda.interfaces.service.catalog import Catalog  # noqa: E402
from mloda.interfaces.service.config import Settings, load_settings  # noqa: E402
from mloda.interfaces.service.planner import Planner  # noqa: E402
from mloda.interfaces.service.request import MAX_FEATURES, SCHEMA_ID, request_schema  # noqa: E402
from mloda.interfaces.service.runner import Runner  # noqa: E402
from mloda.user import PluginCollector  # noqa: E402
from tests.test_interfaces.source_groups import FAILED, FILE_SOURCE_GROUPS, RunnerFailedRowsMcp5  # noqa: E402

# The tools resolve real FeatureGroups and read source files; a cold cache under xdist load can exceed 10s.
pytestmark = [pytest.mark.timeout(30), pytest.mark.anyio]

CLOCK_START = 1_800_000_000.0
ENV: dict[str, str] = {"MLODA_MCP_SIGNING_KEY": "k" * 32}
LOGGER = "mloda.interfaces.mcp"
SEARCH = "mloda_search_features"
DESCRIBE = "mloda_describe_feature"
EXPLAIN = "mloda_explain_request"
RUN = "mloda_run_request"
PLAN_TOOLS = {SEARCH, DESCRIBE, EXPLAIN}
REQUEST: dict[str, Any] = {"features": ["customer_id", "income", "segment"]}
CUSTOMER_IDS = [f"C9M7{index:03d}" for index in range(1, 6)]
PREVIEW_ROWS = [
    {"income": 10, "segment": "a"},
    {"income": 20, "segment": "b"},
    {"income": 30, "segment": "a"},
    {"income": 40, "segment": "b"},
    {"income": 50, "segment": "a"},
]
SECRET_OPTION = "SECRET_OPT_9"  # nosec B105
LEAK_PATH = "/tmp/leak-xyz"  # nosec B108
LEAK = f"secret {LEAK_PATH}"
CONFIG = """
[mcp]
mode = "plan"

[mcp.classification]
customer_id = "personal"
income = "internal"
segment = "public"

[mcp.sources.sales]
files = ["data/sales.csv"]
"""
# One call per tool and outcome, valid in execute mode.
CALLS: list[tuple[str, dict[str, Any]]] = [
    (SEARCH, {"query": ""}),
    (SEARCH, {"query": "", "kinds": ["source"], "limit": 1}),
    (DESCRIBE, {"name": "income"}),
    (EXPLAIN, {"request": REQUEST}),
    (EXPLAIN, {"request": {"features": ["incme"]}}),
    (RUN, {"plan_id": "garbage"}),
]
CALL_IDS = ["search", "search_sources", "describe_error", "explain", "explain_error", "run_error"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _clock() -> float:
    return CLOCK_START


def _base() -> PluginCollector:
    return PluginCollector.enabled_feature_groups({*FILE_SOURCE_GROUPS, RunnerFailedRowsMcp5})


def _settings(tmp_path: Path, mode: str = "plan") -> Settings:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    rows = [f"{CUSTOMER_IDS[i - 1]},{10 * i},{'a' if i % 2 else 'b'}" for i in range(1, 6)]
    (data / "sales.csv").write_text("customer_id,income,segment\n" + "\n".join(rows) + "\n")
    config = tmp_path / "mloda.toml"
    config.write_text(CONFIG, encoding="utf-8")
    return load_settings(config, ENV, {"mode": mode})


def _services(settings: Settings) -> tuple[Catalog, Planner, Runner]:
    services: tuple[Catalog, Planner, Runner] = build_services(
        settings, env=ENV, plugin_collector=_base(), clock=_clock
    )
    return services


def _server(settings: Settings) -> Any:
    return build_server(*_services(settings))


async def _tools(server: Any) -> dict[str, Tool]:
    async with Client(server) as client:
        listed = await client.list_tools()
    return {tool.name: tool for tool in listed.tools}


async def _call(server: Any, name: str, arguments: Mapping[str, Any]) -> CallToolResult:
    async with Client(server) as client:
        return await client.call_tool(name, dict(arguments))


def _texts(result: CallToolResult) -> list[str]:
    return [item.text for item in result.content if isinstance(item, TextContent)]


def _structured(result: CallToolResult) -> dict[str, Any]:
    structured = result.structured_content
    assert isinstance(structured, dict), result
    return structured


def _without_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    stripped: dict[str, Any] = json.loads(json.dumps(result))
    stripped.pop("audit_id", None)
    if isinstance(stripped.get("error"), dict):
        stripped["error"].pop("audit_id", None)
    return stripped


def _assert_envelope(result: CallToolResult) -> dict[str, Any]:
    """Two text items, a one-line summary and the JSON of the structured content; is_error mirrors it."""
    structured = _structured(result)
    assert len(result.content) == 2, result.content
    assert all(isinstance(item, TextContent) for item in result.content)
    summary, body = _texts(result)
    assert summary.strip()
    assert "\n" not in summary
    assert "\r" not in summary
    assert json.loads(body) == structured
    assert result.is_error is ("error" in structured)
    if "error" in structured:
        assert structured["error"]["code"] in summary
    return structured


def _error_code(result: CallToolResult) -> str:
    structured = _assert_envelope(result)
    assert result.is_error is True
    code: str = structured["error"]["code"]
    return code


def _refs(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _refs(item)


def _resolve_pointer(document: Any, pointer: str) -> Any:
    node = document
    for part in pointer.lstrip("/").split("/") if pointer.strip("/") else []:
        key = part.replace("~1", "/").replace("~0", "~")
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def _audit_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == LOGGER]


class TestTools:
    async def test_plan_mode_lists_exactly_the_read_tools(self, tmp_path: Path) -> None:
        tools = await _tools(_server(_settings(tmp_path)))

        assert set(tools) == PLAN_TOOLS

    async def test_execute_mode_adds_the_run_tool(self, tmp_path: Path) -> None:
        tools = await _tools(_server(_settings(tmp_path, "execute")))

        assert set(tools) == PLAN_TOOLS | {RUN}

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (SEARCH, {"read_only_hint": True, "open_world_hint": False}),
            (DESCRIBE, {"read_only_hint": True, "open_world_hint": False}),
            (EXPLAIN, {"read_only_hint": True, "open_world_hint": True}),
            (
                RUN,
                {"read_only_hint": False, "destructive_hint": False, "idempotent_hint": False, "open_world_hint": True},
            ),
        ],
    )
    async def test_annotations(self, tmp_path: Path, name: str, expected: dict[str, bool]) -> None:
        annotations = (await _tools(_server(_settings(tmp_path, "execute"))))[name].annotations

        assert annotations is not None
        assert {key: getattr(annotations, key) for key in expected} == expected

    async def test_every_tool_has_an_object_output_schema_and_a_description(self, tmp_path: Path) -> None:
        tools = await _tools(_server(_settings(tmp_path, "execute")))

        for tool in tools.values():
            assert isinstance(tool.output_schema, dict), tool.name
            assert tool.output_schema.get("type") == "object", tool.name
            assert tool.description is not None and tool.description.strip(), tool.name

    @pytest.mark.parametrize(
        ("name", "properties", "required"),
        [
            (SEARCH, {"query", "kinds", "limit", "cursor"}, set()),
            (DESCRIBE, {"name", "options"}, {"name"}),
            (EXPLAIN, {"request"}, {"request"}),
            (RUN, {"plan_id"}, {"plan_id"}),
        ],
    )
    async def test_tool_arguments(self, tmp_path: Path, name: str, properties: set[str], required: set[str]) -> None:
        schema = (await _tools(_server(_settings(tmp_path, "execute"))))[name].input_schema

        assert set(schema["properties"]) == properties
        assert set(schema.get("required", [])) == required


class TestExplainInputSchema:
    async def _schema(self, tmp_path: Path) -> dict[str, Any]:
        return (await _tools(_server(_settings(tmp_path))))[EXPLAIN].input_schema

    async def test_request_argument_carries_the_request_schema(self, tmp_path: Path) -> None:
        request = (await self._schema(tmp_path))["properties"]["request"]

        assert request["type"] == "object"
        assert request["additionalProperties"] is False
        assert request["required"] == ["features"]
        assert request["properties"]["features"]["maxItems"] == MAX_FEATURES
        assert set(request["properties"]) == {"features", "filters", "restrict_sources", "data"}

    async def test_refs_stay_inside_the_document(self, tmp_path: Path) -> None:
        schema = await self._schema(tmp_path)

        refs = list(_refs(schema))
        assert refs, "the feature-config and filter definitions are referenced"
        for ref in refs:
            assert "mloda://" not in ref
            assert ref.startswith("#/"), ref
            assert isinstance(_resolve_pointer(schema, ref[1:]), dict), ref

    @pytest.mark.parametrize(
        ("request_value", "valid"),
        [
            (REQUEST, True),
            ({"features": [{"name": "income", "options": {"x": 1}}]}, True),
            ({"features": ["income"], "filters": [{"feature": "segment", "op": "eq", "value": "a"}]}, True),
            ({"features": ["income"], "filters": [{"feature": "segment", "op": "eq"}]}, False),
            ({"features": [{"name": "income", "unknown": 1}]}, False),
            ({"features": [], "x": 1}, False),
            ([1, 2], False),
        ],
    )
    async def test_input_schema_validates_on_its_own(self, tmp_path: Path, request_value: Any, valid: bool) -> None:
        schema = await self._schema(tmp_path)
        jsonschema.Draft202012Validator.check_schema(schema)

        errors = list(jsonschema.Draft202012Validator(schema).iter_errors({"request": request_value}))

        assert (not errors) is valid, errors


class TestSchemaResource:
    def test_uri_is_the_request_schema_id(self) -> None:
        assert SCHEMA_URI == SCHEMA_ID == "mloda://schema/feature-request"

    async def test_resource_is_listed_as_json_schema(self, tmp_path: Path) -> None:
        async with Client(_server(_settings(tmp_path))) as client:
            listed = await client.list_resources()

        (resource,) = [resource for resource in listed.resources if str(resource.uri) == SCHEMA_URI]
        assert resource.mime_type == "application/schema+json"

    async def test_resource_text_is_the_request_schema(self, tmp_path: Path) -> None:
        async with Client(_server(_settings(tmp_path))) as client:
            read = await client.read_resource(SCHEMA_URI)

        (contents,) = read.contents
        assert isinstance(contents, TextResourceContents)
        assert contents.mime_type == "application/schema+json"
        assert json.loads(contents.text) == request_schema()


class TestInstructions:
    async def _instructions(self, tmp_path: Path) -> str:
        async with Client(_server(_settings(tmp_path))) as client:
            instructions = client.instructions
        assert isinstance(instructions, str)
        return instructions

    async def test_workflow_tools_in_order(self, tmp_path: Path) -> None:
        instructions = await self._instructions(tmp_path)

        positions = [instructions.find(name) for name in (SEARCH, DESCRIBE, EXPLAIN)]
        assert all(position >= 0 for position in positions), instructions
        assert positions == sorted(positions)

    async def test_descriptions_and_data_are_untrusted(self, tmp_path: Path) -> None:
        instructions = (await self._instructions(tmp_path)).lower()

        assert "untrusted" in instructions
        assert "description" in instructions
        assert "data" in instructions


class TestEnvelope:
    @pytest.mark.parametrize(("name", "arguments"), CALLS, ids=CALL_IDS)
    async def test_summary_line_and_json_body(self, tmp_path: Path, name: str, arguments: dict[str, Any]) -> None:
        result = await _call(_server(_settings(tmp_path, "execute")), name, arguments)

        _assert_envelope(result)

    @pytest.mark.parametrize(("name", "arguments"), CALLS, ids=CALL_IDS)
    async def test_no_paths_in_any_text(self, tmp_path: Path, name: str, arguments: dict[str, Any]) -> None:
        result = await _call(_server(_settings(tmp_path, "execute")), name, arguments)

        for text in _texts(result):
            assert str(tmp_path) not in text
            assert "sales.csv" not in text


class TestParity:
    async def test_explain_equals_the_planner(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _, planner, _ = _services(settings)

        result = await _call(_server(settings), EXPLAIN, {"request": REQUEST})

        assert _assert_envelope(result) == planner.explain(REQUEST)
        assert result.is_error is False

    async def test_explain_error_equals_the_planner_apart_from_audit_id(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        _, planner, _ = _services(settings)
        request = {"features": ["customer_id", "incme"]}

        result = await _call(_server(settings), EXPLAIN, {"request": request})

        assert _without_audit(_structured(result)) == _without_audit(planner.explain(request))
        assert _error_code(result) == "validation_failed"

    @pytest.mark.parametrize(
        "arguments",
        [
            {"query": ""},
            {"query": "", "kinds": ["source"]},
            {"query": "read", "kinds": ["feature_group"]},
            {"query": "", "limit": 1},
        ],
    )
    async def test_search_equals_the_catalog(self, tmp_path: Path, arguments: dict[str, Any]) -> None:
        settings = _settings(tmp_path)
        catalog, _, _ = _services(settings)

        result = await _call(_server(settings), SEARCH, arguments)

        assert _assert_envelope(result) == catalog.search(**arguments)

    async def test_search_cursor_pages_like_the_catalog(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        catalog, _, _ = _services(settings)
        server = _server(settings)

        async with Client(server) as client:
            first = _structured(await client.call_tool(SEARCH, {"query": "", "limit": 1}))
            assert first["next_cursor"] is not None
            second = await client.call_tool(SEARCH, {"query": "", "limit": 1, "cursor": first["next_cursor"]})

        assert _assert_envelope(second) == catalog.search("", limit=1, cursor=first["next_cursor"])

    @pytest.mark.parametrize(
        "arguments",
        [
            {"name": "income"},
            {"name": "CsvFG"},
            {"name": "income", "options": {RESERVED_READER_OPTION_KEY: "x"}},
            {"name": "income", "options": {"CsvFG": "x"}},
        ],
    )
    async def test_describe_equals_the_catalog_apart_from_audit_id(
        self, tmp_path: Path, arguments: dict[str, Any]
    ) -> None:
        settings = _settings(tmp_path)
        catalog, _, _ = _services(settings)

        result = await _call(_server(settings), DESCRIBE, arguments)

        assert _without_audit(_assert_envelope(result)) == _without_audit(catalog.describe(**arguments))


class TestInvalidExplainInput:
    @pytest.mark.parametrize(
        "request_value",
        [[1, 2], {"features": [], "x": 1}, "income", {"features": "income"}, {"filters": []}],
        ids=["array", "unknown_key", "string", "features_not_array", "no_features"],
    )
    async def test_reaches_the_service_as_validation_failed(self, tmp_path: Path, request_value: Any) -> None:
        settings = _settings(tmp_path)
        _, planner, _ = _services(settings)

        result = await _call(_server(settings), EXPLAIN, {"request": request_value})

        assert _error_code(result) == "validation_failed"
        details = _structured(result)["error"]["details"]
        assert details
        assert all(detail["path"] == "" or detail["path"].startswith("/") for detail in details)
        assert _without_audit(_structured(result)) == _without_audit(planner.explain(request_value))


class TestExecute:
    async def test_explain_then_run_completes_with_a_classified_preview(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, "execute")
        _, planner, runner = _services(settings)
        expected = runner.run(planner.explain(REQUEST)["plan_id"])

        async with Client(_server(settings)) as client:
            explained = await client.call_tool(EXPLAIN, {"request": REQUEST})
            plan_id = _assert_envelope(explained)["plan_id"]
            ran = await client.call_tool(RUN, {"plan_id": plan_id})

        result = _assert_envelope(ran)
        assert ran.is_error is False
        assert result["status"] == "completed"
        (table,) = result["tables"]
        assert table["preview"]["rows"] == PREVIEW_ROWS
        assert _without_audit(result) == _without_audit(expected)
        for text in _texts(explained) + _texts(ran):
            for customer_id in CUSTOMER_IDS:
                assert customer_id not in text

    async def test_run_result_carries_failed_rows_and_is_not_an_error(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, "execute")
        request = {"features": ["customer_id", FAILED]}

        async with Client(_server(settings)) as client:
            plan_id = _assert_envelope(await client.call_tool(EXPLAIN, {"request": request}))["plan_id"]
            ran = await client.call_tool(RUN, {"plan_id": plan_id})

        result = _assert_envelope(ran)
        assert ran.is_error is False
        assert result["status"] == "completed"
        assert result["failed_rows"] == [
            {"column": FAILED, "reason": "timeout", "row_count": 2, "rows": [1, 3], "truncated": False}
        ]

    async def test_run_with_a_foreign_plan_id_is_plan_invalid(self, tmp_path: Path) -> None:
        result = await _call(_server(_settings(tmp_path, "execute")), RUN, {"plan_id": "mpl1.x.y"})

        assert _error_code(result) == "plan_invalid"


def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError(LEAK)


class TestBoundary:
    @pytest.mark.parametrize(
        ("name", "service", "method", "arguments"),
        [
            (SEARCH, 0, "search", {"query": ""}),
            (DESCRIBE, 0, "describe", {"name": "income"}),
            (EXPLAIN, 1, "explain", {"request": REQUEST}),
            (RUN, 2, "run", {"plan_id": "garbage"}),
        ],
    )
    async def test_unexpected_exception_is_an_internal_error_without_its_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        name: str,
        service: int,
        method: str,
        arguments: dict[str, Any],
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        services = _services(_settings(tmp_path, "execute"))
        monkeypatch.setattr(services[service], method, _boom)

        result = await _call(build_server(*services), name, arguments)

        assert _error_code(result) == "internal"
        structured = _structured(result)
        audit_id = structured["error"]["audit_id"]
        assert audit_id.startswith("aud_")
        for text in [*_texts(result), json.dumps(structured)]:
            assert "secret" not in text
            assert LEAK_PATH not in text
        (record,) = _audit_records(caplog)
        message = record.getMessage()
        assert audit_id in message
        assert record.exc_info is None
        assert "Traceback" in message
        assert "RuntimeError" in message
        assert "secret" not in message
        assert LEAK_PATH not in message


class TestAuditLog:
    @pytest.mark.parametrize(("name", "arguments"), CALLS, ids=CALL_IDS)
    async def test_one_record_per_call_with_tool_and_error_audit_id(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, arguments: dict[str, Any]
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        server = _server(_settings(tmp_path, "execute"))

        result = await _call(server, name, arguments)

        (record,) = _audit_records(caplog)
        message = record.getMessage()
        assert name in message
        structured = _structured(result)
        if "error" in structured:
            assert structured["error"]["audit_id"] in message

    async def test_every_call_of_a_session_logs_one_record(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=LOGGER)

        async with Client(_server(_settings(tmp_path, "execute"))) as client:
            for name, arguments in CALLS:
                await client.call_tool(name, arguments)

        records = _audit_records(caplog)
        assert len(records) == len(CALLS)
        for record, (name, _) in zip(records, CALLS):
            assert name in record.getMessage()

    async def test_default_server_refuses_an_unlisted_option(self, tmp_path: Path) -> None:
        request = {"features": [{"name": "income", "options": {"opt": SECRET_OPTION}}]}

        result = await _call(_server(_settings(tmp_path)), EXPLAIN, {"request": request})

        assert result.is_error is True
        structured = _structured(result)
        assert structured["error"]["code"] == "policy_denied"
        assert [detail["path"] for detail in structured["error"]["details"]] == ["/features/0/options/opt"]
        assert structured["error"]["details"][0]["problem"] == "option_not_allowed"
        assert SECRET_OPTION not in json.dumps(structured)

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            (EXPLAIN, {"request": {"features": [{"name": "income", "options": {"opt": SECRET_OPTION}}]}}),
            (EXPLAIN, {"request": {"features": [{"name": "incme", "options": {"opt": SECRET_OPTION}}]}}),
            (DESCRIBE, {"name": "income", "options": {"opt": SECRET_OPTION}}),
            (SEARCH, {"query": SECRET_OPTION}),
        ],
        ids=["explain", "explain_error", "describe", "search"],
    )
    async def test_argument_values_never_reach_the_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, arguments: dict[str, Any]
    ) -> None:
        caplog.set_level(logging.DEBUG, logger=LOGGER)

        await _call(_server(_settings(tmp_path, "execute")), name, arguments)

        assert len(_audit_records(caplog)) == 1
        for record in caplog.records:
            assert SECRET_OPTION not in record.getMessage()
