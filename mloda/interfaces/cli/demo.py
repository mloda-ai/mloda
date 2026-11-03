"""The bundled ``mloda demo``: the README first example on in-memory data, with its resolved plan."""

from collections.abc import Mapping, Sequence
from typing import Any, TextIO

from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from mloda.core.api.text_summary import plan_detail, result_and_plan, step_lines, table_lines
from mloda.interfaces.service.config import Settings
from mloda.user import PluginCollector
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

REQUEST: dict[str, Any] = {
    "features": ["customer_id", "income", "income__sum_aggr", "age__avg_aggr"],
    "data": {
        "SampleData": {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "age": [25, None, 45, 30, 50],
            "income": [50000, 75000, None, 60000, 85000],
        }
    },
}
SETTINGS = Settings(mode="execute", inline_data=True, trusted_caller=True)
COLLECTOR = PluginCollector.enabled_feature_groups({ApiDataFG, AggregatedFeatureGroup})
README_URL = "https://github.com/mloda-ai/mloda#run-it"


def _step_line(step: Mapping[str, Any]) -> tuple[int, str, str]:
    label = step["feature_group"] if step["kind"] == "compute" else step["kind"]
    detail = plan_detail(
        step["kind"],
        step["feature_group"],
        step["source_feature_group"],
        step["join_type"],
        step["features"],
        step["source"],
        step["inputs"],
    )
    return step["step"], label, detail


def plan_lines(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """One aligned line per plan step."""
    return step_lines(_step_line(step) for step in steps)


def _table_text(table: Mapping[str, Any]) -> str:
    names = [column["name"] for column in table["columns"]]
    return "\n".join(table_lines(names, ([row.get(name) for name in names] for row in table["rows"])))


def write_demo(explained: Mapping[str, Any], ran: Mapping[str, Any], out: TextIO) -> None:
    """Print the run result tables, the resolved plan and the onward lines."""
    tables = [_table_text(table) for table in ran["tables"]]
    out.write(result_and_plan(tables, plan_lines(explained["steps"])) + "\n\nNext\n")
    out.write(f"Edit the README example: {README_URL}\n")
    out.write('DataFrames: pip install "mloda[pandas]"\n')
