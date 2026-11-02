"""The bundled ``mloda demo``: the README first example on in-memory data, with its resolved plan."""

from collections.abc import Mapping, Sequence
from typing import Any, TextIO

from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
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
README_URL = "https://github.com/mloda-ai/mloda#30-second-example"


def _aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _plan_detail(step: Mapping[str, Any]) -> str:
    if step["kind"] == "transform":
        return f"{step['source_feature_group']} -> {step['feature_group']}"
    if step["kind"] == "join":
        return f"{step['feature_group']} + {step['source_feature_group']}  ({step['join_type']})"
    detail = ", ".join(step["features"])
    if step["source"]:
        return f"{detail}  (source {step['source']})"
    if step["inputs"]:
        return f"{detail}  (from {', '.join(step['inputs'])})"
    return detail


def plan_lines(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """One aligned line per plan step."""
    labels = [step["feature_group"] if step["kind"] == "compute" else step["kind"] for step in steps]
    return _aligned([[f"step {step['step']}", label, _plan_detail(step)] for step, label in zip(steps, labels)])


def write_demo(explained: Mapping[str, Any], ran: Mapping[str, Any], out: TextIO) -> None:
    """Print the run result tables, the resolved plan and the onward lines."""
    out.write("Result\n")
    blocks = []
    for table in ran["tables"]:
        names = [column["name"] for column in table["columns"]]
        rows = [names] + [[str(row.get(name)) for name in names] for row in table["rows"]]
        blocks.append("\n".join(_aligned(rows)))
    out.write("\n\n".join(blocks) + "\n\nPlan\n")
    out.write("\n".join(plan_lines(explained["steps"])) + "\n\nNext\n")
    out.write(f"Edit the README example: {README_URL}\n")
    out.write('DataFrames: pip install "mloda[pandas]"\n')
