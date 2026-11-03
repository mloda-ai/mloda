"""The bundled ``mloda demo``: the README first example on in-memory data, with its resolved plan."""

from collections.abc import Mapping
from typing import Any, TextIO

from mloda.core.abstract_plugins.components.input_data.api_data_fg import ApiDataFG
from mloda.core.api.text_summary import plan_detail, plan_lines, result_and_plan, table_lines
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


def _label(step: Mapping[str, Any]) -> str:
    return str(step["feature_group"] if step["kind"] == "compute" else step["kind"])


def write_demo(explained: Mapping[str, Any], ran: Mapping[str, Any], out: TextIO) -> None:
    """Print the run result tables, the resolved plan and the onward lines."""
    tables = []
    for table in ran["tables"]:
        names = [column["name"] for column in table["columns"]]
        tables.append("\n".join(table_lines(names, [[row.get(name) for name in names] for row in table["rows"]])))
    steps = plan_lines(
        (step["step"], _label(step), plan_detail(step["features"], step["source"], step["inputs"]))
        for step in explained["steps"]
    )
    out.write(result_and_plan(tables, steps) + "\n\nNext\n")
    out.write(f"Edit the README example: {README_URL}\n")
    out.write('DataFrames: pip install "mloda[pandas]"\n')
