import argparse
from typing import Any
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.api.request import mlodaAPI

import logging

logger = logging.getLogger(__name__)


def format_array(prefix: str, array: Any, indent: int = 2, color: str = "34") -> str:
    """Formats a NumPy array for better console output."""
    indent_str = " " * indent
    formatted_values = ", ".join(map(str, array.tolist()))
    return f"{indent_str}\033[{color}m{prefix} [\033[0m{formatted_values}\033[{color}m]\033[{color}m]\033[0m"


def print_results(feature_group: str, results: Any) -> None:
    for i, res in enumerate(results):
        if feature_group in res:  # Ensure the expected feature group is in the result
            formatted_output = format_array(f"Result {i} values: ", res[feature_group].values)
            print(formatted_output)
        else:
            print(f"Error: Feature group '{feature_group}' not found in result {i}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mlodaAPI.run_all() with a specified feature group.")
    parser.add_argument("feature_group", help="The feature group to process.")
    args = parser.parse_args()

    PluginLoader().load_group("feature_group")

    feature_group = args.feature_group

    results = mlodaAPI.run_all(features=[feature_group])
    print_results(feature_group, results)
