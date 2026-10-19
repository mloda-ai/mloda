"""Prints one json line with the result column names of a csv read through run_all on PyArrowTable.
No test_ prefix, so pytest never collects it; the column order test runs it as a script.
"""

import json
import tempfile
from pathlib import Path

from mloda.user import DataAccessCollection, PluginLoader, mloda

COLUMNS = ["zeta", "alpha", "mu", "kappa", "omega", "beta", "sigma", "delta"]


def collect() -> dict[str, str]:
    PluginLoader.all()
    with tempfile.TemporaryDirectory(prefix="column_order_probe_") as folder:
        path = Path(folder) / "probe.csv"
        path.write_text(",".join(COLUMNS) + "\n" + ",".join("1" for _ in COLUMNS) + "\n")
        result = mloda.run_all(
            COLUMNS,
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(files={str(path)}),
        )
    return {"columns": ",".join(result[0].column_names)}


if __name__ == "__main__":
    print(json.dumps(collect(), sort_keys=True))
