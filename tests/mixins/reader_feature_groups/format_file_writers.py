"""Write a small file of each stock format holding exactly the given columns (backends imported lazily)."""

import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, columns: dict[str, list[Any]]) -> None:
    names = list(columns)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in zip(*(columns[name] for name in names)):
            writer.writerow(row)


def write_json(path: Path, columns: dict[str, list[Any]]) -> None:
    names = list(columns)
    with open(path, "w") as handle:
        for row in zip(*(columns[name] for name in names)):
            handle.write(json.dumps(dict(zip(names, row))) + "\n")


def write_parquet(path: Path, columns: dict[str, list[Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pyarrow_parquet

    pyarrow_parquet.write_table(pa.Table.from_pydict(columns), str(path))


def write_feather(path: Path, columns: dict[str, list[Any]]) -> None:
    import pyarrow as pa

    table = pa.Table.from_pydict(columns)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def write_orc(path: Path, columns: dict[str, list[Any]]) -> None:
    import pyarrow as pa
    import pyarrow.orc as pyarrow_orc

    with pa.OSFile(str(path), "wb") as sink:
        pyarrow_orc.write_table(pa.Table.from_pydict(columns), sink)
