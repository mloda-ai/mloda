"""Write a small file or database of each stock format holding exactly the given columns (backends imported lazily)."""

import csv
import json
import sqlite3
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


def write_sqlite(path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
    """Create a SQLite database holding one table per entry, each with exactly the given columns."""
    declared = {int: "INTEGER", float: "REAL", str: "TEXT", bytes: "BLOB"}
    connection = sqlite3.connect(path)
    try:
        for table, columns in tables.items():
            names = list(columns)
            quoted_table = '"' + table.replace('"', '""') + '"'
            quoted = ['"' + name.replace('"', '""') + '"' for name in names]
            types = [declared[type(columns[name][0])] for name in names]
            connection.execute(f"CREATE TABLE {quoted_table} ({', '.join(f'{q} {t}' for q, t in zip(quoted, types))})")
            placeholders = ", ".join("?" for _ in names)
            insert = f"INSERT INTO {quoted_table} VALUES ({placeholders})"  # nosec B608 - quoted names, bound values
            connection.executemany(insert, list(zip(*(columns[name] for name in names))))
        connection.commit()
    finally:
        connection.close()
