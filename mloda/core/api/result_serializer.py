"""Framework-aware serialization helpers for ``mlodaAPI.run_all`` results (issue #573).

Each ``result`` is a SINGLE compute-framework object (one element of the
``run_all`` list), e.g. a ``pd.DataFrame``, ``pa.Table``, polars DataFrame, or a
python_dict result (``list[dict]``).

Notes:
- Record-list (python_dict) results serialize with the stdlib only. Framework
  objects (pandas, polars, pyarrow Table, ...) go through the pyarrow hub, so
  pyarrow is required for non-python_dict framework results.
- ``to_json_records`` values are Python natives produced by the compute
  framework and may include non-JSON-primitive types (e.g. ``datetime``) for
  such columns, mirroring pandas' ``to_dict("records")``.
"""

import csv
import io
from typing import Any

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]


def _require_pyarrow() -> None:
    if pa is None:
        raise ImportError("pyarrow is required for this operation. Install it with: pip install 'mloda[pyarrow]'")


def _is_record_list(result: Any) -> bool:
    """A python_dict result is an empty list or a list where every element is a dict."""
    return isinstance(result, list) and (len(result) == 0 or all(isinstance(item, dict) for item in result))


def _ambiguous_list_error() -> ValueError:
    return ValueError(
        "Expected a single run_all result object, got a list whose elements are not all dicts. "
        "Pass a single result element, e.g. results[0], not the whole run_all() list."
    )


def _record_union_fieldnames(records: list[dict[str, Any]]) -> list[str]:
    """Union of all keys across all records, preserving first-seen order."""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _rows_to_csv(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _framework_to_pyarrow_table(result: Any) -> "pa.Table":
    _require_pyarrow()

    if isinstance(result, pa.Table):
        return result

    from_framework = type(result)
    chain = ComputeFrameworkTransformer().get_transformation_chain(from_framework, pa.Table)

    if chain is None:
        raise ValueError(f"No transformation path found from {from_framework} to pyarrow.Table.")
    if len(chain) != 1:
        raise ValueError(
            f"Unexpected multi-step transformation chain from {from_framework} to pyarrow.Table; "
            "a direct transformer was expected."
        )

    return chain[0].transform(from_framework, pa.Table, result, None)


def to_json_records(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        if _is_record_list(result):
            return list(result)
        raise _ambiguous_list_error()

    records: list[dict[str, Any]] = _framework_to_pyarrow_table(result).to_pylist()
    return records


def to_csv(result: Any) -> str:
    if isinstance(result, list):
        if _is_record_list(result):
            return _rows_to_csv(_record_union_fieldnames(result), result)
        raise _ambiguous_list_error()

    table = _framework_to_pyarrow_table(result)
    return _rows_to_csv(table.column_names, table.to_pylist())
