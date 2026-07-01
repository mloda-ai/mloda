"""Framework-aware serialization helpers for ``mlodaAPI.run_all`` results (issue #573).

Each ``result`` is a SINGLE compute-framework object (one element of the
``run_all`` list), e.g. a ``pd.DataFrame``, ``pa.Table``, polars DataFrame, or a
python_dict result (``list[dict]``).
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


def _to_pyarrow_table(result: Any) -> "pa.Table":
    _require_pyarrow()

    if isinstance(result, pa.Table):
        return result

    if _is_record_list(result):
        return pa.Table.from_pylist(result)

    transformer = ComputeFrameworkTransformer()
    from_framework = type(result)
    chain = transformer.get_transformation_chain(from_framework, pa.Table)

    if chain is None:
        raise ValueError(f"No transformation path found from {from_framework} to pyarrow.Table.")

    data = result
    current_fw = from_framework
    for i, transformer_cls in enumerate(chain):
        if i == len(chain) - 1:
            target_fw = pa.Table
        else:
            for (src, dst), trans in transformer.transformer_map.items():
                if trans == transformer_cls and src == current_fw:
                    target_fw = dst
                    break
        data = transformer_cls.transform(current_fw, target_fw, data, None)
        current_fw = target_fw

    return data


def to_json_records(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        if _is_record_list(result):
            return list(result)
        raise ValueError(
            f"Expected a single run_all result object, got a list of {type(result[0])}. "
            "Pass a single result element, e.g. results[0], not the whole run_all() list."
        )

    table = _to_pyarrow_table(result)
    records: list[dict[str, Any]] = table.to_pylist()
    return records


def to_csv(result: Any) -> str:
    if isinstance(result, list) and not _is_record_list(result):
        raise ValueError(
            f"Expected a single run_all result object, got a list of {type(result[0])}. "
            "Pass a single result element, e.g. results[0], not the whole run_all() list."
        )

    table = _to_pyarrow_table(result)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=table.column_names, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table.to_pylist())
    return buf.getvalue()
