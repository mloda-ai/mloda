"""Framework-aware serialization helpers for ``mlodaAPI.run_all`` results (issue #573).

Each ``result`` is a SINGLE compute-framework object (one element of the
``run_all`` list), e.g. a ``pd.DataFrame``, ``pa.Table``, polars DataFrame, or a
python_dict result (``list[dict]``).

Notes:
- Conversions dispatch through mloda's compute-framework registry. Only
  python_dict record-list CSV output stays pyarrow-free; all other framework
  conversions go through the pyarrow hub, so pyarrow is required for
  non-python_dict framework results.
- Record values are Python natives produced by the target compute framework and
  may include non-JSON-primitive types (e.g. ``datetime``) for such columns.
"""

import csv
import io
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)
from mloda.core.abstract_plugins.components.utils import get_all_subclasses

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


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite records to the UNION of all keys, filling missing keys with ``None``."""
    fieldnames = _record_union_fieldnames(records)
    return [{name: rec.get(name) for name in fieldnames} for rec in records]


def _load_compute_frameworks() -> None:
    from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

    if "compute_framework" not in PluginLoader._disabled_groups:
        PluginLoader().load_group("compute_framework")


def _resolve_framework(framework: str | type[ComputeFramework]) -> type[ComputeFramework]:
    if isinstance(framework, type) and issubclass(framework, ComputeFramework):
        return framework

    for cls in get_all_subclasses(ComputeFramework):
        if cls.get_class_name() == framework:
            return cls

    _load_compute_frameworks()
    sub_classes = get_all_subclasses(ComputeFramework)
    for cls in sub_classes:
        if cls.get_class_name() == framework:
            return cls

    available_names = sorted(cls.get_class_name() for cls in sub_classes)
    raise ValueError(
        f"No compute framework named '{framework}' found in available compute frameworks: {available_names}."
    )


def to_framework(
    result: Any,
    framework: str | type[ComputeFramework],
    framework_connection_object: Any = None,
) -> Any:
    """Convert a single run_all result into the native object of the target compute framework."""
    if isinstance(result, list) and not _is_record_list(result):
        raise _ambiguous_list_error()

    target_cls = _resolve_framework(framework)

    target_type = target_cls.expected_data_framework()
    if target_type is None:
        raise ValueError(f"Compute framework '{target_cls.get_class_name()}' does not declare a native data type.")

    source_type = type(result)
    if source_type == target_type:
        return result

    data = result
    if _is_record_list(result):
        data = _normalize_records(result)

    transformer = ComputeFrameworkTransformer()
    chain = transformer.get_transformation_chain(source_type, target_type)
    if chain is None:
        raise ValueError(f"No transformation path found from {source_type} to {target_type}.")

    current_fw = source_type
    for i, transformer_cls in enumerate(chain):
        if i == len(chain) - 1:
            step_target = target_type
        else:
            step_target = None
            for (src, dst), trans in transformer.transformer_map.items():
                if trans == transformer_cls and src == current_fw:
                    step_target = dst
                    break
            if step_target is None:
                raise ValueError(
                    f"Could not determine intermediate target for transformer {transformer_cls} from {current_fw}."
                )

        data = transformer_cls.transform(current_fw, step_target, data, framework_connection_object)
        current_fw = step_target

    return data


def to_records(result: Any) -> list[dict[str, Any]]:
    """Serialize a single ``run_all`` result to a list of row dicts (python_dict native)."""
    records: list[dict[str, Any]] = to_framework(result, "PythonDictFramework")
    return records


def to_arrow(result: Any) -> "pa.Table":
    """Serialize a single ``run_all`` result to a pyarrow Table."""
    _require_pyarrow()
    return to_framework(result, "PyArrowTable")


def to_csv(result: Any) -> str:
    """Serialize a single ``run_all`` result to a CSV string."""
    if isinstance(result, list):
        if _is_record_list(result):
            return _rows_to_csv(_record_union_fieldnames(result), result)
        raise _ambiguous_list_error()

    table = to_arrow(result)
    return _rows_to_csv(table.column_names, table.to_pylist())
