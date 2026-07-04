"""Fluent result list returned by mlodaAPI.run_all (issues #564, #568, #569)."""

from typing import Any


def _element_columns(element: Any) -> list[str]:
    """List the column names of one result element."""
    if isinstance(element, dict):
        return list(element.keys())
    if isinstance(element, list):
        if element and not isinstance(element[0], dict):
            raise ValueError(f"Unsupported result element type: list of {type(element[0]).__name__}")
        return list(element[0].keys()) if element else []
    if hasattr(element, "column_names"):
        return list(element.column_names)
    if hasattr(element, "columns"):
        return list(element.columns)
    raise ValueError(f"Unsupported result element type: {type(element).__name__}")


def _column_mapping(results: list[Any]) -> dict[str, Any]:
    """Map each column name (and each multi-output base name before the first "~") to its element.

    First occurrence wins on duplicate names.
    """
    mapping: dict[str, Any] = {}
    for element in results:
        for column_name in _element_columns(element):
            if column_name not in mapping:
                mapping[column_name] = element
            if "~" in column_name:
                base_name = column_name.split("~", 1)[0]
                if base_name not in mapping:
                    mapping[base_name] = element
    return mapping


def _to_rows(result: Any) -> list[dict[str, Any]]:
    """Convert one result element to a flat list of row dicts."""
    if isinstance(result, list):
        return list(result)
    if isinstance(result, dict):
        column_lengths = {len(values) for values in result.values()}
        if len(column_lengths) > 1:
            raise ValueError(f"Columnar dict has mismatched column lengths: {sorted(column_lengths)}.")
        keys = list(result.keys())
        return [dict(zip(keys, values)) for values in zip(*result.values())]
    if hasattr(result, "to_pylist"):
        pylist_rows: list[dict[str, Any]] = result.to_pylist()
        return pylist_rows
    if hasattr(result, "to_dicts"):
        dicts_rows: list[dict[str, Any]] = result.to_dicts()
        return dicts_rows
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        records_rows: list[dict[str, Any]] = result.to_dict("records")
        return records_rows
    if hasattr(result, "collect"):
        collected = result.collect()
        if hasattr(collected, "to_dicts"):
            collected_rows: list[dict[str, Any]] = collected.to_dicts()
            return collected_rows
        if isinstance(collected, list):
            return [row.asDict() if hasattr(row, "asDict") else row for row in collected]
    if hasattr(result, "df"):
        return _to_rows(result.df())
    raise ValueError(f"Unsupported result element type: {type(result).__name__}")


def _concat_frames(results: list[Any]) -> Any:
    """Horizontally combine result elements into one pandas or polars DataFrame."""
    elements = [
        element.collect()
        if type(element).__module__.split(".")[0] == "polars" and hasattr(element, "collect")
        else element
        for element in results
    ]

    frames: list[Any] = []
    non_frames: list[Any] = []
    for element in elements:
        if type(element).__module__.split(".")[0] in ("pandas", "polars") and hasattr(element, "columns"):
            frames.append(element)
        else:
            non_frames.append(element)

    if not frames:
        raise ValueError(
            "run_all produced no DataFrame results; use get_one, get_rows or get_values "
            "for non-DataFrame compute frameworks."
        )
    if non_frames:
        non_frame_types = ", ".join(sorted({type(element).__name__ for element in non_frames}))
        raise ValueError(f"run_all produced non-DataFrame results that cannot be concatenated: {non_frame_types}.")
    if len(frames) == 1:
        return frames[0]

    packages = {type(frame).__module__.split(".")[0] for frame in frames}
    if len(packages) > 1:
        raise ValueError("Cannot concatenate mixed pandas and polars frames in one run.")

    row_counts = {len(frame) for frame in frames}
    if len(row_counts) > 1:
        raise ValueError(f"Cannot concatenate frames with mismatched row counts: {sorted(row_counts)}.")

    seen: set[str] = set()
    deduped: list[Any] = []
    for frame in frames:
        keep = [column for column in frame.columns if column not in seen]
        seen.update(frame.columns)
        if not keep:
            continue
        deduped.append(frame if len(keep) == len(frame.columns) else frame[keep])

    if packages == {"pandas"}:
        import pandas as pd

        return pd.concat(deduped, axis=1)
    import polars as pl

    return pl.concat(deduped, how="horizontal")


class Results(list[Any]):
    """Result list of mlodaAPI.run_all: one element per resolving feature group (issues #564, #568, #569).

    Accessors read results without knowing the element shape:

    - get_one(name): the raw element containing a column (identity preserved)
    - get_rows(name): one element converted to a flat list of row dicts
    - get_values(name): one column as a plain Python list
    - get_df(): all elements horizontally concatenated into one pandas or polars DataFrame

    A multi-output column "feature~suffix" is also reachable via its base name "feature"
    (split on the first "~"). First occurrence wins on duplicate names.
    """

    def get_one(self, name: str | None = None) -> Any:
        """Return the element containing column ``name``; without a name, the sole element."""
        if name is None:
            if len(self) != 1:
                raise ValueError(
                    f"Expected exactly one result element, got {len(self)}; pass a feature name to get_one."
                )
            return self[0]
        mapping = _column_mapping(self)
        if name not in mapping:
            raise ValueError(f"Unknown feature name '{name}'. Available names: {sorted(mapping)}.")
        return mapping[name]

    def get_rows(self, name: str | None = None) -> list[dict[str, Any]]:
        """Return the element containing column ``name`` as a flat list of row dicts."""
        return _to_rows(self.get_one(name))

    def get_values(self, name: str) -> list[Any]:
        """Return the column ``name`` as a plain Python list."""
        element = self.get_one(name)
        if name not in _element_columns(element):
            raise ValueError(
                f"'{name}' is not a column of the resolved element. "
                f"Available columns: {sorted(_element_columns(element))}."
            )
        if isinstance(element, dict):
            return list(element[name])
        if isinstance(element, list):
            return [row[name] for row in element]
        column = element[name]
        if hasattr(column, "to_pylist"):
            return list(column.to_pylist())
        if hasattr(column, "to_list"):
            return list(column.to_list())
        if hasattr(column, "tolist"):
            return list(column.tolist())
        return list(column)

    def get_df(self) -> Any:
        """Return all elements horizontally concatenated into one pandas or polars DataFrame."""
        return _concat_frames(self)
