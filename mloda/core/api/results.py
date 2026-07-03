"""Public helpers for reading mlodaAPI.run_all output (issue #569)."""

from typing import Any


def results_by_feature(results: list[Any]) -> dict[str, Any]:
    """Map each column name of each result element to the element itself.

    A multi-output column "feature~suffix" is also reachable via its base name "feature".
    First occurrence wins on duplicate names. Looking up a missing name raises KeyError.
    """
    mapping: dict[str, Any] = {}
    for element in results:
        if isinstance(element, dict):
            column_names = list(element.keys())
        elif isinstance(element, list):
            column_names = list(element[0].keys()) if element else []
        elif hasattr(element, "column_names"):
            column_names = list(element.column_names)
        elif hasattr(element, "columns"):
            column_names = list(element.columns)
        else:
            raise ValueError(f"Unsupported result element type: {type(element).__name__}")

        for column_name in column_names:
            if column_name not in mapping:
                mapping[column_name] = element
            if "~" in column_name:
                base_name = column_name.split("~", 1)[0]
                if base_name not in mapping:
                    mapping[base_name] = element
    return mapping


def _to_rows(result: Any) -> list[dict[str, Any]]:
    """Convert one run_all result element to a flat list of row dicts."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
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
    raise ValueError(f"Unsupported result element type: {type(result).__name__}")


def _concat_frames(results: list[Any]) -> Any:
    """Horizontally combine run_all results into one pandas or polars DataFrame."""
    frames = [
        element
        for element in results
        if type(element).__module__.split(".")[0] in ("pandas", "polars") and hasattr(element, "columns")
    ]
    if not frames:
        raise ValueError(
            "run_all produced no DataFrame results; use results_by_feature or run_one "
            "for non-DataFrame compute frameworks."
        )
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
