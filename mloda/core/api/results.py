"""Public helpers for reading mlodaAPI.run_all output (issue #569)."""

from typing import Any


def results_by_feature(results: list[Any]) -> dict[str, Any]:
    """Map each column name of each result element to the element itself.

    A multi-output column "feature~suffix" is also reachable via its base name "feature".
    First occurrence wins on duplicate names. Looking up a missing name raises KeyError.
    Caveat: a literal column name containing "~" also registers its base name, which can
    shadow a same-named column from a later result.
    """
    mapping: dict[str, Any] = {}
    for element in results:
        if isinstance(element, dict):
            column_names = list(element.keys())
        elif isinstance(element, list):
            if element and not isinstance(element[0], dict):
                raise ValueError(f"Unsupported result element type: list of {type(element[0]).__name__}")
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
    """Horizontally combine run_all results into one pandas or polars DataFrame."""
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
            "run_all produced no DataFrame results; use results_by_feature or run_one "
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
