import re
from typing import Any

from mloda.provider import BaseTransformer

_FIXED_OFFSET = re.compile(r"^[+-]\d{2}:?(\d{2})$")

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


class PolarsPyArrowTransformer(BaseTransformer):
    """
    Transformer for converting between Polars DataFrame and PyArrow Table.

    This transformer handles bidirectional conversion between Polars DataFrame
    and PyArrow Table data structures, leveraging Polars' native PyArrow integration
    for efficient zero-copy operations where possible.
    """

    @classmethod
    def framework(cls) -> Any:
        if pl is None:
            return NotImplementedError
        return pl.DataFrame

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        import polars as pl  # noqa: F401

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa  # noqa: F401

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform a Polars DataFrame to a PyArrow Table.

        This method uses Polars' native to_arrow() method for efficient conversion.
        """
        return data.to_arrow()

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
        """
        Transform a PyArrow Table to a Polars DataFrame.

        This method uses Polars' native from_arrow() method for efficient conversion.
        """
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        cls._reject_sub_hour_offsets(data)
        return pl.from_arrow(data)

    @classmethod
    def _reject_sub_hour_offsets(cls, data: Any) -> None:
        """Raise ValueError for timestamp columns with fixed offsets that have non-zero minutes."""
        for field in data.schema:
            if not pa.types.is_timestamp(field.type):
                continue
            tz = field.type.tz
            match = _FIXED_OFFSET.match(tz) if tz else None
            if match and match.group(1) != "00":
                raise ValueError(
                    f"Column '{field.name}' has time zone '{tz}', but Polars has no sub-hour fixed offset. "
                    "Convert it to UTC first."
                )
