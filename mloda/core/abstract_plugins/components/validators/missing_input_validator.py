from collections.abc import Collection
from typing import Any, Literal

InputCheck = Literal["checked", "skipped_source", "skipped_empty"]

JOIN_SUFFIXES = ("_x", "_y", "_right")


def join_renamed_columns(missing: str, columns: Collection[str]) -> list[str]:
    """Columns a join between two inputs that both carry `missing` would have renamed it to."""
    return [f"{missing}{suffix}" for suffix in JOIN_SUFFIXES if f"{missing}{suffix}" in columns]


class MissingInputFeatureError(ValueError):
    """A feature group's declared input is absent from the data handed to calculate_feature."""

    code = "missing_input_feature"

    def __init__(
        self,
        feature_group: str,
        feature: str,
        missing_inputs: tuple[str, ...],
        available_columns: tuple[str, ...],
    ) -> None:
        self.feature_group = feature_group
        self.feature = feature
        self.missing_inputs = tuple(missing_inputs)
        self.available_columns = tuple(available_columns)
        message = (
            f"Feature group {feature_group} cannot compute feature '{feature}': declared input(s) "
            f"{list(self.missing_inputs)} missing from the incoming data; available columns: "
            f"{list(self.available_columns)}."
        )
        for name in self.missing_inputs:
            renamed = join_renamed_columns(name, self.available_columns)
            if renamed:
                message += (
                    f" Column(s) {renamed} suggest a join between two inputs that both carry '{name}' renamed it."
                )
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (self.feature_group, self.feature, self.missing_inputs, self.available_columns),
        )


def _rebuild(
    feature_group: str, feature: str, missing_inputs: tuple[str, ...], available_columns: tuple[str, ...]
) -> MissingInputFeatureError:
    return MissingInputFeatureError(feature_group, feature, missing_inputs, available_columns)
