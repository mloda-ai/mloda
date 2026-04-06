from typing import Any, Optional

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda.core.filter.single_filter import SingleFilter


class FilterMask:
    """Build a boolean mask from SingleFilters for use inside calculate_feature().

    The engine must be wired at setup time via ComputeFramework.filter_mask_engine().
    It is set on the FeatureSet before calculate_feature() is called.

    Usage inside a FeatureGroup:
        mask = FilterMask.build(data, features, column="status")
    """

    @classmethod
    def build(
        cls,
        data: Any,
        features: Any,
        column: str,
    ) -> Any:
        """Build a boolean mask by AND-combining all filters targeting the given column.

        Args:
            data: The tabular data container (pa.Table, pd.DataFrame, etc.).
            features: A FeatureSet with .filters and .filter_mask_engine attributes.
            column: The column name to filter on.

        Returns a framework-native boolean array/series. If no filters match,
        returns an all-True mask.

        Raises:
            TypeError: If no filter_mask_engine is set on features.
        """
        engine = cls._get_engine(features)
        matching = cls._matching_filters(features.filters, column)

        mask = engine.all_true(data)
        for sf in matching:
            single = cls._single_mask(engine, data, sf, column)
            mask = engine.combine(mask, single)
        return mask

    @classmethod
    def _get_engine(cls, features: Any) -> type[BaseFilterMaskEngine]:
        """Get the mask engine from features. Raises if not wired at setup time."""
        engine: type[BaseFilterMaskEngine] | None = getattr(features, "filter_mask_engine", None)
        if engine is not None:
            return engine
        raise TypeError(
            "No filter_mask_engine set on features. "
            "The ComputeFramework must override filter_mask_engine() "
            "to return a BaseFilterMaskEngine subclass."
        )

    @classmethod
    def _matching_filters(
        cls,
        filters: Optional[set[SingleFilter]],
        column: str,
    ) -> list[SingleFilter]:
        if filters is None:
            return []
        return [sf for sf in filters if sf.name == column]

    @classmethod
    def _single_mask(
        cls,
        engine: type[BaseFilterMaskEngine],
        data: Any,
        sf: SingleFilter,
        column: str,
    ) -> Any:
        ft = sf.filter_type

        if ft == "equal":
            return engine.equal(data, column, sf.parameter.value)

        if ft == "min":
            return engine.greater_equal(data, column, sf.parameter.value)

        if ft == "max":
            max_val = sf.parameter.max_value
            if max_val is not None:
                if sf.parameter.max_exclusive:
                    return engine.less_than(data, column, max_val)
                return engine.less_equal(data, column, max_val)
            return engine.less_equal(data, column, sf.parameter.value)

        if ft == "range":
            min_mask = engine.greater_equal(data, column, sf.parameter.min_value)
            if sf.parameter.max_exclusive:
                max_mask = engine.less_than(data, column, sf.parameter.max_value)
            else:
                max_mask = engine.less_equal(data, column, sf.parameter.max_value)
            return engine.combine(min_mask, max_mask)

        if ft == "categorical_inclusion":
            return engine.is_in(data, column, sf.parameter.values)

        raise ValueError(
            f"Filter type '{ft}' is not supported for mask building. "
            f"Supported types: equal, min, max, range, categorical_inclusion."
        )
