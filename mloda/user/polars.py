# Polars compute frameworks (eager and lazy). Importing this module works without polars installed;
# the frameworks then report themselves unavailable via is_available().
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame as PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import (
    PolarsLazyDataFrame as PolarsLazyDataFrame,
)

__all__ = ["PolarsDataFrame", "PolarsLazyDataFrame"]
