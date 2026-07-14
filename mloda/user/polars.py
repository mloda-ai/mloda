# Polars compute frameworks (eager and lazy). Backend: polars (install: mloda[polars]).
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame as PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import (
    PolarsLazyDataFrame as PolarsLazyDataFrame,
)

__all__ = ["PolarsDataFrame", "PolarsLazyDataFrame"]
