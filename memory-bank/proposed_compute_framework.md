# Compute Framework Expansion: Technical Implementation Plan

## Overview

Expand mloda's compute framework ecosystem from the current Pandas/PyArrow implementations to support five additional frameworks. Each framework will integrate with the existing `BaseTransformer` architecture using PyArrow as the common interchange format.

## Current Architecture Analysis

### Existing Components
- **ComputeFrameWork**: Abstract base class defining framework interface
- **BaseTransformer**: Abstract class for bidirectional data conversion
- **ComputeFrameworkTransformer**: Registry that auto-discovers transformer subclasses
- **MergeEngine**: Framework-specific join operations
- **FilterEngine**: Framework-specific filtering implementations

### Current Transformer Pattern
```python
class PandasPyarrowTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return pd.DataFrame
    
    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table
    
    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        # Pandas → PyArrow conversion
        
    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        # PyArrow → Pandas conversion
```

## Proposed Framework Implementations

### 1. PythonDict Framework

**Data Structure**: `List[Dict[str, Any]]` for tabular data

**ComputeFrameWork Implementation**:
```python
class PythonDictFramework(ComputeFrameWork):
    def expected_data_framework(self) -> Type:
        return list
    
    def transform(self, data: Dict) -> List[Dict]:
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            # Columnar: {"col1": [1,2], "col2": [3,4]} → [{"col1":1,"col2":3}, {"col1":2,"col2":4}]
            return [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        return data
    
    def select_data_by_column_names(self, data: List[Dict], column_names: List[str]) -> List[Dict]:
        return [{k: record[k] for k in column_names if k in record} for record in data]
```

**Required Transformer**:
```python
class PythonDictPyarrowTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return list  # List[Dict]
    
    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table
    
    @classmethod
    def import_fw(cls) -> None:
        pass  # Built-in Python types
    
    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa
    
    @classmethod
    def transform_fw_to_other_fw(cls, data: List[Dict]) -> pa.Table:
        return pa.Table.from_pylist(data)
    
    @classmethod
    def transform_other_fw_to_fw(cls, data: pa.Table) -> List[Dict]:
        return data.to_pylist()
```

**MergeEngine Implementation**:
```python
class PythonDictMergeEngine(BaseMergeEngine):
    def merge_inner(self, left: List[Dict], right: List[Dict], on: List[str]) -> List[Dict]:
        right_index = {tuple(r[k] for k in on): r for r in right}
        return [
            {**l, **right_index[tuple(l[k] for k in on)]}
            for l in left
            if tuple(l[k] for k in on) in right_index
        ]
```

### 2. Polars Framework

**Data Structure**: `polars.DataFrame`

**ComputeFrameWork Implementation**:
```python
class PolarsFramework(ComputeFrameWork):
    def expected_data_framework(self) -> Type:
        return pl.DataFrame
    
    def transform(self, data: Dict) -> pl.DataFrame:
        return pl.DataFrame(data)
    
    def select_data_by_column_names(self, data: pl.DataFrame, column_names: List[str]) -> pl.DataFrame:
        return data.select(column_names)
```

**Required Transformer**:
```python
class PolarsPyarrowTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return pl.DataFrame
    
    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table
    
    @classmethod
    def import_fw(cls) -> None:
        import polars as pl
    
    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa
    
    @classmethod
    def transform_fw_to_other_fw(cls, data: pl.DataFrame) -> pa.Table:
        return data.to_arrow()
    
    @classmethod
    def transform_other_fw_to_fw(cls, data: pa.Table) -> pl.DataFrame:
        return pl.from_arrow(data)
```

**Technical Considerations**:
- Lazy evaluation: Use `pl.LazyFrame` for complex operations
- Expression API: `pl.col("name").sum()` for transformations
- Memory efficiency: Columnar storage with zero-copy operations

### 3. DuckDB Framework

**Data Structure**: `duckdb.DuckDBPyRelation`

**ComputeFrameWork Implementation**:
```python
class DuckDBFramework(ComputeFrameWork):
    def __init__(self) -> None:
        self.conn = duckdb.connect()
    
    def expected_data_framework(self) -> Type:
        return duckdb.DuckDBPyRelation
    
    def transform(self, data: Dict) -> duckdb.DuckDBPyRelation:
        # Convert dict to PyArrow first, then to DuckDB relation
        arrow_table = pa.Table.from_pydict(data)
        return self.conn.from_arrow(arrow_table)
    
    def select_data_by_column_names(self, data: duckdb.DuckDBPyRelation, column_names: List[str]) -> duckdb.DuckDBPyRelation:
        return data.select(*column_names)
```

**Required Transformer**:
```python
class DuckDBPyarrowTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return duckdb.DuckDBPyRelation
    
    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table
    
    @classmethod
    def import_fw(cls) -> None:
        import duckdb
    
    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa
    
    @classmethod
    def transform_fw_to_other_fw(cls, data: duckdb.DuckDBPyRelation) -> pa.Table:
        return data.to_arrow_table()
    
    @classmethod
    def transform_other_fw_to_fw(cls, data: pa.Table) -> duckdb.DuckDBPyRelation:
        conn = duckdb.connect()
        return conn.from_arrow(data)
```

### 4. Iceberg Framework

**Data Structure**: `pyiceberg.table.Table`

**ComputeFrameWork Implementation**:
```python
class IcebergFramework(ComputeFrameWork):
    def __init__(self, catalog_config: Dict):
        self.catalog = load_catalog(catalog_config)
    
    def expected_data_framework(self) -> Type:
        return pyiceberg.table.Table
    
    def transform(self, data: Dict) -> pyiceberg.table.Table:
        # Convert to PyArrow first, then create Iceberg table
        arrow_table = pa.Table.from_pydict(data)
        return self._create_iceberg_table(arrow_table)
```

**Required Transformer**:
```python
class IcebergPyarrowTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return pyiceberg.table.Table
    
    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table
    
    @classmethod
    def import_fw(cls) -> None:
        import pyiceberg
    
    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa
    
    @classmethod
    def transform_fw_to_other_fw(cls, data: pyiceberg.table.Table) -> pa.Table:
        return data.scan().to_arrow()
    
    @classmethod
    def transform_other_fw_to_fw(cls, data: pa.Table) -> pyiceberg.table.Table:
        # This requires creating a new Iceberg table or appending to existing
        # Implementation depends on catalog configuration
        raise NotImplementedError("Iceberg table creation requires catalog context")
```

**Technical Complexity**:
- Catalog integration (Hive, Glue, Nessie)
- Schema evolution handling
- Partition management
- Transaction isolation

### 5. Spark Framework

**Data Structure**: `pyspark.sql.DataFrame`

**ComputeFrameWork Implementation**:
```python
class SparkFramework(ComputeFrameWork):
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
    
    def expected_data_framework(self) -> Type:
        return pyspark.sql.DataFrame
    
    def transform(self, data: Dict) -> pyspark.sql.DataFrame:
        # Convert to PyArrow first, then to Spark DataFrame
        arrow_table = pa.Table.from_pydict(data)
        return self.spark.createDataFrame(arrow_table.to_pandas())
```

**Required Transformer**:
```python
class SparkPyarrowTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return pyspark.sql.DataFrame
    
    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table
    
    @classmethod
    def import_fw(cls) -> None:
        import pyspark.sql
    
    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa
    
    @classmethod
    def transform_fw_to_other_fw(cls, data: pyspark.sql.DataFrame) -> pa.Table:
        return pa.Table.from_pandas(data.toPandas())
    
    @classmethod
    def transform_other_fw_to_fw(cls, data: pa.Table) -> pyspark.sql.DataFrame:
        # Requires active SparkSession
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        return spark.createDataFrame(data.to_pandas())
```

## Transformation Architecture

### PyArrow as Central Hub

All new frameworks connect to the existing ecosystem through PyArrow:

```
PythonDict ←→ PyArrow ←→ Pandas
  Polars   ←→ PyArrow ←→ Existing
  DuckDB   ←→ PyArrow     System
  Iceberg  ←→ PyArrow
  Spark    ←→ PyArrow
```

**Benefits**:
- Single transformation path per framework
- Leverages PyArrow's broad compatibility
- Maintains existing Pandas/PyArrow transformer
- Reduces complexity from N×N to N×1 transformations

### Auto-Discovery

The `ComputeFrameworkTransformer` automatically discovers all new transformers via `get_all_subclasses(BaseTransformer)`.

## MergeEngine Implementations

### Framework-Specific Optimizations

**PythonDict**: Hash-based joins using dictionary lookups
```python
def merge_inner(self, left: List[Dict], right: List[Dict], on: List[str]) -> List[Dict]:
    right_index = {tuple(r[k] for k in on): r for r in right}
    return [{**l, **right_index[key]} for l in left 
            if (key := tuple(l[k] for k in on)) in right_index]
```

**Polars**: Expression-based joins with lazy evaluation
```python
def merge_inner(self, left: pl.DataFrame, right: pl.DataFrame, on: List[str]) -> pl.DataFrame:
    return left.join(right, on=on, how="inner")
```

**DuckDB**: SQL JOIN operations with query optimization
```python
def merge_inner(self, left: duckdb.DuckDBPyRelation, right: duckdb.DuckDBPyRelation, on: List[str]) -> duckdb.DuckDBPyRelation:
    return left.join(right, condition=" AND ".join(f"left.{col} = right.{col}" for col in on))
```

## FilterEngine Implementations

### Unified Filter Expression System

```python
class FilterExpression:
    def to_python_dict(self) -> Callable:
        return lambda record: eval(f"record['{self.column}'] {self.operator} {self.value}")
    
    def to_polars(self) -> pl.Expr:
        return getattr(pl.col(self.column), self._polars_op_map[self.operator])(self.value)
    
    def to_sql(self) -> str:
        return f"{self.column} {self.operator} {self._escape_value(self.value)}"
```

## Testing Strategy

### Unit Test Framework

```python
@pytest.fixture(params=[
    PythonDictFramework,
    PolarsFramework, 
    DuckDBFramework,
    IcebergFramework,
    SparkFramework
])
def compute_framework(request):
    return request.param()

def test_transform_consistency(compute_framework):
    """Test that transform produces expected data structure"""
    input_data = {"col1": [1, 2], "col2": [3, 4]}
    result = compute_framework.transform(input_data)
    assert isinstance(result, compute_framework.expected_data_framework())
```

### Transformer Tests

```python
def test_pyarrow_roundtrip():
    """Test that Framework→PyArrow→Framework preserves data"""
    original_data = [{"col1": 1, "col2": 3}, {"col1": 2, "col2": 4}]
    
    # Test each transformer
    for transformer_cls in [PythonDictPyarrowTransformer, PolarsPyarrowTransformer, ...]:
        arrow_data = transformer_cls.transform_fw_to_other_fw(original_data)
        restored_data = transformer_cls.transform_other_fw_to_fw(arrow_data)
        assert restored_data == original_data
```

## Implementation Phases

### Phase 1: Foundation
- Implement PythonDict framework and PyArrow transformer
- Extend unit test framework for new patterns
- Validate transformer auto-discovery mechanism

### Phase 2: Performance Frameworks  
- ✅ **COMPLETED**: Implement Polars framework with PyArrow transformer
- **Future TODO List (For Later Sessions)**:
  1. **Lazy evaluation support** - `pl.LazyFrame` implementation
  2. ✅ **COMPLETED**: **Hide from mloda core discovery** - Automatic dependency detection implemented for all frameworks
  3. **Feature group implementations** - Polars versions of existing feature groups
- Implement DuckDB framework with PyArrow transformer
- Add integration tests

### ✅ Completed: Automatic Dependency Detection (Phase 2.2)

**Implementation Summary**:
- Added `is_available()` static method to `ComputeFrameWork` base class
- Implemented dependency checking in all existing frameworks:
  - `PandasDataframe.is_available()` - checks for pandas
  - `PyarrowTable.is_available()` - checks for pyarrow  
  - `PolarsDataframe.is_available()` - checks for polars
- Updated `get_cfw_subclasses()` in `accessible_plugins.py` to filter unavailable frameworks
- Added comprehensive tests for availability checking functionality
- Updated documentation in `docs/docs/chapter1/compute-frameworks.md`

**Benefits Achieved**:
- Frameworks automatically hidden when dependencies missing
- No runtime import errors for missing optional dependencies
- Supports minimal deployment environments
- Maintains backward compatibility

### Phase 3: Enterprise Frameworks
- Implement Iceberg framework with PyArrow transformer
- Implement Spark framework with PyArrow transformer
- Add end-to-end integration tests

## Technical Constraints

### Memory Management
- Large dataset streaming through PyArrow intermediate format
- Zero-copy operations where possible (Polars, PyArrow)
- Garbage collection optimization for temporary objects

### Error Handling
- Framework-specific exception mapping to common error types
- Graceful degradation when optional dependencies unavailable
- Detailed error context for transformation failures

### Concurrency
- Thread safety for shared connections (DuckDB, Spark)
- Connection pooling for database-backed frameworks

This implementation plan leverages PyArrow as the central interchange format, simplifying the transformer architecture while expanding mloda's compute framework ecosystem.
