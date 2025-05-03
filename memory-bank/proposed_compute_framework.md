# Compute Framework Improvements

## Recent Improvements

### 2. Integration Tests
While unit tests are now in place for both filter engines, additional integration tests would be beneficial:

- End-to-end tests with both compute frameworks
- Tests for complex scenarios involving multiple filters
- Performance comparison tests between Pandas and PyArrow implementations

## Recommendations

1. **Implement merge_union for PyArrowMergeEngine** by combining merge_append with deduplication
2. **Add integration tests** that verify filtering works end-to-end with both compute frameworks
3. **Add test for PyArrow union merge** once implemented

These remaining improvements would complete the feature parity between the PandasDataframe and PyarrowTable implementations.
