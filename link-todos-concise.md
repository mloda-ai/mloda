# Link System - Concise TODO List

## 🔴 Critical Priority

### T1: Multi-Index Support
- [x] Implement multi-index merge logic in `BaseMergeEngine` and all 6 framework implementations
- [x] Add comprehensive test coverage for single and multi-column index joins
- [x] Update `docs/in_depth/join_data.md` with multi-index examples and usage guidance

### T2: Pointer Field Documentation
- [ ] Research actual usage of `left_pointer` and `right_pointer` fields in codebase
- [ ] Either document with clear examples or deprecate if unused
- [ ] Update `link.py` docstrings and add to FAQ if keeping

### T3: README Links Section
- [ ] Add "When Do You Need Links?" section to README after Compute Frameworks
- [ ] Include working code example with basic Link usage
- [ ] Link to full documentation at `docs/in_depth/join_data.md`

## 🟡 Important Priority

### T4: Link Builder API
- [ ] Create `link_builder.py` with fluent interface for less verbose link creation
- [ ] Implement methods like `.between()`, `.on()`, `.inner()`, `.build()` with full test coverage
- [ ] Document new API in `join_data.md` while maintaining backward compatibility

### T5: Visual Documentation
- [ ] Add decision tree diagram "Do I Need Links?" using Mermaid to `join_data.md`
- [ ] Create execution flow sequence diagram showing Link resolution process
- [ ] Add before/after table examples for all 6 join types (INNER, LEFT, RIGHT, OUTER, APPEND, UNION)

### T6: Error Message Examples
- [ ] Add "Troubleshooting Link Errors" section to `docs/in_depth/join_data.md`
- [ ] Document 4+ common errors with exact error messages and how-to-fix guidance
- [ ] Include before/after code snippets for each error scenario

### T7: Performance Profiling
- [ ] Create benchmark suite `tests/benchmarks/test_link_performance.py` with pytest-benchmark
- [ ] Profile link validation and resolution with cProfile, identify bottlenecks (O(n²) loops)
- [ ] Document performance characteristics and add "Performance Tips" section to docs

## 🟢 Nice-to-Have

### T8: Link Inference Helper
- [ ] Create `link_helper.py` with `LinkInferenceHelper` class for auto-suggesting links
- [ ] Implement column name similarity matching and confidence scoring
- [ ] Add schema introspection to suggest links based on matching column names

### T9: Link Registry System
- [ ] Create `link_registry.py` with central `LinkRegistry` for managing reusable links
- [ ] Implement registration, retrieval by name/tags, and auto-discovery for features
- [ ] Document best practices for organizing links in large projects

### T10: Advanced Join Types
- [ ] Extend `JoinType` enum with ASOF, SEMI, ANTI, CROSS join types
- [ ] Implement new join methods in `BaseMergeEngine` and all 6 framework merge engines
- [ ] Add documentation and examples for each advanced join type use case
