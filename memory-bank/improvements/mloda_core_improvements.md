# mloda_core Code Improvements

This document identifies 10 high-impact code improvements for mloda_core. These changes may break backwards compatibility but will significantly improve code quality, maintainability, and testability.

---

## Improvement Checklist

- [will not do] 1. Decompose ComputeFrameWork class into focused components
- [ ] 2. Split Runner class into single-responsibility modules
- [ ] 3. Replace get_all_subclasses() reflection with dependency injection
- [x] 4. Make FeatureName immutable
- [ ] 5. Standardize Abstract/Base prefix naming convention
- [ ] 6. Replace mlodaAPI 8-parameter constructor with builder/config pattern
- [x] 7. Use NamedTuple/dataclass for Link tuple parameters
- [ ] 8. Replace Options dual-category dict with explicit factory methods
- [x] 9. Eliminate type: ignore suppressions with proper typing
- [x] 10. Extract validation logic into separate validator classes

---

## 1. Decompose ComputeFrameWork Class

**File:** `mloda_core/abstract_plugins/compute_frame_work.py` (506 lines)

The ComputeFrameWork class violates the Single Responsibility Principle by handling 7+ distinct concerns: data transformation, validation, merge operations, upload/download, calculation orchestration, function extension support, and framework-specific conversions. This monolithic design makes the class difficult to understand, test in isolation, and extend. Decomposing it into focused components (DataTransformer, FeatureValidator, DataMerger, DataLoader, CalculationOrchestrator) would allow each concern to evolve independently and enable proper unit testing without complex setup.

**Pros:**
- Enables isolated unit testing for each component
- Easier to understand and maintain smaller classes
- Allows independent evolution of concerns
- Reduces cognitive load when implementing new compute frameworks

**Cons:**
- Significant refactoring effort across all compute framework implementations
- May introduce coordination overhead between components
- Breaking change for all existing custom compute frameworks

---

## 2. Split Runner Class

**File:** `mloda_core/runtime/run.py` (615 lines)

The Runner class manages execution planning, multiprocessing, thread management, data lifecycle, and artifact tracking in a single 600+ line file with 10+ instance variables tracking complex state. This tight coupling makes it nearly impossible to unit test individual behaviors and creates race condition risks in parallel execution paths. Splitting into ExecutionOrchestrator, ComputeFrameworkExecutor, WorkerManager, and DataLifecycleManager would enable proper testing, clearer ownership of responsibilities, and safer concurrent execution.

**Pros:**
- Enables proper unit testing of each component
- Reduces risk of race conditions through clearer state ownership
- Easier to reason about execution flow
- Simplifies debugging of parallel execution issues

**Cons:**
- Complex refactoring with high risk of introducing bugs
- Need to carefully manage inter-component communication
- May introduce performance overhead from additional abstraction layers

---

## 3. Replace get_all_subclasses() with Dependency Injection

**Files:** `mloda_core/prepare/accessible_plugins.py`, multiple locations

The current plugin discovery uses Python's `__subclasses__()` reflection, causing global state pollution where importing any feature group or compute framework registers it system-wide. This creates test pollution (tests affect each other), makes it impossible to run tests with a controlled subset of plugins, and couples the entire system to import-time side effects. Replacing with explicit plugin registration via dependency injection would enable test isolation, clearer plugin lifecycle management, and removal of hidden global state.

**Pros:**
- Complete test isolation - no more test pollution
- Explicit plugin lifecycle management
- Easier to run with subset of plugins for testing
- Removes hidden coupling to import order

**Cons:**
- Requires changes to all plugin registration patterns
- More verbose plugin setup in user code
- Breaking change for all existing plugin implementations

---

## 5. Standardize Abstract/Base Prefix Naming

**Files:** Multiple abstract class files

The codebase inconsistently uses both `Abstract*` (e.g., AbstractFeatureGroup) and `Base*` (e.g., BaseInputData, BaseValidator, BaseMergeEngine) prefixes for abstract classes without clear distinction. This inconsistency creates confusion about the intended use pattern and makes it harder to discover related classes. Standardizing on a single prefix (preferably `Base*` as it's more common in Python) would improve codebase navigability, reduce cognitive load, and establish a clear naming convention for contributors.

**Pros:**
- Consistent naming improves discoverability
- Reduces confusion about class roles
- Clearer convention for new contributors
- Better IDE autocomplete experience

**Cons:**
- Breaking change affecting all subclass imports
- Requires updating all documentation
- Need to update all test files referencing renamed classes

---

## 6. Replace mlodaAPI Constructor with Builder/Config Pattern

**File:** `mloda_core/api/request.py` (lines 21-32)

The mlodaAPI constructor has 8 parameters including confusing union types like `Union[Set[Type[ComputeFrameWork]], Optional[list[str]]]` for compute_frameworks. This makes the API difficult to use correctly, hard to extend without breaking existing code, and prone to parameter ordering mistakes. A builder pattern or configuration object would provide better discoverability, enable validation at configuration time, and allow adding new options without changing the constructor signature.

**Pros:**
- Clearer, more discoverable API
- Enables validation at configuration time
- Extensible without breaking changes
- Self-documenting configuration

**Cons:**
- More verbose for simple use cases
- Existing code must migrate to new pattern
- Additional classes to maintain

---

## 8. Replace Options Dual-Category Dict with Factory Methods

**File:** `mloda_core/abstract_plugins/components/options.py`

The Options class uses an implicit dual-category system (group vs context dicts) with confusing rules about which category affects feature group matching. The `add()` method even has legacy behavior comments. Replacing with explicit factory methods like `Options.from_feature_group_parameters()` and `Options.from_metadata()` would make the categorization explicit, remove magic behavior, and help developers understand the impact of their configuration choices.

**Pros:**
- Explicit categorization removes confusion
- Self-documenting factory methods
- Removes legacy behavior complexity
- Clearer validation per category

**Cons:**
- Breaking change for all Options usage
- More verbose than dict-style construction
- Requires updating all existing Options instantiations
