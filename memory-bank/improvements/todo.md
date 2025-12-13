# Split Runner Class Refactoring

**Goal:** Split the monolithic Runner class (638 lines, 10+ instance variables) into focused, single-responsibility modules to improve testability, maintainability, and reduce race condition risks.

**Source File:** `mloda_core/runtime/run.py`

---

## Phase Checklist

- [x] **Phase 1: WorkerManager** - Extract thread/process lifecycle management
- [x] **Phase 2: DataLifecycleManager** - Extract data dropping and result collection
- [ ] **Phase 3: ComputeFrameworkExecutor** - Extract step execution and CFW initialization
- [ ] **Phase 4: ExecutionOrchestrator** - Refactor Runner as thin orchestration layer
- [ ] **Phase 5: Integration & Cleanup** - Final validation and documentation

---

## Target Architecture

Split into 4 focused components:

| Component | Responsibility |
|-----------|---------------|
| **ExecutionOrchestrator** | Main compute loop, step scheduling, dependency checking |
| **ComputeFrameworkExecutor** | Step execution (sync/thread/multiprocessing), CFW initialization |
| **WorkerManager** | Thread/process lifecycle, task joining, result queue handling |
| **DataLifecycleManager** | Data dropping, result collection, artifact tracking |

---

## Phase 1: Create WorkerManager Class ✅
> Extract thread/process management into dedicated class

- [x] Create `mloda_core/runtime/worker_manager.py`
- [x] Extract `tasks` list management
- [x] Extract `process_register` dictionary
- [x] Extract `result_queues_collection` and `result_uuids_collection`
- [x] Move `join()` method
- [x] Move `get_done_steps_of_multiprocessing_result_queue()` method
- [x] Move `_wait_for_drop_completion()` method
- [x] Add unit tests for WorkerManager in isolation (31 tests)
- [x] Integrate WorkerManager into Runner (inject as dependency)
- [x] Run tox to validate (1463 passed)

---

## Phase 2: Create DataLifecycleManager Class ✅
> Extract data dropping and result collection logic

- [x] Create `mloda_core/runtime/data_lifecycle_manager.py`
- [x] Extract `result_data_collection` dictionary
- [x] Extract `track_data_to_drop` dictionary
- [x] Extract `artifacts` dictionary
- [x] Move `_drop_data_for_finished_cfws()` method
- [x] Move `_drop_cfw_data()` method
- [x] Move `_drop_data_if_possible()` method
- [x] Move `add_to_result_data_collection()` method
- [x] Move `get_result_data()` method
- [x] Move `get_result()` method
- [x] Move `get_artifacts()` method
- [x] Add unit tests for DataLifecycleManager in isolation (32 tests)
- [x] Integrate DataLifecycleManager into Runner
- [x] Run tox to validate (1495 passed)

---

## Phase 3: Create ComputeFrameworkExecutor Class
> Extract step execution and CFW initialization logic

- [ ] Create `mloda_core/runtime/compute_framework_executor.py`
- [ ] Extract `cfw_collection` dictionary
- [ ] Extract `transformer` instance
- [ ] Move `prepare_execute_step()` method
- [ ] Move `prepare_tfs_right_cfw()` method
- [ ] Move `prepare_tfs_and_joinstep()` method
- [ ] Move `sync_execute_step()` method
- [ ] Move `thread_execute_step()` method
- [ ] Move `multi_execute_step()` method
- [ ] Move `_get_execution_function()` method
- [ ] Move `add_compute_framework()` method
- [ ] Move `init_compute_framework()` method
- [ ] Move `get_cfw()` method
- [ ] Add unit tests for ComputeFrameworkExecutor in isolation
- [ ] Integrate ComputeFrameworkExecutor into Runner
- [ ] Run tox to validate

---

## Phase 4: Refactor Runner as ExecutionOrchestrator
> Runner becomes a thin orchestration layer

- [ ] Rename `Runner` to `ExecutionOrchestrator`
- [ ] Keep only orchestration logic:
  - `__init__()` - inject dependencies
  - `__enter__()` / `__exit__()` - context management
  - `compute()` - main execution loop
  - `_is_step_done()` - dependency checking
  - `_can_run_step()` - step scheduling
  - `_mark_step_as_finished()` - step completion
  - `currently_running_step()` - running step check
  - `_execute_step()` - delegate to executor
  - `_process_step_result()` - delegate to data lifecycle
- [ ] Update all import statements across codebase
- [ ] Add unit tests for ExecutionOrchestrator in isolation
- [ ] Run tox to validate

---

## Phase 5: Integration & Cleanup
> Final integration and documentation

- [ ] Ensure all components work together correctly
- [ ] Run full integration test suite
- [ ] Update any docstrings referencing old structure
- [ ] Add module-level docstrings to new files
- [ ] Run tox for final validation
- [ ] Update `mloda_core_improvements.md` to mark item as complete

---

## State Variables Distribution

| Variable | Target Class |
|----------|--------------|
| `execution_planner` | ExecutionOrchestrator |
| `cfw_register` | ExecutionOrchestrator (shared) |
| `result_data_collection` | DataLifecycleManager |
| `track_data_to_drop` | DataLifecycleManager |
| `artifacts` | DataLifecycleManager |
| `location` | ExecutionOrchestrator (shared) |
| `tasks` | WorkerManager |
| `process_register` | WorkerManager |
| `result_queues_collection` | WorkerManager |
| `result_uuids_collection` | WorkerManager |
| `transformer` | ComputeFrameworkExecutor |
| `flight_server` | ExecutionOrchestrator |
| `cfw_collection` | ComputeFrameworkExecutor |
| `manager` | ExecutionOrchestrator |

---

## Method Distribution

### WorkerManager
- `join()`
- `get_done_steps_of_multiprocessing_result_queue()`
- `_wait_for_drop_completion()`

### DataLifecycleManager
- `_drop_data_for_finished_cfws()`
- `_drop_cfw_data()`
- `_drop_data_if_possible()`
- `add_to_result_data_collection()`
- `get_result_data()`
- `get_result()`
- `get_artifacts()`

### ComputeFrameworkExecutor
- `prepare_execute_step()`
- `prepare_tfs_right_cfw()`
- `prepare_tfs_and_joinstep()`
- `sync_execute_step()`
- `thread_execute_step()`
- `multi_execute_step()`
- `_get_execution_function()`
- `add_compute_framework()`
- `init_compute_framework()`
- `get_cfw()`

### ExecutionOrchestrator (Runner)
- `__init__()`
- `__enter__()` / `__exit__()`
- `compute()`
- `_is_step_done()`
- `_can_run_step()`
- `_mark_step_as_finished()`
- `currently_running_step()`
- `_execute_step()` (delegates)
- `_process_step_result()` (delegates)

---

## Risk Mitigation

1. **Race Conditions**: Each class owns its state clearly; use locks where shared access is needed
2. **Test Coverage**: Write unit tests for each component before integration
3. **Incremental Changes**: Complete each phase and validate with tox before proceeding

---

## Dependencies Between Components

```
ExecutionOrchestrator
    ├── WorkerManager (manages threads/processes)
    ├── DataLifecycleManager (manages data lifecycle)
    └── ComputeFrameworkExecutor (executes steps)
            └── WorkerManager (for thread/process execution)
```

The ExecutionOrchestrator coordinates all components, while ComputeFrameworkExecutor needs access to WorkerManager for thread/multiprocess step execution.
