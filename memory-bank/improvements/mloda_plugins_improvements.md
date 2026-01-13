# mloda_plugins Code Improvements

This document identifies 10 breaking-change improvements for the mloda_plugins codebase. Each improvement includes a rationale, pros/cons analysis, and testing checklist.

## 6. Standardize Data Access Patterns

### Status
- [ ] Design DataAccessProtocol interface
- [ ] Create FileDataAccess implementation
- [ ] Create DatabaseDataAccess implementation
- [ ] Create CredentialDataAccess implementation
- [ ] Refactor ReadFile to use protocol
- [ ] Refactor ReadDB to use protocol
- [ ] Refactor SQLite to use protocol
- [ ] Standardize validation return types
- [ ] Update all data access tests
- [ ] Run tox validation

### Rationale
Three different patterns exist for data access: ReadFile checks `isinstance(data_access, (DataAccessCollection, str, Path))`, ReadDB checks `isinstance(data_access, (DataAccessCollection, HashableDict))`, and SQLite accesses `credentials.data.get()` directly. This inconsistency makes it hard to understand what data_access types are supported, leads to different validation approaches, and produces inconsistent error messages. A unified DataAccessProtocol would standardize behavior.

**Pros:**
- Consistent validation across all data sources
- Clear contract for data access objects
- Easier to add new data source types
- Predictable error handling

**Cons:**
- Breaking change to data access signatures
- Existing code using raw strings/paths needs updating
- More verbose for simple use cases
