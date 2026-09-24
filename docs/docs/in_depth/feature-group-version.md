# Feature Group Versioning

`FeatureGroup.version()` returns an identifier that changes when the code a feature group can run changes. Extenders receive it as `HookContext.feature_group_version`, so lineage, audit and tracing records can tell two implementations apart.

## The version string

The version has three parts, joined by `-`:

- the installed mloda version,
- the module that defines the feature group,
- a SHA-256 hash of the feature group's implementation.

## What the implementation hash covers

- **Roots**: the feature group class and every base class in its MRO that is first-party. First-party means the feature group's own top-level package or the `mloda.*` plugin namespace, except mloda's own `mloda.core`, `mloda.user`, `mloda.provider` and `mloda.steward`. The version prefix covers mloda itself.
- **Reachable code**: every function, class and module-level constant the roots reference by name or by `module.attr`, followed through first-party code. This includes helpers in other modules and constants imported with `from ... import`. A module used as a value, for example in `getattr(helpers, name)`, counts as a whole.
- **Canonical form**: each definition is hashed from its syntax tree. Code, constants, decorators, base classes and type annotations count. Docstrings, comments, blank lines and formatting do not. Functions the feature group never references do not count either. The hash is the same on every supported Python version.

```python
from typing import Any

from mloda.provider import FeatureGroup, FeatureSet

LIMIT = 10


def clip(value: int) -> int:
    return min(value, LIMIT)


class ClippedValue(FeatureGroup):
    """Editing this docstring leaves the version unchanged."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return clip(data)
```

Changing `LIMIT` or the body of `clip` changes `ClippedValue.version()`, even if `clip` lives in another module of the same package.

## Third-party dependencies

The code of third-party packages is never hashed. By default, `ThirdPartyVersionMode.INCLUDE` adds the name and version of every third-party package the reachable code references, so upgrading pandas changes the version of every feature group that uses pandas. A package's version comes from its distribution metadata or its `__version__`. A package with neither is recorded by name only. To leave dependency versions out, override `version_third_party_mode()`. A shared base class sets it for all its subclasses:

```python
from mloda.provider import FeatureGroup, ThirdPartyVersionMode


class DependencyAgnostic(FeatureGroup):
    @classmethod
    def version_third_party_mode(cls) -> ThirdPartyVersionMode:
        return ThirdPartyVersionMode.EXCLUDE
```

## What the hash does not cover

- mloda itself, beyond the version prefix. Edits to an editable mloda install do not change it.
- Code reached only through runtime values:
    - registries filled elsewhere (`REGISTRY["k"] = f`, `REGISTRY.update(...)`),
    - classes discovered by reflection, such as the file readers a reader feature group finds through `__subclasses__()`,
    - imports by a computed name (`importlib.import_module(name)`),
    - attributes assigned outside the class body,
    - `getattr` on objects that are not modules,
    - names captured from an enclosing function.
- Imports inside a function body, for example to avoid a circular import. The import statement counts; the imported code does not.
- Modules without Python source (C extensions, bytecode-only installs). Their definitions are recorded by name only.
- Names served by a module-level `__getattr__`, and constants in modules without a source file (for example notebook cells). These are not recorded at all.
- Data and configuration files the code reads.
- Source edited after import in a long-lived process. The hash reads the source files the first time it runs for a class, then caches the result for that class object.

## When it is computed

`version()` runs for every hook call while an extender is active, and in `get_feature_group_docs()`. The hash is computed once per class and cached for the lifetime of the class object. Each module is parsed once per process, and only the definitions the walk reaches are hashed.

## Custom versioning

Override `FeatureGroup.version()`:

```python
from mloda.provider import FeatureGroup


class PinnedVersion(FeatureGroup):
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
```

`BaseFeatureGroupVersion` provides the building blocks `mloda_version()`, `module_name()` and `implementation_hash()`. Subclassing it alone changes nothing, because `FeatureGroup.version()` calls `BaseFeatureGroupVersion.version(cls)` directly.

## Reading the mloda Package Version

Three supported programmatic paths, all backed by `mloda.core.version.get_mloda_version()`:

```python
from mloda.user import __version__          # also on mloda.provider and mloda.steward
from mloda.core.version import get_mloda_version
from mloda.provider import BaseFeatureGroupVersion

get_mloda_version()                          # memoized, "0.0.0" if not installed
BaseFeatureGroupVersion.mloda_version()      # same value
```

There is no `mloda.__version__`: `mloda` is a PEP 420 namespace root with no `__init__.py`, so plugin packages can add subpackages under it.
