# Divergence Records and Rule Variants

A divergence record says where a feature group or one of its Implementations differs from the group's reference contract. How a provider declares records and `VARIANTS` is in [Operation Groups](compute-framework-integration.md#operation-groups-one-group-one-implementation-per-framework), and how records rank implementations is in [Choosing among implementations](compute-framework-integration.md#choosing-among-implementations).

## What a record holds

| Field | Meaning |
| --- | --- |
| `source` | `feature_group` or `implementation`, stamped by mloda. |
| `framework` | The compute framework an Implementation record applies on, stamped by mloda; `None` for a group record. |
| `kind` | `value`, `type`, `variant` (follows a named rule variant) or `gap` (a known shortfall that is shown but never ranks). |
| `cause` | `not_implemented`, `backend`, `environment` or `other`. |
| `reason` | One printable line; `describe()` returns `not implemented` for a reasonless `not_implemented` record, while the surfaces carry an empty reason. |
| `subtypes` / `outputs` | The subtypes or output parts it covers; both empty means the whole plugin. |
| `variant` | The rule variant name, on `variant` records only. |
| `docs` | Further explanation, required for cause `other`. |

Text fields are credential-scrubbed when stamped. A group may also declare `CONTRACT_VERSION`, an int of at least 1 naming the version of the reference contract it follows.

## Where it shows

| Surface | Fields |
| --- | --- |
| `PlanStep`, and so `explain()` | `divergences`, `contract_version`, `allowed_variants` ([explain and resolved_plan](mloda-api.md#explain-and-resolved_plan)) |
| Plan lock | the same keys on compute records ([What the file holds](plan-lock.md#what-the-file-holds)) |
| `HookContext` | `divergences`, `contract_version`, `allowed_variants` ([Reading call facts via HookContext](../chapter1/extender.md#5-reading-call-facts-via-hookcontext)) |
| `FeatureGroupInfo` | `contract_version`, `divergences`, `variants` ([Inspecting Feature Groups](discover-plugins.md#inspecting-feature-groups)) |
| Service, MCP and CLI explain step record | `contract_version`, `divergences`, `allowed_variants` ([MCP Server and CLI](mcp-server.md#tools)) |
| OpenLineage | the registry's `mlodaStep` facet |

`PlanStep`, the plan lock, and the service and CLI step records also carry `potential_nan_null_loss` ([Frameworks that hold NaN as null](#frameworks-that-hold-nan-as-null)).

A step lists the records of its group and its serving Implementation that apply, `gap` records included; `gap` records never rank an Implementation.

## Rule variants

A group lists the rule variants its contract allows in `VARIANTS`. An Implementation that follows one serves a feature only when the feature allows it with the group option `allow_variants`: a non-empty single-line string or a collection of them, read from the group options, never the context. Without it, when every fitting framework follows a variant the feature does not allow, resolution fails with a `FeatureResolutionError` whose eliminated candidates block reads:

```text
  - MyOperationFG (rule variant): MyOperationPandas on PandasDataFrame follows variant 'skip_nan', which this feature does not allow; to allow it add {"allow_variants": ("skip_nan",)} to the feature's options
```

When the group does not list the variant, the line ends `which MyOperationFG.VARIANTS does not list, so it cannot be allowed` and offers no option. The stage is the `rule variant` row of [the eliminated candidates block](troubleshooting/feature-group-resolution-errors.md#the-eliminated-candidates-block); `Elimination.variants` and `Elimination.allow_variants` carry the names ([Feature Group Matching](feature-group-matching.md)). What gets blocked, and when a pin raises, is in [Operation Groups](compute-framework-integration.md#operation-groups-one-group-one-implementation-per-framework). The service reports a `variant_not_allowed` detail with a `fix` of `{"allow_variants": [...]}` only when the group lists every blocking variant and the caller may set the option ([Tools](mcp-server.md#tools)).

## Frameworks that hold NaN as null

`ComputeFramework.nan_is_null()` declares that a framework cannot tell NaN from null (Pandas and SQLite return `True`, every other shipped framework `False`). The plan marks `potential_nan_null_loss` on a step where data that can hold both moves onto such a framework: a transform, a requested result conversion, or a neutral source load. A join step is never marked; the transform into its destination carries it. The mark is advisory: nothing is rerouted and no `allow_variants` is needed.

It replaces the operation records whose only reason is that the framework cannot tell NaN from null (NaN and null merged as partition keys, tied in `order_by`, NaN skipped because it is null), registered as `nan_as_null` variants. Records for operation-level choices, such as Pandas `sum` skipping NaN, stay. Until the registry retires the old records, the fact is stated twice.

mloda's transfers into Pandas produce float64, which folds NaN and null. A Pandas frame built by the user with nullable `Float64` can still hold both, so a Pandas to SQLite transfer is not marked.

## Upgrading

A plugin release that marks a backend behavior as a rule variant can turn a run that worked into this error; add `allow_variants` or use a framework without the variant.
