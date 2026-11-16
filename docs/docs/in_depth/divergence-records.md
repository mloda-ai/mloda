# Divergence Records and Rule Variants

A divergence record says where a feature group or one of its Implementations differs from the group's reference contract. How a provider declares records and `VARIANTS` is in [Operation Groups](compute-framework-integration.md#operation-groups-one-group-one-implementation-per-framework), and how records rank implementations is in [Choosing among implementations](compute-framework-integration.md#choosing-among-implementations).

## What a record holds

| Field | Meaning |
| --- | --- |
| `source` | `feature_group` or `implementation`, stamped by mloda. |
| `framework` | The compute framework an Implementation record applies on, stamped by mloda; `None` for a group record. |
| `kind` | `value`, `type`, `variant` (follows a named rule variant) or `gap` (a known shortfall that is shown but never ranks). |
| `cause` | `not_implemented`, `backend`, `environment` or `other`. |
| `reason` | One printable line; a reasonless `not_implemented` record reads `not implemented`. |
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

A step lists the records of its group and its serving Implementation that apply, `gap` records included; `gap` records never rank an Implementation.

## Rule variants

A group lists the rule variants its contract allows in `VARIANTS`. An Implementation that follows one serves a feature only when the feature allows it with the group option `allow_variants`: a non-empty single-line string or a collection of them, read from the group options, never the context. Without it, when every fitting framework follows a variant the feature does not allow, resolution fails with a `FeatureResolutionError` whose eliminated candidates block reads:

```text
  - MyOperationFG (rule variant): MyOperationPandas on PandasDataFrame follows variant 'skip_nan', which this feature does not allow; to allow it add {"allow_variants": ("skip_nan",)} to the feature's options
```

When the group does not list the variant, the line ends `which MyOperationFG.VARIANTS does not list, so it cannot be allowed` and offers no option. The stage is the `rule variant` row of [the eliminated candidates block](troubleshooting/feature-group-resolution-errors.md#the-eliminated-candidates-block); `Elimination.variants` and `Elimination.allow_variants` carry the names ([Feature Group Matching](feature-group-matching.md)). A pin on a chained step that a variant blocks raises an `ImplementationPinError`. The service reports a `variant_not_allowed` detail with a `fix` of `{"allow_variants": [...]}` when the caller may set the option ([Tools](mcp-server.md#tools)).

A plugin release that marks a backend behavior as a rule variant can turn a run that worked into this error; add `allow_variants` or use a framework without the variant.
