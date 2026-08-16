## Join planner pipeline

How a declared `Link` becomes an executed merge. Four stages run during `prepare()`, before any
data moves. [Join data](join_data.md) covers the user-facing `Link` API; this page covers what the
planner does with it, and which parts of the decision are observable.

### 1. Discovery

`mloda/core/prepare/resolve_links.py` matches each declared `Link` against the feature-group nodes
that resolution produced, and records the matches in a `LinkTrekker`. A link that matches no pair of
nodes plans nothing. One link can match more than once (polymorphic matching, multiple same-class
nodes separated by discriminators).

### 2. Agreement

`mloda/core/prepare/resolve_compute_frameworks.py` decides which compute framework each side runs
in. Agreement is placement plus transform, not set intersection: a valid join can have parents whose
declared framework sets are disjoint, because one side can be transformed into the other's framework.

This stage may **invert** the link, making the declared right side the merge destination. Inversion
is why the declared sides and the executed direction are two different facts, and why both are
reported separately (see [Reading the plan](#reading-the-plan)).

### 3. Materialization and validation

`ExecutionPlan.run_link` builds one `ResolvedJoin` record per planned orientation. The record is the
join decision: the declared sides and their indices, the destination side, the destination and source
frameworks, the parents on each side, the consumers waiting on it, and a completion token.

Two checks then run over the whole set:

- `raise_on_join_plan_divergence` cross-checks every record against the `JoinStep` built from it. A
  disagreement is an internal planning bug, not a user error.
- `raise_on_orphaned_join_source` rejects two joins that drain a shared parent which no join writes
  back into, for a consumer both joins feed. Those branches can never reunite, so a consumer needing
  both would silently lose one.

### 4. Lowering

The `JoinStep` is constructed from the record's own fields, and its completion token is the record's
token. When the two sides sit in different frameworks, `add_tfs` inserts a `TransformFrameworkStep`
hop that moves the source side into the destination side, taking the direction from the record rather
than re-deriving it from the join type.

### Reading the plan

`PlanStep` reports the join decision, so the planned shape does not need hand-instrumented dumps.

```py
from mloda.user import mloda

for step in mloda.explain(["my_feature"], compute_frameworks=["PandasDataFrame"]):
    if step.step_kind == "join":
        print(
            step.join_type,
            step.feature_group_name,         # declared left side
            step.source_feature_group_name,  # declared right side
            step.join_destination_side,      # "left" or "right": the side the merge runs in
            step.join_inverted,              # True when the destination is the declared right side
            step.compute_framework_name,     # merge destination framework
            step.source_compute_framework_name,
        )
```

The declared sides are fixed by the `Link`. `join_destination_side` and `join_inverted` are the
planner's answer and can differ from run to run of different requests, because agreement depends on
which frameworks the surrounding features resolved to.

`join_token` is the join's completion token, the same uuid the scheduler tracks and the value a
stalled-plan error reports. It is the join key between an explain record and a scheduling message.

All three are `None` on compute and transform steps.
