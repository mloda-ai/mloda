# MCP Server and CLI

mloda ships an [MCP](https://modelcontextprotocol.io) server, `mloda-mcp`, so agents can discover features, plan
requests and, when you allow it, run them. The `mloda` command line calls the same functions, so whatever an agent
sees you can reproduce locally.

## Install

The server needs the `mcp` extra; the `mloda` command needs nothing beyond mloda itself.

```bash
pip install "mloda[mcp]"
```

Add it to Claude Code, or to any client that starts MCP servers over stdio:

```bash
claude mcp add mloda -- mloda-mcp --config /abs/path/mloda.toml
```

```json
{ "mcpServers": { "mloda": { "command": "mloda-mcp", "args": ["--config", "/abs/path/mloda.toml"] } } }
```

`mloda mcp serve` is the same server. Plugins are whatever is installed in the server's environment when it starts;
agents cannot install or load plugins.

## Configure

Settings live in the `[mcp]` table of `mloda.toml`. Without `--config`, `./mloda.toml` is used if it exists;
`--no-config` ignores it. Relative paths resolve against the file.

```toml
[mcp]
mode = "plan"                          # "plan" (default) or "execute"
compute_frameworks = ["PythonDictFramework"]
max_rows = 100000                      # refuse requests that may read or return more rows (see below)
deny_feature_groups = ["CustomerEmailFeatureGroup"]
alternatives = { customer_email = ["customer_id"] }
preview_allows = ["public", "internal"]
max_preview_rows = 20
inline_data = false                    # allow inline request data (trusted callers only)
# max_inline_values = 1000             # refuse inline data with more values (default: no limit)
# max_inline_chars = 10000             # refuse inline data with more string characters (default: no limit)
plan_ttl = "24h"
# default_classification = "internal"  # label for unlisted leaf columns (default: unclassified)
# allow_options = ["window"]           # option keys a request may set (default: none)

[mcp.sources.sales]
files = ["data/sales.csv"]

[mcp.classification]
customer_id = "personal"
income = "internal"
segment = "public"

[mcp.options]                          # default group options for every requested feature
model = "small"

[mcp.context_options]                  # default context options for every requested feature
timeout = 30
```

`MLODA_MCP_MODE`, `MLODA_MCP_MAX_ROWS`, `MLODA_MCP_MAX_INLINE_VALUES`, `MLODA_MCP_MAX_INLINE_CHARS` and
`MLODA_MCP_COMPUTE_FRAMEWORKS` override the file. `MLODA_MCP_SIGNING_KEY`
signs plan ids; without it every process uses a random key, so a plan id is valid only in the process that issued it.

`[mcp.options]` and `[mcp.context_options]` are operator defaults merged under each requested feature; a feature's own
value wins. They are not subject to `allow_options` and their values never appear in results. Values are strings,
numbers, booleans, lists or tables (no TOML dates or times), and `in_features` is not allowed.

## Tools

| Tool | What it does |
| --- | --- |
| `mloda_search_features` | Lists FeatureGroups and source aliases matching a query, one page at a time |
| `mloda_describe_feature` | Resolves one feature name and lists the options of its FeatureGroup |
| `mloda_explain_request` | Validates and plans a request without reading data; returns the plan and a `plan_id` |
| `mloda_run_request` | Runs a `plan_id`; only offered when `mode = "execute"` |

The resource `mloda://schema/feature-request` is the JSON Schema of a request. A request is the
[feature configuration](feature-config.md) JSON plus optional filters, source restrictions and inline data:

```json
{
  "features": ["customer_id", {"name": "income", "options": {}}],
  "filters": [{"feature": "segment", "op": "eq", "value": "a"}],
  "restrict_sources": ["sales"],
  "data": {"Inline": {"token": ["007", "", "12"]}}
}
```

`data` is optional inline data, `{"<set>": {"<column>": [values]}}`. It needs `inline_data = true` under `[mcp]`
(default off) and a set name that is not a source alias. Values travel in the signed, unencrypted plan id and are echoed in the `normalized_request` of explain. Inline
columns are unclassified, so the preview is empty unless you classify them in `[mcp.classification]`. Use it for trusted
callers only, since row counts over caller-chosen values can reveal plugin data;
`deny_feature_groups = ["ApiDataFG"]` turns it off entirely.
`max_inline_values` and `max_inline_chars` (integers, at least 1) refuse oversized inline data as `limit_exceeded`
before planning and again on run. Inline data is also capped by the plan id size (16384 JSON characters).

Every result is a one-line summary followed by the result as JSON, which is also the structured content. A failed
call has `isError` set and one error shape: `{"error": {"code", "message", "details", "audit_id"}}` with the codes
`validation_failed`, `policy_denied`, `limit_exceeded`, `plan_invalid`, `plan_stale`, `missing_input_feature` and
`internal`. Details point at the request by JSON pointer, with a suggestion when a close name exists. A
`missing_input_feature` detail holds `feature_group`, `feature` and `missing_inputs` (never the available columns).

## What stays on the server

- Sources are named by alias only. Paths, connection details, data-access objects and option values never appear in a
  result, and a request cannot set data-access options.
- A request may set only the option keys listed in `allow_options` (default: none). Any other key in `options`,
  `group_options` or `context_options`, nested `in_features` configs included, is refused as `policy_denied` by
  JSON pointer, never by value. Data-access keys stay refused even when listed. Options encoded in a chained feature
  name are not covered. `mloda explain` and `mloda run` are local and not restricted. `mloda_describe_feature` shows
  `caller_settable` for each option and refuses option keys not in `allow_options`.
- Denied FeatureGroups are disabled before matching, so their code does not run; the agent sees `policy_denied` and
  the alternatives you configured. Requested names, `index` columns, `in_features` source names and filter columns are
  checked before planning. Any planned feature name the deny list matches, such as a chained name over a denied
  column, is refused after planning.
- `run` re-plans the request: validation, policy, filters and row limits are checked again, and a plan that resolves
  differently is refused as `plan_stale`.
- With `max_rows` set, a request is refused as `limit_exceeded` when a configured source has more rows or cannot be
  counted (a CSV is counted exactly only when read on `PythonDictFramework`), when it reads data that a FeatureGroup
  loads itself (never countable), and, after `run`, when a result table has more rows.
- The preview holds only columns whose classification is in `preview_allows`. A derived feature takes the strictest
  classification of its inputs unless you classify it yourself. Unclassified columns are withheld.
- `keep_index: true` in a request requires `index`. Index columns are classified like any column.
- A filter on a column the preview withholds is refused as `policy_denied`: row counts would reveal its values.
- Descriptions and previewed values come from plugins and data: the server marks them untrusted.
- Each call writes one audit record to the `mloda.interfaces.mcp` logger, without argument values. The principal is the
  local OS user. Unexpected failures log a traceback without exception messages.

`default_classification` replaces `unclassified` for every leaf column without its own entry, including columns a
FeatureGroup loads itself; derived features still take the strictest input. A permissive label such as `public` exposes
all of them in the preview, so prefer listing columns in `[mcp.classification]`.

Upgrade note: MCP callers that set options are refused until you list the keys in `allow_options`.

## Local HTTP

```bash
mloda-mcp --transport http --host 127.0.0.1 --port 8000
```

The server is at `/mcp`. Bound to loopback, it accepts only loopback `Host` and `Origin` headers.

There is no authentication: a non-loopback host exposes the server to the network (the CLI warns). All clients share one
principal (the OS user) and one set of plan ids.

To embed it, call `mloda.interfaces.mcp.server.create_app(settings, env=env, host=...)`, which returns a Starlette app.
A mounted sub-app's lifespan does not run, so run the returned app's lifespan or serve it directly.

Not yet: paging through full results, approvals, cancellation, and remote HTTP with OAuth.

## Command line

```bash
mloda --version                     # report the installed package version
mloda explain request.json        # the same JSON as mloda_explain_request
mloda run request.json --max-rows 100000 --format jsonl
cat request.json | mloda explain -
mloda run request.json --full --format csv
mloda run income segment -o model=small -c timeout:=30
```

`mloda run FEATURE ... -o KEY=VALUE -c KEY=VALUE` builds the request from feature names: `-o` sets group options and
`-c` context options on every feature, and both repeat. A value is `KEY=VALUE` (string), `KEY:=JSON` (parsed JSON) or
`KEY=@FILE` (file text). A single target that is `-`, an existing file or ends in `.json` is a request file instead,
and `-o` or `-c` with it is an error.

`--full` returns every row and all columns, withheld ones included. `max_rows` still applies. The server never offers
it. `mloda explain` and `mloda run` do not refuse filters on withheld columns (the caller is local); `mloda mcp serve`
does.

`mloda run` executes regardless of `mode`, since you started it yourself, but applies the same limits and preview
rules. Exit codes: `0` success, `2` an error result or nothing could run (bad config, bad request, policy, limit, or a
FeatureGroup that failed while planning or running, reported as `internal`), `3` a completed run whose result
carries `failed_rows`, `1` any other failure.

A completed run result has a `failed_rows` list only when feature groups reported rows that got no value. Each entry has
`column`, `reason` (a short code), `row_count`, `rows` and `truncated`; `rows` holds at most 1000 positions unless
`--full` is used, and at most 100 entries are listed. The positions are rows of the feature group's own output. With
`--format jsonl` or `csv` the entries go to stderr as a short summary and stdout is unchanged.
