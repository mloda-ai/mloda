"""Plan a feature request with mloda and describe the plan without configured paths, data or core messages.

A successful plan carries a signed, expiring plan id bound to the signing key, the settings and the principal.
"""

import base64
import difflib
import hashlib
import hmac
import json
import math
import re
import secrets
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.interfaces.service.catalog import MAX_DESCRIPTION_LENGTH, Catalog, plugin_collector_for, sanitize_text
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import error_result, failure_result, problems_to_details
from mloda.interfaces.service.lock import MLODA_LOCK
from mloda.interfaces.service.request import (
    MAX_NAME_LENGTH,
    ParsedRequest,
    build_request,
    caller_refused_option_keys,
    _pointer,
    data_access_violations,
    refused_option_violations,
    unlisted_option_violations,
    in_feature_sources,
    validate_request,
)
from mloda.interfaces.service.sources import READER_ERRORS, data_access_collection, source_columns
from mloda.provider import ApiDataFG, ReadFileFG, SourceMatch
from mloda.steward import verified_context
from mloda.user import (
    Feature,
    ParallelizationMode,
    PlanStep,
    PluginCollector,
    PluginLoader,
    ResolutionDiagnosis,
    RunResult,
    mloda,
)

# A token format tag, not a secret.
PLAN_TOKEN_PREFIX = "mpl1"  # nosec B105
MAX_PLAN_ID_LENGTH = 16384
UNCLASSIFIED = "unclassified"
STRICTNESS = ("public", "internal", "confidential", "personal", UNCLASSIFIED)
ROWS_RETURNED_NOTE = "Rows returned depend on filters and joins; they are known only after execution."
_SEGMENT = re.compile(r"[A-Za-z0-9_-]+")
_PAYLOAD_KEYS = frozenset({"kid", "req", "digest", "cfg", "sub", "exp"})
_TOO_LARGE_MESSAGE = "The request is too large for a plan id."
_DENIED_MESSAGE = "This feature is not available under the service policy."
_UNPLANNABLE_MESSAGE = "The request cannot be planned under the service policy and sources."
_WITHHELD_FILTER_MESSAGE = "Filters on columns the preview withholds are refused; row counts would reveal their values."


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(segment: str) -> bytes | None:
    if not _SEGMENT.fullmatch(segment) or len(segment) % 4 == 1:
        return None
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _item_name(item: Any) -> str:
    return item if isinstance(item, str) else str(item["name"])


def _feature_name(feature: Feature | str) -> str:
    return feature if isinstance(feature, str) else str(feature.name)


def _strictness(label: str) -> int:
    return STRICTNESS.index(label) if label in STRICTNESS else len(STRICTNESS)


def _file_identity(step: PlanStep) -> str | None:
    fg = step.feature_group
    if not (isinstance(fg, type) and issubclass(fg, ReadFileFG)):
        return None
    return step.data_access_identity


def _alias_of(step: PlanStep, sources: Mapping[str, SourceConfig]) -> str | None:
    identity = _file_identity(step)
    if identity is None:
        return None
    path = Path(identity)
    for alias in sorted(sources):
        source = sources[alias]
        if str(path) in source.files or any(path.is_relative_to(folder) for folder in source.folders):
            return alias
    return None


def _inline_alias(step: PlanStep, inline: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The inline set owning the columns of an api_data step."""
    group = step.feature_group
    if not (isinstance(group, type) and issubclass(group, ApiDataFG)):
        return None
    if not (step.data_access_identity or "").startswith("api_data:"):
        return None
    return next((name for name in sorted(inline) if set(step.feature_names) & set(inline[name])), None)


def _is_plugin_root(step: PlanStep, alias: str | None) -> bool:
    """A compute step that reads data itself: no declared inputs and no configured source."""
    has_inputs = bool(step.input_feature_names) or any(step.input_feature_edges.values())
    return step.step_kind == "compute" and alias is None and not has_inputs


def _inputs(step: PlanStep, name: str) -> tuple[str, ...]:
    return step.input_feature_edges.get(name, step.input_feature_names)


def _step_record(index: int, step: PlanStep, alias: str | None) -> dict[str, Any]:
    reason = step.compute_framework_reason
    return {
        "step": index,
        "kind": step.step_kind,
        "features": sorted(step.feature_names),
        "feature_group": step.feature_group_name,
        "compute_framework": step.compute_framework_name,
        "compute_framework_reason": None if reason is None else sanitize_text(reason, MAX_DESCRIPTION_LENGTH),
        "specialized_from": [fg.get_class_name() for fg in step.specialized_from],
        "source": alias,
        "inputs": list(step.input_feature_names),
    }


def _digest(steps: Sequence[PlanStep]) -> str:
    rows = [[s.step_kind, sorted(s.feature_names), s.feature_group_name, s.compute_framework_name] for s in steps]
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def _total(counts: Iterable[int | None]) -> int | None:
    known = list(counts)
    return None if None in known else sum(count for count in known if count is not None)


def _plan_invalid(message: str) -> dict[str, Any]:
    return error_result("plan_invalid", message)


def _producers(steps: Sequence[PlanStep], aliases: Sequence[str | None]) -> dict[str, tuple[PlanStep, str | None]]:
    producers: dict[str, tuple[PlanStep, str | None]] = {}
    for step, alias in zip(steps, aliases):
        for name in step.feature_names:
            producers.setdefault(name, (step, alias))
    return producers


@dataclass(frozen=True)
class PlannedRequest:
    """A request that passed validation, policy, filter and row-limit checks, with its resolved plan."""

    parsed: ParsedRequest
    names: tuple[str, ...]
    sources: Mapping[str, SourceConfig]
    steps: tuple[PlanStep, ...]
    aliases: tuple[str | None, ...]
    producers: Mapping[str, tuple[PlanStep, str | None]]
    estimate: dict[str, Any]
    digest: str


class Planner:
    """Plans feature requests under one Settings and signs each plan into a verifiable plan id."""

    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        *,
        signing_key: bytes | None = None,
        principal: str | None = None,
        plugin_collector: PluginCollector | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._key = signing_key if signing_key is not None else secrets.token_bytes(32)
        self._kid = hashlib.sha256(self._key).hexdigest()[:16]
        self._principal = principal
        self._base = plugin_collector
        self._clock = clock

    @property
    def settings(self) -> Settings:
        """The settings this planner plans under."""
        return self._settings

    def explain(self, request: Any) -> dict[str, Any]:
        """Plan ``request`` and describe the plan, or return an error result."""
        planned = self.plan(request)
        if isinstance(planned, dict):
            return planned
        described = self._described(planned)
        if len(described["plan_id"]) > MAX_PLAN_ID_LENGTH:
            return error_result("limit_exceeded", _TOO_LARGE_MESSAGE)
        return described

    def plan(self, request: Any) -> PlannedRequest | dict[str, Any]:
        """Validate, policy-check and plan ``request`` within the row limits, or return the error result."""
        problems = validate_request(request, self._settings)
        if problems:
            return error_result("validation_failed", "The request is invalid.", problems_to_details(problems))
        refused = self._inline_refusal(request)
        if refused is not None:
            return refused
        violations = data_access_violations(request)
        if violations:
            details = problems_to_details(violations)
            return error_result("policy_denied", "A request may not set data-access options.", details)
        if not self._settings.trusted_caller:
            locked = refused_option_violations(request)
            unlisted = unlisted_option_violations(request, self._settings.allow_options)
            if locked:
                message = "A request may not set options a plugin marks as not caller-settable."
                return error_result("policy_denied", message, problems_to_details([*locked, *unlisted]))
            if unlisted:
                details = problems_to_details(unlisted)
                return error_result("policy_denied", "A request may only set options listed in allow_options.", details)
        parsed = build_request(request, settings=self._settings)
        if isinstance(parsed, list):
            return error_result("validation_failed", "A feature config is invalid.", problems_to_details(parsed))
        request_names = [_item_name(item) for item in request["features"]]
        names = [(f"/features/{index}", name) for index, name in enumerate(request_names)]
        filter_entries = [
            (f"/filters/{i}/feature", item["feature"]) for i, item in enumerate(request.get("filters", []))
        ]
        denied = self._denied([*names, *self._index_entries(request), *in_feature_sources(request), *filter_entries])
        if denied is not None:
            return denied
        sources = self._sources(parsed.restrict_sources)
        try:  # plugin boundary: plugin code runs during planning and its exception text may hold data
            with verified_context(principal=self._principal), MLODA_LOCK:
                PluginLoader.all()
                diagnosis = mloda.diagnose(parsed.features, **self._plan_kwargs(parsed, sources))
                steps = (
                    mloda.explain(parsed.features, **self._plan_kwargs(parsed, sources)) if diagnosis.complete else []
                )
        except Exception as error:
            return failure_result("Planning", error)
        if not diagnosis.complete:
            return self._unresolved(diagnosis, request_names, sources)
        planned = self._denied([("/features", name) for step in steps for name in step.feature_names])
        if planned is not None:
            return planned
        if not self._settings.trusted_caller:
            bound = self._name_bound_refusal(steps, request_names)
            if bound is not None:
                return bound
        filters = request.get("filters", [])
        unmatched = self._unmatched_filters(parsed, filters)
        if unmatched is not None:
            return unmatched
        aliases = tuple(_alias_of(step, sources) or _inline_alias(step, parsed.inline_data) for step in steps)
        producers = _producers(steps, aliases)
        withheld = None if self._settings.trusted_caller else self._withheld_filters(filters, producers)
        if withheld is not None:
            return withheld
        estimate = self._estimate(steps, aliases, parsed.inline_data)
        limited = self._limit_exceeded(estimate["sources"])
        if limited is not None:
            return limited
        return PlannedRequest(
            parsed=parsed,
            names=tuple(dict.fromkeys(_feature_name(feature) for feature in parsed.features)),
            sources=sources,
            steps=tuple(steps),
            aliases=aliases,
            producers=producers,
            estimate=estimate,
            digest=_digest(steps),
        )

    def execute(self, planned: PlannedRequest) -> RunResult:
        """Run a planned request with mloda under the service lock and this planner's principal."""
        with verified_context(principal=self._principal), MLODA_LOCK:
            PluginLoader.all()
            kwargs = self._plan_kwargs(planned.parsed, planned.sources)
            return mloda.run_all(planned.parsed.features, output_framework="PythonDictFramework", **kwargs)

    def classifications(self, names: Sequence[str], steps: Sequence[PlanStep]) -> dict[str, str]:
        """Each name's classification: explicit in the settings, else the strictest of its inputs'."""
        producers = _producers(steps, [_alias_of(step, self._settings.sources) for step in steps])
        return {name: self._classification(name, producers, frozenset()) for name in names}

    def verify_plan_id(self, plan_id: Any) -> dict[str, Any]:
        """The request and plan digest of an unexpired plan id this planner issued, else a plan_invalid error."""
        if not isinstance(plan_id, str) or len(plan_id) > MAX_PLAN_ID_LENGTH:
            return _plan_invalid("The plan id is malformed.")
        parts = plan_id.split(".")
        if len(parts) != 3 or parts[0] != PLAN_TOKEN_PREFIX or _unb64(parts[2]) is None:
            return _plan_invalid("The plan id is malformed.")
        body = _unb64(parts[1])
        if body is None:
            return _plan_invalid("The plan id is malformed.")
        if not hmac.compare_digest(parts[2], self._signature(parts[1])):
            return _plan_invalid("The plan id was not issued by this server or was altered.")
        payload = json.loads(body.decode("utf-8", errors="replace"))
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS or payload["kid"] != self._kid:
            return _plan_invalid("The plan id is malformed.")
        if payload["cfg"] != self._settings.config_hash():
            return _plan_invalid("The plan id was issued under different settings; plan the request again.")
        if payload["sub"] != self._principal:
            return _plan_invalid("The plan id was issued to a different principal.")
        expires = payload["exp"]
        if not isinstance(expires, int) or isinstance(expires, bool) or self._clock() >= expires:
            return _plan_invalid("The plan id has expired; plan the request again.")
        if not isinstance(payload["req"], dict) or not isinstance(payload["digest"], str):
            return _plan_invalid("The plan id is malformed.")
        return {"request": payload["req"], "digest": payload["digest"]}

    def _signature(self, body: str) -> str:
        message = f"{PLAN_TOKEN_PREFIX}.{body}".encode("ascii")
        return _b64(hmac.new(self._key, message, hashlib.sha256).digest())

    def _sign(self, normalized: dict[str, Any], digest: str, expires: int) -> str:
        payload = {
            "kid": self._kid,
            "req": normalized,
            "digest": digest,
            "cfg": self._settings.config_hash(),
            "sub": self._principal,
            "exp": expires,
        }
        body = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return f"{PLAN_TOKEN_PREFIX}.{body}.{self._signature(body)}"

    def _sources(self, restrict: Sequence[str]) -> dict[str, SourceConfig]:
        if not restrict:
            return dict(self._settings.sources)
        return {alias: self._settings.sources[alias] for alias in restrict}

    def _plan_kwargs(self, parsed: ParsedRequest, sources: Mapping[str, SourceConfig]) -> dict[str, Any]:
        return {
            "compute_frameworks": list(self._settings.compute_frameworks),
            "data_access_collection": data_access_collection(sources),
            "global_filter": parsed.global_filter,
            "api_data": dict(parsed.inline_data) or None,
            "plugin_collector": plugin_collector_for(self._settings, self._base),
            "parallelization_modes": {ParallelizationMode.SYNC},
        }

    def _inline_refusal(self, request: Any) -> dict[str, Any] | None:
        if "data" not in request:
            return None
        if not self._settings.inline_data:
            detail = {"path": "/data", "problem": "policy_denied", "message": "Inline data is not enabled."}
            return error_result("policy_denied", "Inline data is not enabled under the service policy.", [detail])
        if len(json.dumps(request["data"])) > MAX_PLAN_ID_LENGTH:
            return error_result("limit_exceeded", _TOO_LARGE_MESSAGE)
        over = self._inline_over_limit(request["data"])
        if over:
            return error_result("limit_exceeded", "Inline data is over the configured limit.", over)
        details = [
            {
                "path": _pointer("data", name),
                "problem": "invalid",
                "message": "An inline set may not reuse a source alias.",
            }
            for name in request["data"]
            if name in self._settings.sources
        ]
        if not details:
            return None
        return error_result("validation_failed", "An inline set is named like a configured source.", details)

    def _inline_over_limit(self, data: Any) -> list[dict[str, Any]]:
        columns = [
            column
            for table in (data.values() if isinstance(data, dict) else ())
            if isinstance(table, dict)
            for column in table.values()
            if isinstance(column, list)
        ]
        counts = {
            "inline_values_over_limit": (sum(len(column) for column in columns), self._settings.max_inline_values),
            "inline_chars_over_limit": (
                sum(len(value) for column in columns for value in column if isinstance(value, str)),
                self._settings.max_inline_chars,
            ),
        }
        return [
            {"reason": reason, "count": count, "limit": limit}
            for reason, (count, limit) in counts.items()
            if limit is not None and count > limit
        ]

    @staticmethod
    def _index_entries(request: Any) -> list[tuple[str, str]]:
        """Each declared index column of a feature config item with its pointer."""
        return [
            (f"/features/{i}/index/{j}", column)
            for i, item in enumerate(request["features"])
            if isinstance(item, dict)
            for j, column in enumerate(item.get("index", ()))
        ]

    def _denied(self, entries: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
        denied = [(path, name) for path, name in entries if self._catalog.is_denied(name)]
        if not denied:
            return None
        alternatives = [alt for _, name in denied for alt in self._settings.alternatives.get(name, ())]
        extra: dict[str, Any] = {"alternatives": list(dict.fromkeys(alternatives))} if alternatives else {}
        details = [
            {"path": path, "problem": "policy_denied", "message": _DENIED_MESSAGE}
            for path in dict.fromkeys(p for p, _ in denied)
        ]
        return error_result("policy_denied", _DENIED_MESSAGE, details, **extra)

    def _unresolved(
        self, diagnosis: ResolutionDiagnosis, names: Sequence[str], sources: Mapping[str, SourceConfig]
    ) -> dict[str, Any]:
        name = diagnosis.feature_name
        if name is None:
            # The core message may name configured paths, so it is never forwarded.
            detail = {"path": "/features", "problem": "unresolved", "message": _UNPLANNABLE_MESSAGE}
            return error_result("validation_failed", _UNPLANNABLE_MESSAGE, [detail])
        known = diagnosis.failed_result.facts.known_names if diagnosis.failed_result is not None else ()
        candidates = [
            candidate
            for candidate in dict.fromkeys([*known, *source_columns(sources)])
            if candidate != name and not self._catalog.is_denied(candidate)
        ]
        close = difflib.get_close_matches(name, candidates, n=1)
        shown = sanitize_text(name, MAX_NAME_LENGTH)
        detail = {
            "path": f"/features/{names.index(name)}" if name in names else "/features",
            "problem": "unresolved",
            "message": f"No FeatureGroup resolves '{shown}' under the service policy and sources.",
        }
        if close:
            detail["suggestion"] = close[0]
        return error_result("validation_failed", detail["message"], [detail])

    def _name_bound_refusal(self, steps: Sequence[PlanStep], request_names: Sequence[str]) -> dict[str, Any] | None:
        """Refuse a planned feature name that binds an option a plugin marks as not caller-settable."""
        keys = caller_refused_option_keys()
        if not keys:
            return None
        details: list[dict[str, str]] = []
        for step in steps:
            group = step.feature_group
            if step.step_kind != "compute" or not (isinstance(group, type) and issubclass(group, FeatureGroup)):
                continue
            for name in step.feature_names:
                path = f"/features/{request_names.index(name)}" if name in request_names else "/features"
                for key in sorted(set(group.name_bindings(name)) & keys):
                    detail = {
                        "path": path,
                        "problem": "option_not_caller_settable",
                        "message": f"Option key '{key}' is bound from a feature name and not caller-settable.",
                    }
                    if detail not in details:
                        details.append(detail)
        if not details:
            return None
        return error_result(
            "policy_denied", "A feature name may not bind options a plugin marks as not caller-settable.", details
        )

    def _unmatched_filters(self, parsed: ParsedRequest, filters: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        if parsed.global_filter is None:
            return None
        matched = parsed.global_filter.matched_filter_uuids
        details = [
            {
                "path": f"/filters/{index}",
                "problem": "unmatched_filter",
                "message": f"No planned FeatureGroup applies the filter on '{sanitize_text(filters[index]['feature'])}'.",
            }
            for index, uuid in enumerate(parsed.filter_uuids)
            if uuid not in matched
        ]
        if not details:
            return None
        return error_result("validation_failed", "A filter matches no planned feature.", details)

    def _withheld_filters(
        self, filters: Sequence[Mapping[str, Any]], producers: Mapping[str, tuple[PlanStep, str | None]]
    ) -> dict[str, Any] | None:
        details = [
            {
                "path": f"/filters/{index}",
                "problem": "filter_on_withheld_column",
                "message": f"The filter on '{sanitize_text(item['feature'])}' uses a column the preview withholds.",
            }
            for index, item in enumerate(filters)
            if self._classification(item["feature"], producers, frozenset()) not in self._settings.preview_allows
        ]
        if not details:
            return None
        return error_result("policy_denied", _WITHHELD_FILTER_MESSAGE, details)

    def _described(self, planned: PlannedRequest) -> dict[str, Any]:
        expires = math.floor(self._clock()) + self._settings.plan_ttl_seconds
        steps = zip(planned.steps, planned.aliases)
        kept = (
            column
            for feature in planned.parsed.features
            if isinstance(feature, Feature) and feature.keep_index and feature.index is not None
            for column in feature.index.index
        )
        names = tuple(dict.fromkeys([*planned.names, *kept]))
        return {
            "plan_id": self._sign(planned.parsed.normalized, planned.digest, expires),
            "expires_at": datetime.fromtimestamp(expires, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "normalized_request": planned.parsed.normalized,
            "steps": [_step_record(index, step, alias) for index, (step, alias) in enumerate(steps, 1)],
            "lineage": {name: sorted(self._lineage(name, planned.producers, frozenset())) for name in names},
            "estimate": planned.estimate,
            "egress": self._egress(names, planned.producers),
            "policy": {"verdict": "allowed"},
        }

    def _lineage(
        self, name: str, producers: Mapping[str, tuple[PlanStep, str | None]], seen: frozenset[str]
    ) -> set[str]:
        if name in seen or name not in producers:
            return set()
        step, alias = producers[name]
        if alias is not None:
            return {f"{alias}.{name}"}
        found: set[str] = set()
        for source in _inputs(step, name):
            found |= self._lineage(source, producers, seen | {name})
        return found

    def _classification(
        self, name: str, producers: Mapping[str, tuple[PlanStep, str | None]], seen: frozenset[str]
    ) -> str:
        explicit = self._settings.classification.get(name)
        if explicit is not None:
            return explicit
        inputs = _inputs(producers[name][0], name) if name in producers and name not in seen else ()
        if not inputs:
            return self._settings.default_classification or UNCLASSIFIED
        labels = [self._classification(source, producers, seen | {name}) for source in inputs]
        return max(labels, key=_strictness)

    def _egress(self, names: Sequence[str], producers: Mapping[str, tuple[PlanStep, str | None]]) -> dict[str, Any]:
        preview: list[str] = []
        withheld: list[dict[str, str]] = []
        for name in names:
            label = self._classification(name, producers, frozenset())
            if label in self._settings.preview_allows:
                preview.append(name)
            else:
                withheld.append({"column": name, "reason": f"classification: {label}"})
        return {"preview_columns": preview, "withheld": withheld}

    def _estimate(
        self, steps: Sequence[PlanStep], aliases: Sequence[str | None], inline: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        reads: dict[str, dict[str, PlanStep]] = {}
        for step, alias in zip(steps, aliases):
            identity = _file_identity(step)
            if alias is not None and identity is not None and step.compute_framework is not None:
                reads.setdefault(alias, {}).setdefault(identity, step)
        totals = {
            alias: _total(self._count(path, step) for path, step in sorted(reads[alias].items())) for alias in reads
        }
        for step, alias in zip(steps, aliases):
            if alias is not None and alias in inline and alias not in totals:
                totals[alias] = len(next(iter(inline[alias].values())))
        entries: list[dict[str, Any]] = [
            {"alias": alias, "rows": totals[alias], "rows_exact": totals[alias] is not None} for alias in sorted(totals)
        ]
        plugin_reads = sorted(
            {str(step.feature_group_name) for step, alias in zip(steps, aliases) if _is_plugin_root(step, alias)}
        )
        entries += [{"alias": None, "feature_group": name, "rows": None, "rows_exact": False} for name in plugin_reads]
        rows_read = None if plugin_reads else _total(totals.values())
        return {
            "sources": entries,
            "rows_read": rows_read,
            "rows_returned": None,
            "rows_returned_note": ROWS_RETURNED_NOTE,
        }

    @staticmethod
    def _count(path: str, step: PlanStep) -> int | None:
        fg = step.feature_group
        if not (isinstance(fg, type) and issubclass(fg, ReadFileFG)) or step.compute_framework is None:
            return None
        count = partial(fg.count_rows, SourceMatch(source=path, access=path), step.compute_framework)
        return safe_field(count, None, catching=READER_ERRORS)

    def _limit_exceeded(self, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        max_rows = self._settings.max_rows
        if max_rows is None:
            return None
        details: list[dict[str, Any]] = []
        for entry in entries:
            if entry["alias"] is None:
                plugin = {"source": None, "feature_group": entry["feature_group"]}
                details.append({**plugin, "reason": "uncountable", "max_rows": max_rows})
            elif entry["rows"] is None:
                details.append({"source": entry["alias"], "reason": "uncountable", "max_rows": max_rows})
            elif entry["rows"] > max_rows:
                details.append(
                    {"source": entry["alias"], "reason": "over_limit", "rows": entry["rows"], "max_rows": max_rows}
                )
        if not details:
            return None
        names = ", ".join(detail["source"] or detail["feature_group"] for detail in details)
        message = f"Sources exceed max_rows ({max_rows}) or cannot be counted: {names}."
        return error_result("limit_exceeded", message, details)
