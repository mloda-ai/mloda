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
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from mloda.core.abstract_plugins.components.utils import safe_field
from mloda.interfaces.service.catalog import MAX_DESCRIPTION_LENGTH, Catalog, plugin_collector_for, sanitize_text
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import error_result, problems_to_details
from mloda.interfaces.service.lock import MLODA_LOCK
from mloda.interfaces.service.request import (
    MAX_NAME_LENGTH,
    ParsedRequest,
    build_request,
    data_access_violations,
    validate_request,
)
from mloda.interfaces.service.sources import READER_ERRORS, data_access_collection, source_columns
from mloda.provider import ReadFileFG, SourceMatch
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
_DENIED_MESSAGE = "This feature is not available under the service policy."
_UNPLANNABLE_MESSAGE = "The request cannot be planned under the service policy and sources."


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
        return self._described(planned)

    def plan(self, request: Any) -> PlannedRequest | dict[str, Any]:
        """Validate, policy-check and plan ``request`` within the row limits, or return the error result."""
        problems = validate_request(request, self._settings)
        if problems:
            return error_result("validation_failed", "The request is invalid.", problems_to_details(problems))
        violations = data_access_violations(request)
        if violations:
            details = problems_to_details(violations)
            return error_result("policy_denied", "A request may not set data-access options.", details)
        parsed = build_request(request)
        if isinstance(parsed, list):
            return error_result("validation_failed", "A feature config is invalid.", problems_to_details(parsed))
        request_names = [_item_name(item) for item in request["features"]]
        denied = self._denied(request_names)
        if denied is not None:
            return denied
        sources = self._sources(parsed.restrict_sources)
        with verified_context(principal=self._principal), MLODA_LOCK:
            PluginLoader.all()
            diagnosis = mloda.diagnose(parsed.features, **self._plan_kwargs(parsed, sources))
            steps = mloda.explain(parsed.features, **self._plan_kwargs(parsed, sources)) if diagnosis.complete else []
        if not diagnosis.complete:
            return self._unresolved(diagnosis, request_names, sources)
        unmatched = self._unmatched_filters(parsed, request.get("filters", []))
        if unmatched is not None:
            return unmatched
        aliases = tuple(_alias_of(step, sources) for step in steps)
        estimate = self._estimate(steps, aliases)
        limited = self._limit_exceeded(estimate["sources"])
        if limited is not None:
            return limited
        return PlannedRequest(
            parsed=parsed,
            names=tuple(dict.fromkeys(_feature_name(feature) for feature in parsed.features)),
            sources=sources,
            steps=tuple(steps),
            aliases=aliases,
            producers=_producers(steps, aliases),
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
            "plugin_collector": plugin_collector_for(self._settings, self._base),
            "parallelization_modes": {ParallelizationMode.SYNC},
        }

    def _denied(self, names: Sequence[str]) -> dict[str, Any] | None:
        denied = [(index, name) for index, name in enumerate(names) if self._catalog.is_denied(name)]
        if not denied:
            return None
        alternatives = [alt for _, name in denied for alt in self._settings.alternatives.get(name, ())]
        extra: dict[str, Any] = {"alternatives": list(dict.fromkeys(alternatives))} if alternatives else {}
        details = [
            {"path": f"/features/{index}", "problem": "policy_denied", "message": _DENIED_MESSAGE}
            for index, _ in denied
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
        candidates = [candidate for candidate in dict.fromkeys([*known, *source_columns(sources)]) if candidate != name]
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

    def _described(self, planned: PlannedRequest) -> dict[str, Any]:
        expires = math.floor(self._clock()) + self._settings.plan_ttl_seconds
        steps = zip(planned.steps, planned.aliases)
        return {
            "plan_id": self._sign(planned.parsed.normalized, planned.digest, expires),
            "expires_at": datetime.fromtimestamp(expires, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "normalized_request": planned.parsed.normalized,
            "steps": [_step_record(index, step, alias) for index, (step, alias) in enumerate(steps, 1)],
            "lineage": {name: sorted(self._lineage(name, planned.producers, frozenset())) for name in planned.names},
            "estimate": planned.estimate,
            "egress": self._egress(planned.names, planned.producers),
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
            return UNCLASSIFIED
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

    def _estimate(self, steps: Sequence[PlanStep], aliases: Sequence[str | None]) -> dict[str, Any]:
        reads: dict[str, dict[str, PlanStep]] = {}
        for step, alias in zip(steps, aliases):
            identity = _file_identity(step)
            if alias is not None and identity is not None and step.compute_framework is not None:
                reads.setdefault(alias, {}).setdefault(identity, step)
        totals = {
            alias: _total(self._count(path, step) for path, step in sorted(reads[alias].items())) for alias in reads
        }
        entries = [
            {"alias": alias, "rows": totals[alias], "rows_exact": totals[alias] is not None} for alias in sorted(totals)
        ]
        rows_read = _total(totals.values())
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
            if entry["rows"] is None:
                details.append({"source": entry["alias"], "reason": "uncountable", "max_rows": max_rows})
            elif entry["rows"] > max_rows:
                details.append(
                    {"source": entry["alias"], "reason": "over_limit", "rows": entry["rows"], "max_rows": max_rows}
                )
        if not details:
            return None
        names = ", ".join(detail["source"] for detail in details)
        message = f"Sources exceed max_rows ({max_rows}) or cannot be counted: {names}."
        return error_result("limit_exceeded", message, details)
