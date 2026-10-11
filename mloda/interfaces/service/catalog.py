"""Catalog search and describe over the FeatureGroups a service policy allows and its configured sources.

The FeatureGroup index is built once, scoped so denied FeatureGroups are never enumerated or matched.
"""

import base64
import difflib
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.property_spec import is_no_default
from mloda.core.abstract_plugins.components.utils import as_str, get_all_subclasses, safe_field
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import get_feature_group_docs, resolve_feature
from mloda.core.api.plugin_info import FeatureGroupInfo, ResolvedFeature
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import error_result, problems_to_details, strip_control_characters
from mloda.interfaces.service.request import MAX_NAME_LENGTH, Problem, option_data_access_violations

MAX_DESCRIPTION_LENGTH = 300
MAX_SEARCH_LIMIT = 50
DEFAULT_SEARCH_LIMIT = 20
MAX_SUPPORTED_NAMES = 20
KINDS = ("feature_group", "source")
_WHITESPACE = re.compile(r"\s+")
_CURSOR = re.compile(r"[A-Za-z0-9_-]{4,64}")
_CURSOR_BODY = re.compile(r"v1:(\d{1,9}):([0-9a-f]{16})")


def sanitize_text(text: str | None, limit: int = MAX_DESCRIPTION_LENGTH) -> str:
    """Strip control characters, collapse whitespace and truncate to ``limit`` with an ellipsis."""
    cleaned = _WHITESPACE.sub(" ", strip_control_characters(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def plugin_collector_for(settings: Settings, base: PluginCollector | None = None) -> PluginCollector:
    """Copy ``base`` (or allow everything) and disable every FeatureGroup whose class name is denied."""
    collector = PluginCollector()
    if base is not None:
        collector.add_enabled_feature_group_classes(base.enabled_feature_group_classes)
        collector.add_disabled_feature_group_classes(base.disabled_feature_group_classes)
        collector.allow_redefinition = base.allow_redefinition
        collector.strict_mode = base.strict_mode
        collector.registry = base.registry
    denied = {fg for fg in get_all_subclasses(FeatureGroup) if fg.__name__ in settings.deny_feature_groups}
    collector.add_disabled_feature_group_classes(denied)
    return collector


@dataclass(frozen=True)
class _Entry:
    """One indexed FeatureGroup; an empty description means none was declared."""

    name: str
    description: str
    prefix: str
    supported_names: tuple[str, ...]

    def item(self, match: str) -> dict[str, Any]:
        return {
            "kind": "feature_group",
            "feature_group": self.name,
            "description": self.description,
            "prefix": self.prefix,
            "supported_feature_names": list(self.supported_names[:MAX_SUPPORTED_NAMES]),
            "match": match,
        }

    def match(self, needle: str) -> tuple[int, str] | None:
        """Rank of the best match: name, then supported feature name, then description."""
        if not needle:
            return 0, "all"
        if needle in self.name.lower():
            return 0, "name"
        if any(needle in name.lower() for name in self.supported_names):
            return 1, "feature_name"
        if needle in self.description.lower():
            return 2, "description"
        return None


def _description(name: str, raw: str) -> str:
    # FeatureGroup.description() falls back to the class name when no docstring is declared.
    return "" if raw.strip() == name else sanitize_text(raw)


def _entry(info: FeatureGroupInfo) -> _Entry:
    return _Entry(
        name=info.name,
        description=_description(info.name, info.description),
        prefix=info.prefix,
        supported_names=tuple(sorted(info.supported_feature_names)),
    )


def _source_item(source: SourceConfig, match: str) -> dict[str, Any]:
    kind = "files" if not source.folders else "folders" if not source.files else "files_and_folders"
    return {"kind": "source", "alias": source.alias, "type": kind, "match": match}


def _fingerprint(query: str, kinds: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps([query, list(kinds)]).encode("utf-8")).hexdigest()[:16]


def _encode_cursor(offset: int, fingerprint: str) -> str:
    return base64.urlsafe_b64encode(f"v1:{offset}:{fingerprint}".encode("ascii")).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> int | None:
    """Offset of a cursor issued for this fingerprint, or None for a malformed or foreign one."""
    if not _CURSOR.fullmatch(cursor) or len(cursor) % 4 == 1:
        return None
    raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("ascii", errors="replace")
    match = _CURSOR_BODY.fullmatch(raw)
    if match is None or match.group(2) != fingerprint:
        return None
    return int(match.group(1))


def _invalid(path: str, message: str) -> dict[str, dict[str, Any]]:
    return error_result("validation_failed", message, problems_to_details([Problem(path, "invalid", message)]))


def _json_value(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _options(fg: type[FeatureGroup]) -> tuple[list[dict[str, Any]], list[str]]:
    options: list[dict[str, Any]] = []
    missing: list[str] = []
    mapping = fg.PROPERTY_MAPPING or {}
    for key in sorted(mapping, key=str):
        spec = mapping[key]
        entry: dict[str, Any] = {
            "name": str(key),
            "explanation": sanitize_text(spec.explanation),
            "required": is_no_default(spec.default),
        }
        if not is_no_default(spec.default):
            entry["default"] = _json_value(spec.default)
        if spec.allowed_values is not None:
            entry["allowed_values"] = [_json_value(value) for value in spec.allowed_values]
        if not entry["explanation"]:
            missing.append(f"explanation for option '{key}'")
        options.append(entry)
    return options, missing


class Catalog:
    """A search and describe index over the FeatureGroups and sources one Settings allows."""

    def __init__(self, settings: Settings, collector: PluginCollector, entries: Sequence[_Entry]) -> None:
        self._settings = settings
        self._collector = collector
        self._entries = tuple(entries)
        self._by_name = {entry.name: entry for entry in self._entries}
        self._known_names = sorted({name for entry in self._entries for name in entry.supported_names})

    @classmethod
    def build(cls, settings: Settings, base: PluginCollector | None = None) -> "Catalog":
        """Index the FeatureGroups ``base`` enables minus the denied ones, once."""
        collector = plugin_collector_for(settings, base)
        return cls(settings, collector, [_entry(info) for info in get_feature_group_docs(plugin_collector=collector)])

    def search(
        self,
        query: str = "",
        kinds: Sequence[str] | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """One page of ranked FeatureGroups and sources matching ``query``, with an opaque next cursor."""
        selected = sorted(set(KINDS if kinds is None else kinds))
        if any(kind not in KINDS for kind in selected):
            return _invalid("/kinds", f"'kinds' may only hold {list(KINDS)}.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
            return _invalid("/limit", f"'limit' must be an integer from 1 to {MAX_SEARCH_LIMIT}.")
        fingerprint = _fingerprint(query, selected)
        offset = 0
        if cursor is not None:
            decoded = _decode_cursor(cursor, fingerprint)
            if decoded is None:
                return _invalid("/cursor", "The cursor is malformed or belongs to another query.")
            offset = decoded
        items = self._matches(query.strip().lower(), selected)
        end = offset + limit
        return {
            "items": items[offset:end],
            "next_cursor": _encode_cursor(end, fingerprint) if end < len(items) else None,
        }

    def _matches(self, needle: str, kinds: Sequence[str]) -> list[dict[str, Any]]:
        ranked: list[tuple[int, dict[str, Any]]] = []
        if "feature_group" in kinds:
            for entry in self._entries:
                found = entry.match(needle)
                if found is not None:
                    ranked.append((found[0], entry.item(found[1])))
        items = [item for _, item in sorted(ranked, key=lambda pair: pair[0])]
        if "source" in kinds:
            for alias in sorted(self._settings.sources):
                if needle in alias.lower():
                    items.append(_source_item(self._settings.sources[alias], "name" if needle else "all"))
        return items

    def describe(self, name: str, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Describe the FeatureGroup that resolves ``name`` under ``options``, or explain why none does."""
        if not isinstance(name, str) or not 1 <= len(name) <= MAX_NAME_LENGTH:
            return _invalid("/name", f"'name' must be a string of 1 to {MAX_NAME_LENGTH} characters.")
        group = dict(options or {})
        violations = option_data_access_violations(group)
        if violations:
            details = problems_to_details(violations)
            return error_result("policy_denied", "A request may not set data-access options.", details)
        resolved = resolve_feature(name, options=Options(group=group), plugin_collector=self._collector)
        if resolved.feature_group is not None:
            return self._described(name, resolved.feature_group, resolved)
        if resolved.candidates:
            message = sanitize_text(resolved.error)
            return error_result(
                "validation_failed", message, [{"path": "/name", "problem": "unresolved", "message": message}]
            )
        if self.is_denied(name):
            alternatives = self._settings.alternatives.get(name)
            extra: dict[str, Any] = {"alternatives": list(alternatives)} if alternatives else {}
            return error_result("policy_denied", "This feature is not available under the service policy.", **extra)
        close = difflib.get_close_matches(name, self._known_names, n=1)
        problem = Problem(
            path="/name",
            problem="unknown_feature",
            message=f"No FeatureGroup provides '{sanitize_text(name, MAX_NAME_LENGTH)}'.",
            suggestion=close[0] if close else None,
        )
        return error_result("validation_failed", problem.message, problems_to_details([problem]))

    def _described(self, name: str, fg: type[FeatureGroup], resolved: ResolvedFeature) -> dict[str, Any]:
        fg_name = fg.get_class_name()
        entry = self._by_name.get(fg_name)
        description = (
            entry.description
            if entry is not None
            else _description(fg_name, safe_field(lambda: as_str(fg.description()), fg_name))
        )
        options, missing = _options(fg)
        in_features = None
        if issubclass(fg, FeatureChainParserMixin):
            in_features = {"min": fg.MIN_IN_FEATURES, "max": fg.MAX_IN_FEATURES}
        return {
            "name": name,
            "feature_group": fg_name,
            "candidates": [candidate.get_class_name() for candidate in resolved.candidates],
            "description": description,
            "options": options,
            "in_features": in_features,
            "metadata_missing": ([] if description else ["description"]) + missing,
        }

    def is_denied(self, name: str) -> bool:
        """Match ``name`` against denied FeatureGroups' class name, prefix and supported names, never their matchers."""
        enabled = self._collector.enabled_feature_group_classes
        for fg in self._collector.disabled_feature_group_classes:
            if fg.__name__ not in self._settings.deny_feature_groups or (enabled and fg not in enabled):
                continue
            prefix = safe_field(fg.prefix, f"{fg.__name__}_")
            if name == fg.__name__ or (prefix and name.startswith(prefix)):
                return True
            supported: set[str] = safe_field(fg.feature_names_supported, set())
            if name in supported:
                return True
        return False
