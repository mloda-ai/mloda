"""Feature-request contract: JSON Schema, a matching hand validator, and the build into mloda objects.

The schema and the validator derive from the same constants so they accept exactly the same requests.
"""

import copy
import difflib
import json
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.feature_group import feature_group_subclasses
from mloda.core.abstract_plugins.components.input_data.source_feature_group import concrete_source_groups
from mloda.core.abstract_plugins.components.input_data.read_db_fg import QUERY_OPTION
from mloda.core.api.feature_config.loader import load_features_from_config
from mloda.core.api.feature_config.models import feature_config_schema
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.filter.single_filter import SingleFilter
from mloda.interfaces.service.config import Settings

SCHEMA_ID = "mloda://schema/feature-request"
MAX_FEATURES = 50
MAX_FILTERS = 20
MAX_NAME_LENGTH = 200

_NAME_SCHEMA: dict[str, Any] = {"type": "string", "minLength": 1, "maxLength": MAX_NAME_LENGTH}
_SCALAR_SCHEMA: dict[str, Any] = {"type": ["string", "number", "boolean"]}
_OP_REQUIRES_ANY: dict[str, tuple[str, ...]] = {"eq": ("value",), "in": ("values",), "range": ("min", "max")}
_FILTER_PROPERTIES: dict[str, dict[str, Any]] = {
    "feature": _NAME_SCHEMA,
    "op": {"enum": list(_OP_REQUIRES_ANY)},
    "value": _SCALAR_SCHEMA,
    "values": {"type": "array", "items": _SCALAR_SCHEMA},
    "min": {"type": "number"},
    "max": {"type": "number"},
    "max_exclusive": {"type": "boolean"},
}
_FILTER_REQUIRED = ["feature", "op"]
_OP_PARAMETERS: dict[str, tuple[str, ...]] = {
    "eq": ("value",),
    "in": ("values",),
    "range": ("min", "max", "max_exclusive"),
}
_OP_FILTER_TYPES = {"eq": FilterType.EQUAL, "in": FilterType.CATEGORICAL_INCLUSION, "range": FilterType.RANGE}
_RESTRICT_SOURCES_SCHEMA: dict[str, Any] = {"type": "array", "uniqueItems": True, "items": {"type": "string"}}
_DATA_VALUE_SCHEMA: dict[str, Any] = {"type": ["string", "number", "boolean", "null"]}
_DATA_COLUMN_SCHEMA: dict[str, Any] = {"type": "array", "items": _DATA_VALUE_SCHEMA}
_DATA_SET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "minProperties": 1,
    "propertyNames": _NAME_SCHEMA,
    "additionalProperties": _DATA_COLUMN_SCHEMA,
}
_DATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "propertyNames": _NAME_SCHEMA,
    "additionalProperties": _DATA_SET_SCHEMA,
}
_OPTION_CONTAINERS = ("options", "group_options", "context_options")


@dataclass(frozen=True)
class Problem:
    """One request problem located by a JSON pointer."""

    path: str
    problem: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True)
class ParsedRequest:
    """A request built into mloda Features, an optional GlobalFilter, each filter's uuid and a canonical form."""

    features: list[Feature | str]
    global_filter: GlobalFilter | None
    restrict_sources: tuple[str, ...]
    normalized: dict[str, Any]
    filter_uuids: tuple[UUID, ...] = ()
    inline_data: Mapping[str, Mapping[str, list[Any]]] = field(default_factory=dict)


def _feature_config_def() -> dict[str, Any]:
    schema = copy.deepcopy(feature_config_schema())
    schema["properties"]["name"] = dict(_NAME_SCHEMA)
    schema["properties"]["index"]["items"] = dict(_NAME_SCHEMA)
    schema["if"] = {"properties": {"keep_index": {"const": True}}, "required": ["keep_index"]}
    schema["then"] = {"required": ["index"]}
    return schema


_FEATURE_CONFIG = _feature_config_def()


def request_schema() -> dict[str, Any]:
    """Return the draft 2020-12 JSON Schema of a feature request."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "type": "object",
        "additionalProperties": False,
        "required": ["features"],
        "properties": {
            "features": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_FEATURES,
                "items": {"oneOf": [dict(_NAME_SCHEMA), {"$ref": "#/$defs/featureConfig"}]},
            },
            "filters": {"type": "array", "maxItems": MAX_FILTERS, "items": {"$ref": "#/$defs/filter"}},
            "restrict_sources": copy.deepcopy(_RESTRICT_SOURCES_SCHEMA),
            "data": copy.deepcopy(_DATA_SCHEMA),
        },
        "$defs": {
            "featureConfig": copy.deepcopy(_FEATURE_CONFIG),
            "filter": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_FILTER_REQUIRED),
                "properties": copy.deepcopy(_FILTER_PROPERTIES),
                "allOf": [
                    {
                        "if": {"properties": {"op": {"const": op}}, "required": ["op"]},
                        "then": {"anyOf": [{"required": [key]} for key in keys]},
                    }
                    for op, keys in _OP_REQUIRES_ANY.items()
                ],
            },
        },
    }


def _pointer(*parts: str | int) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _is_type(value: Any, name: str) -> bool:
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        if isinstance(value, float):
            return value.is_integer()
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "null":
        return value is None
    expected = {"string": str, "boolean": bool, "object": dict, "array": list}[name]
    return isinstance(value, expected)


def _conforms(value: Any, schema: Mapping[str, Any]) -> bool:
    """Evaluate the keywords the schemas use: type, enum, min/maxLength, minimum, min/uniqueItems, oneOf, items."""
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else types
        if not any(_is_type(value, name) for name in names):
            return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, str) and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
        return False
    if isinstance(value, list) and len(value) < schema.get("minItems", 0):
        return False
    if isinstance(value, list) and schema.get("uniqueItems") and len({json.dumps(item) for item in value}) < len(value):
        return False
    if _is_type(value, "number") and value < schema.get("minimum", value):
        return False
    if "oneOf" in schema and sum(_conforms(value, sub) for sub in schema["oneOf"]) != 1:
        return False
    if isinstance(value, list) and "items" in schema:
        return all(_conforms(item, schema["items"]) for item in value)
    return True


def _invalid(path: str, message: str) -> Problem:
    return Problem(path=path, problem="invalid", message=message)


def _object_problems(
    value: dict[str, Any], path: str, properties: Mapping[str, Mapping[str, Any]], required: list[str]
) -> list[Problem]:
    problems = [_invalid(path + _pointer(key), f"Unknown key '{key}'.") for key in value if key not in properties]
    missing = [key for key in required if key not in value]
    if missing:
        problems.append(_invalid(path, f"Missing required key(s): {missing}."))
    for key, schema in properties.items():
        if key in value and not _conforms(value[key], schema):
            problems.append(_invalid(path + _pointer(key), f"'{key}' does not match its schema: {dict(schema)}."))
    return problems


def _feature_problems(features: Any) -> list[Problem]:
    if not isinstance(features, list):
        return [_invalid("/features", "'features' must be an array.")]
    problems: list[Problem] = []
    if not 1 <= len(features) <= MAX_FEATURES:
        problems.append(_invalid("/features", f"'features' must hold 1 to {MAX_FEATURES} items."))
    for index, item in enumerate(features):
        path = _pointer("features", index)
        if isinstance(item, dict):
            problems.extend(_object_problems(item, path, _FEATURE_CONFIG["properties"], _FEATURE_CONFIG["required"]))
            if item.get("keep_index") is True and "index" not in item:
                problems.append(_invalid(path, "'keep_index' requires 'index' so the returned key columns are known."))
        elif not _conforms(item, _NAME_SCHEMA):
            problems.append(
                _invalid(path, f"A feature is a name of 1 to {MAX_NAME_LENGTH} characters or a feature config object.")
            )
    return problems


def _filter_problems(filters: Any) -> list[Problem]:
    if not isinstance(filters, list):
        return [_invalid("/filters", "'filters' must be an array.")]
    problems: list[Problem] = []
    if len(filters) > MAX_FILTERS:
        problems.append(_invalid("/filters", f"'filters' must hold at most {MAX_FILTERS} items."))
    for index, item in enumerate(filters):
        path = _pointer("filters", index)
        if not isinstance(item, dict):
            problems.append(_invalid(path, "A filter must be an object."))
            continue
        problems.extend(_object_problems(item, path, _FILTER_PROPERTIES, _FILTER_REQUIRED))
        op = item.get("op")
        keys = _OP_REQUIRES_ANY.get(op, ()) if isinstance(op, str) else ()
        if keys and not any(key in item for key in keys):
            problems.append(_invalid(path, f"Filter op '{op}' requires one of {list(keys)}."))
    return problems


def _source_problems(sources: Any, settings: Settings) -> list[Problem]:
    if not _is_type(sources, "array"):
        return [_invalid("/restrict_sources", "'restrict_sources' must be an array of source names.")]
    problems: list[Problem] = []
    if any(sources[i] == sources[j] for i in range(len(sources)) for j in range(i)):
        problems.append(_invalid("/restrict_sources", "'restrict_sources' must not repeat a source."))
    known = sorted(settings.sources)
    for index, source in enumerate(sources):
        path = _pointer("restrict_sources", index)
        if not _conforms(source, _RESTRICT_SOURCES_SCHEMA["items"]):
            problems.append(_invalid(path, "A source name must be a string."))
        elif source not in settings.sources:
            close = difflib.get_close_matches(source, known, n=1)
            problems.append(
                Problem(
                    path=path,
                    problem="unknown_source",
                    message=f"Unknown source '{source}'; configured sources: {known}.",
                    suggestion=close[0] if close else None,
                )
            )
    return problems


def _data_problems(data: Any) -> list[Problem]:
    if not isinstance(data, dict):
        return [_invalid("/data", "'data' must be an object of named sets.")]
    problems: list[Problem] = []
    if not all(_conforms(name, _NAME_SCHEMA) for name in data):
        problems.append(_invalid("/data", f"A set name is 1 to {MAX_NAME_LENGTH} characters."))
    for set_name, columns in data.items():
        path = _pointer("data", set_name)
        if not isinstance(columns, dict) or not columns:
            problems.append(_invalid(path, "A data set must be an object with at least one column."))
            continue
        if not all(_conforms(name, _NAME_SCHEMA) for name in columns):
            problems.append(_invalid(path, f"A column name is 1 to {MAX_NAME_LENGTH} characters."))
        for column, values in columns.items():
            column_path = path + _pointer(column)
            if not _conforms(values, _DATA_COLUMN_SCHEMA):
                if not isinstance(values, list):
                    problems.append(_invalid(column_path, "A column must be an array."))
                    continue
                for index, value in enumerate(values):
                    if not _conforms(value, _DATA_VALUE_SCHEMA):
                        problems.append(
                            _invalid(
                                column_path + _pointer(index), "A value must be a string, number, boolean or null."
                            )
                        )
    return problems


def validate_request(request: Any, settings: Settings) -> list[Problem]:
    """Return every problem of a feature request; an empty list means it is valid."""
    if not isinstance(request, dict):
        return [_invalid("", "The request must be a JSON object.")]
    allowed = ("features", "filters", "restrict_sources", "data")
    problems = [
        _invalid(_pointer(key), f"Unknown key '{key}'; allowed: {list(allowed)}.")
        for key in request
        if key not in allowed
    ]
    if "features" not in request:
        problems.append(_invalid("/features", "'features' is required."))
    else:
        problems.extend(_feature_problems(request["features"]))
    if "filters" in request:
        problems.extend(_filter_problems(request["filters"]))
    if "restrict_sources" in request:
        problems.extend(_source_problems(request["restrict_sources"], settings))
    if "data" in request:
        problems.extend(_data_problems(request["data"]))
    return problems


def data_access_option_keys() -> frozenset[str]:
    """Option keys that configure data access: reserved and query keys and source pointers."""
    pointers = {key for group in concrete_source_groups().values() for key in group.pointer_keys()}
    return frozenset({RESERVED_READER_OPTION_KEY, QUERY_OPTION, *pointers})


def option_data_access_violations(options: Mapping[str, Any], path: str = "") -> list[Problem]:
    """Report every data-access key in a group ``options`` mapping, including nested in_features configs."""
    return _scan_data_access({"options": dict(options)}, path, data_access_option_keys())


def _declared_in_features(config: Mapping[str, Any], path: str) -> Iterator[tuple[str, str | dict[str, Any]]]:
    """Yield ``(pointer, source name or nested config)`` for every ``in_features`` declaration at any depth."""
    for key, value in config.items():
        key_path = path + _pointer(key)
        if str(key) != str(DefaultOptionKeys.in_features):
            if isinstance(value, dict):
                yield from _declared_in_features(value, key_path)
        elif isinstance(value, str):
            yield key_path, value
        elif isinstance(value, list):
            for j, element in enumerate(value):
                if isinstance(element, str):
                    yield key_path + _pointer(j), element
        elif isinstance(value, dict):
            yield key_path, value
            yield from _declared_in_features(value, key_path)


def in_feature_sources(request: Any) -> list[tuple[str, str]]:
    """Return ``(pointer, name)`` for every source name the request's feature configs declare in ``in_features``."""
    features = request.get("features") if isinstance(request, dict) else None
    if not isinstance(features, list):
        return []
    sources: list[tuple[str, str]] = []
    for index, item in enumerate(features):
        if not isinstance(item, dict):
            continue
        for pointer, declared in _declared_in_features(item, _pointer("features", index)):
            if isinstance(declared, str):
                sources.append((pointer, declared))
            elif isinstance(declared.get("name"), str):
                sources.append((pointer + _pointer("name"), declared["name"]))
    return sources


def _scan_feature_config(
    item: Mapping[str, Any], path: str, flagged: Callable[[str], bool], problem: str, message: str
) -> list[Problem]:
    problems: list[Problem] = []
    configs = [(path, item)] + [(p, d) for p, d in _declared_in_features(item, path) if isinstance(d, dict)]
    for config_path, config in configs:
        for container in _OPTION_CONTAINERS:
            options = config.get(container)
            if not isinstance(options, dict):
                continue
            for key in options:
                if flagged(key):
                    problems.append(
                        Problem(
                            path=config_path + _pointer(container, key),
                            problem=problem,
                            message=message.format(key=key),
                        )
                    )
    return problems


_DATA_ACCESS_MESSAGE = "Option key '{key}' configures data access, which a request may not set."
_NOT_ALLOWED_MESSAGE = "Option key '{key}' is not in allow_options, so a request may not set it."
_NOT_CALLER_SETTABLE_MESSAGE = "Option key '{key}' is marked by its plugin as not settable by a caller."


def _scan_data_access(item: Mapping[str, Any], path: str, keys: frozenset[str]) -> list[Problem]:
    return _scan_feature_config(item, path, keys.__contains__, "data_access_option", _DATA_ACCESS_MESSAGE)


def caller_refused_option_keys() -> frozenset[str]:
    """Option keys a plugin marks as not settable by an untrusted caller."""
    keys: set[str] = set()
    for group in feature_group_subclasses():
        mapping = getattr(group, "PROPERTY_MAPPING", None)
        if not isinstance(mapping, Mapping):
            continue
        keys.update(
            str(key) for key, spec in mapping.items() if isinstance(spec, PropertySpec) and not spec.caller_settable
        )
    return frozenset(keys)


def _scan_refused(item: Mapping[str, Any], path: str, keys: frozenset[str]) -> list[Problem]:
    return _scan_feature_config(
        item, path, lambda key: str(key) in keys, "option_not_caller_settable", _NOT_CALLER_SETTABLE_MESSAGE
    )


def refused_option_violations(request: Any) -> list[Problem]:
    """Report every option key a plugin marks as not caller-settable by path, never by value."""
    features = request.get("features") if isinstance(request, dict) else None
    if not isinstance(features, list):
        return []
    keys = caller_refused_option_keys()
    problems: list[Problem] = []
    for index, item in enumerate(features):
        if isinstance(item, dict):
            problems.extend(_scan_refused(item, _pointer("features", index), keys))
    return problems


def refused_group_option_violations(options: Mapping[str, Any]) -> list[Problem]:
    """Report every not-caller-settable key of a group ``options`` mapping as ``/options/<key>``."""
    return _scan_refused({"options": dict(options)}, "", caller_refused_option_keys())


def _unlisted_skipped() -> frozenset[str]:
    return data_access_option_keys() | caller_refused_option_keys() | {str(DefaultOptionKeys.in_features)}


def _unlisted_scan(
    item: Mapping[str, Any], path: str, allowed: Collection[str], skipped: frozenset[str]
) -> list[Problem]:
    return _scan_feature_config(
        item,
        path,
        lambda key: str(key) not in skipped and key not in allowed,
        "option_not_allowed",
        _NOT_ALLOWED_MESSAGE,
    )


def unlisted_option_violations(request: Any, allowed: Collection[str]) -> list[Problem]:
    """Report every option key outside ``allowed`` by path, never by value; data-access keys are left out."""
    features = request.get("features") if isinstance(request, dict) else None
    if not isinstance(features, list):
        return []
    skipped = _unlisted_skipped()
    problems: list[Problem] = []
    for index, item in enumerate(features):
        if isinstance(item, dict):
            problems.extend(_unlisted_scan(item, _pointer("features", index), allowed, skipped))
    return problems


def unlisted_group_option_violations(options: Mapping[str, Any], allowed: Collection[str]) -> list[Problem]:
    """Report every key of a group ``options`` mapping outside ``allowed`` as ``/options/<key>``."""
    return _unlisted_scan({"options": dict(options)}, "", allowed, _unlisted_skipped())


def data_access_violations(request: Any) -> list[Problem]:
    """Report every data-access option key in the request's feature configs by path, never by value."""
    features = request.get("features") if isinstance(request, dict) else None
    if not isinstance(features, list):
        return []
    keys = data_access_option_keys()
    problems: list[Problem] = []
    for index, item in enumerate(features):
        if isinstance(item, dict):
            problems.extend(_scan_data_access(item, _pointer("features", index), keys))
    return problems


def _load_feature_config(item: dict[str, Any], path: str) -> list[Feature | str] | Problem:
    # The core loader is the only semantic check and signals a bad config by raising.
    try:
        return load_features_from_config(json.dumps([item]))
    except (ValueError, TypeError) as error:
        return Problem(path=path, problem="invalid_feature_config", message=str(error))


def _build_filter(filters: list[dict[str, Any]]) -> tuple[GlobalFilter | None, tuple[UUID, ...]]:
    if not filters:
        return None, ()
    global_filter = GlobalFilter()
    uuids: list[UUID] = []
    for item in filters:
        op = item["op"]
        parameter = {key: item[key] for key in _OP_PARAMETERS[op] if key in item}
        if "values" in parameter:
            parameter["values"] = tuple(parameter["values"])
        global_filter.add_filter(item["feature"], _OP_FILTER_TYPES[op], parameter)
        # An equal filter merges into the stored one, so look the stored uuid up by equality.
        declared = SingleFilter(item["feature"], _OP_FILTER_TYPES[op], parameter)
        uuids.append(next(stored.uuid for stored in global_filter.filters if stored == declared))
    return global_filter, tuple(uuids)


def _inline_data_problems(data: Mapping[str, Mapping[str, list[Any]]]) -> list[Problem]:
    problems: list[Problem] = []
    owner: dict[str, str] = {}
    for set_name, columns in data.items():
        path = _pointer("data", set_name)
        if len({len(values) for values in columns.values()}) > 1:
            problems.append(
                Problem(
                    path=path, problem="invalid_inline_data", message="The columns of a set must have equal lengths."
                )
            )
        for column in columns:
            if column in owner:
                problems.append(
                    Problem(
                        path=path + _pointer(column),
                        problem="invalid_inline_data",
                        message=f"Column '{column}' is already in set '{owner[column]}'; a column belongs to one set.",
                    )
                )
            else:
                owner[column] = set_name
    return problems


def _with_defaults(item: str | dict[str, Any], settings: Settings | None) -> str | dict[str, Any]:
    """Merge the operator default options under a feature; the feature's own values win."""
    if settings is None or not (settings.options or settings.context_options):
        return item
    config: dict[str, Any] = {"name": item} if isinstance(item, str) else dict(item)
    if "options" in config and ("group_options" in config or "context_options" in config):
        return config
    own = {key for container in _OPTION_CONTAINERS for key in config.get(container) or {}}
    group = {
        **{key: value for key, value in settings.options.items() if key not in own},
        **config.pop("options", {}),
        **config.get("group_options", {}),
    }
    context = {
        **{key: value for key, value in settings.context_options.items() if key not in own},
        **config.get("context_options", {}),
    }
    if group:
        config["group_options"] = group
    if context:
        config["context_options"] = context
    return config


def build_request(request: Mapping[str, Any], *, settings: Settings | None = None) -> ParsedRequest | list[Problem]:
    """Build a structurally valid request; semantic feature-config errors come back as problems."""
    features: list[Feature | str] = []
    problems: list[Problem] = []
    for index, original in enumerate(request["features"]):
        item = _with_defaults(original, settings)
        if isinstance(item, str):
            features.append(item)
            continue
        loaded = _load_feature_config(item, _pointer("features", index))
        if isinstance(loaded, Problem):
            problems.append(loaded)
        else:
            features.extend(loaded)
    inline_data = request.get("data", {})
    problems.extend(_inline_data_problems(inline_data))
    if problems:
        return problems
    global_filter, filter_uuids = _build_filter(request.get("filters", []))
    return ParsedRequest(
        features=features,
        global_filter=global_filter,
        restrict_sources=tuple(request.get("restrict_sources", ())),
        normalized=json.loads(json.dumps(request, sort_keys=True)),
        filter_uuids=filter_uuids,
        inline_data=json.loads(json.dumps(inline_data)),
    )
