"""Settings for the mloda service layer: TOML file, then MLODA_MCP_* env vars, then explicit overrides."""

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODES = ("plan", "execute")
_MCP_KEYS = frozenset(
    {
        "mode",
        "compute_frameworks",
        "max_rows",
        "deny_feature_groups",
        "alternatives",
        "preview_allows",
        "max_preview_rows",
        "plan_ttl",
        "sources",
        "classification",
        "inline_data",
        "result_paging",
        "max_inline_values",
        "max_inline_chars",
        "default_classification",
        "allow_options",
        "options",
        "context_options",
    }
)
_SOURCE_KEYS = frozenset({"files", "folders"})
_ENV_PREFIX = "MLODA_MCP_"
_INLINE_LIMIT_KEYS = ("max_inline_values", "max_inline_chars")
_ENV_INT_KEYS = frozenset({"max_rows", *_INLINE_LIMIT_KEYS})
_ENV_LIST_KEYS = frozenset({"compute_frameworks"})
_ENV_STR_KEYS = frozenset({"mode"})
_TTL_PATTERN = re.compile(r"(\d+)([hms])")
_TTL_UNITS = {"h": 3600, "m": 60, "s": 1}


class ConfigError(ValueError):
    """Invalid or unreadable service configuration."""


@dataclass(frozen=True)
class SourceConfig:
    """A named data source: absolute file and folder paths."""

    alias: str
    files: tuple[str, ...] = ()
    folders: tuple[str, ...] = ()


@dataclass(frozen=True)
class Settings:
    """Resolved service settings."""

    mode: str = "plan"
    compute_frameworks: tuple[str, ...] = ("PythonDictFramework",)
    max_rows: int | None = None
    deny_feature_groups: tuple[str, ...] = ()
    alternatives: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    preview_allows: tuple[str, ...] = ("public", "internal")
    max_preview_rows: int = 20
    plan_ttl_seconds: int = 86400
    sources: Mapping[str, SourceConfig] = field(default_factory=dict)
    classification: Mapping[str, str] = field(default_factory=dict)
    inline_data: bool = False
    result_paging: bool = False
    max_inline_values: int | None = None
    max_inline_chars: int | None = None
    default_classification: str | None = None
    allow_options: tuple[str, ...] = ()
    trusted_caller: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)
    context_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ConfigError(f"'mode' must be one of {list(MODES)}, got {self.mode!r}")
        if self.max_preview_rows <= 0:
            raise ConfigError("'max_preview_rows' must be a positive integer")

    def config_hash(self) -> str:
        """SHA-256 hex digest of the settings, independent of mapping insertion order."""
        canonical = json.dumps(asdict(self), sort_keys=True, default=dict)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_settings(
    config_path: str | Path | None, env: Mapping[str, str], overrides: Mapping[str, Any] | None = None
) -> Settings:
    """Merge file, env and overrides (later wins) into validated Settings."""
    raw: dict[str, Any] = {}
    base_dir = Path.cwd()
    if config_path is not None:
        path = Path(config_path)
        raw.update(_read_mcp_table(path))
        base_dir = path.resolve().parent
    raw.update(_from_env(env))
    raw.update(overrides or {})
    _reject_unknown(raw, _MCP_KEYS, "[mcp]")
    return _build(raw, base_dir)


def _read_mcp_table(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        # TOMLDecodeError subclasses ValueError.
        try:
            data = tomllib.load(handle)
        except ValueError as error:
            raise ConfigError(f"{path.name} is not valid TOML: {error}") from None
    table = data.get("mcp", {})
    if not isinstance(table, dict):
        raise ConfigError("'mcp' must be a table")
    return table


def _from_env(env: Mapping[str, str]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for key in _ENV_STR_KEYS | _ENV_INT_KEYS | _ENV_LIST_KEYS:
        value = env.get(_ENV_PREFIX + key.upper())
        if value is None:
            continue
        if key in _ENV_INT_KEYS:
            if not re.fullmatch(r"-?\d+", value.strip()):
                raise ConfigError(f"{_ENV_PREFIX}{key.upper()} must be an integer")
            raw[key] = int(value)
        elif key in _ENV_LIST_KEYS:
            raw[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            raw[key] = value
    return raw


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {unknown}; allowed: {sorted(allowed)}")


def _str_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{key}' must be a list of strings")
    return tuple(value)


def _str_mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise ConfigError(f"'{key}' must be a table")
    return dict(value)


def _json_safe(value: Any) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_safe(item) for item in value)
    return isinstance(value, (str, int, float, bool))


def _default_options(value: Any, key: str) -> dict[str, Any]:
    table = _str_mapping(value, key)
    for reserved in ("in_features", "feature_group"):
        if reserved in table:
            raise ConfigError(f"'{key}' must not set {reserved}")
    if not _json_safe(table):
        raise ConfigError(f"'{key}' values must be strings, numbers, booleans, lists or tables")
    return table


def _int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer")
    return value


def _parse_ttl(value: Any) -> int:
    match = _TTL_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ConfigError("'plan_ttl' must be a number followed by h, m or s, e.g. '24h'")
    return int(match.group(1)) * _TTL_UNITS[match.group(2)]


def _resolve_paths(value: Any, key: str, base_dir: Path) -> tuple[str, ...]:
    return tuple(str((base_dir / item).resolve()) for item in _str_tuple(value, key))


def _build_source(alias: str, value: Any, base_dir: Path) -> SourceConfig:
    table = _str_mapping(value, f"sources.{alias}")
    _reject_unknown(table, _SOURCE_KEYS, f"[mcp.sources.{alias}]")
    return SourceConfig(
        alias=alias,
        files=_resolve_paths(table.get("files", []), "files", base_dir),
        folders=_resolve_paths(table.get("folders", []), "folders", base_dir),
    )


def _build(raw: Mapping[str, Any], base_dir: Path) -> Settings:
    kwargs: dict[str, Any] = {}
    if "mode" in raw:
        kwargs["mode"] = raw["mode"]
    for key in ("compute_frameworks", "deny_feature_groups", "preview_allows", "allow_options"):
        if key in raw:
            kwargs[key] = _str_tuple(raw[key], key)
    if "ApiInputDataFeature" in kwargs.get("deny_feature_groups", ()):
        raise ConfigError("'deny_feature_groups' names the removed ApiInputDataFeature; use ApiDataFG instead")
    if raw.get("max_rows") is not None:
        kwargs["max_rows"] = _int(raw["max_rows"], "max_rows")
    for key in _INLINE_LIMIT_KEYS:
        if raw.get(key) is not None:
            kwargs[key] = _int(raw[key], key)
            if kwargs[key] < 1:
                raise ConfigError(f"'{key}' must be an integer of at least 1")
    if "max_preview_rows" in raw:
        kwargs["max_preview_rows"] = _int(raw["max_preview_rows"], "max_preview_rows")
    if "plan_ttl" in raw:
        kwargs["plan_ttl_seconds"] = _parse_ttl(raw["plan_ttl"])
    if "alternatives" in raw:
        alternatives = _str_mapping(raw["alternatives"], "alternatives")
        kwargs["alternatives"] = {k: _str_tuple(v, f"alternatives.{k}") for k, v in alternatives.items()}
    if "classification" in raw:
        classification = _str_mapping(raw["classification"], "classification")
        if not all(isinstance(v, str) for v in classification.values()):
            raise ConfigError("'classification' values must be strings")
        kwargs["classification"] = classification
    if "inline_data" in raw:
        if not isinstance(raw["inline_data"], bool):
            raise ConfigError("'inline_data' must be true or false")
        kwargs["inline_data"] = raw["inline_data"]
    if "result_paging" in raw:
        if not isinstance(raw["result_paging"], bool):
            raise ConfigError("'result_paging' must be true or false")
        kwargs["result_paging"] = raw["result_paging"]
    if raw.get("default_classification") is not None:
        if not isinstance(raw["default_classification"], str) or not raw["default_classification"]:
            raise ConfigError("'default_classification' must be a non-empty string")
        kwargs["default_classification"] = raw["default_classification"]
    for key in ("options", "context_options"):
        if key in raw:
            kwargs[key] = _default_options(raw[key], key)
    if "sources" in raw:
        sources = _str_mapping(raw["sources"], "sources")
        kwargs["sources"] = {alias: _build_source(alias, v, base_dir) for alias, v in sources.items()}
    return Settings(**kwargs)
