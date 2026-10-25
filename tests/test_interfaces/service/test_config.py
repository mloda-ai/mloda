"""Settings loading for the MCP service layer: TOML file, env vars, overrides, strictness, config hash."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mloda.interfaces.service.config import ConfigError, Settings, SourceConfig, load_settings


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "mloda-mcp.toml"
    path.write_text(text, encoding="utf-8")
    return path


FULL_TOML = """
[mcp]
mode = "execute"
compute_frameworks = ["PandasDataFrame", "PythonDictFramework"]
max_rows = 1000
deny_feature_groups = ["SecretFeatureGroup"]
preview_allows = ["public"]
max_preview_rows = 5
plan_ttl = "30m"

[mcp.alternatives]
ssn = ["ssn_hash", "ssn_last4"]

[mcp.classification]
email = "personal"

[mcp.sources.sales]
files = ["data/sales.csv"]
folders = ["{abs_folder}"]
"""


def test_config_error_is_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.mode == "plan"
    assert settings.compute_frameworks == ("PythonDictFramework",)
    assert settings.max_rows is None
    assert settings.deny_feature_groups == ()
    assert dict(settings.alternatives) == {}
    assert settings.preview_allows == ("public", "internal")
    assert settings.max_preview_rows == 20
    assert settings.plan_ttl_seconds == 86400
    assert dict(settings.sources) == {}
    assert dict(settings.classification) == {}
    assert settings.inline_data is False


def test_source_config_defaults() -> None:
    source = SourceConfig(alias="sales")
    assert source.files == ()
    assert source.folders == ()


def test_load_without_path_env_or_overrides_gives_defaults() -> None:
    assert load_settings(None, env={}, overrides=None) == Settings()


def test_load_full_toml_file(tmp_path: Path) -> None:
    abs_folder = tmp_path / "abs_folder"
    path = _write(tmp_path, FULL_TOML.format(abs_folder=abs_folder.as_posix()))

    settings = load_settings(path, env={})

    assert settings.mode == "execute"
    assert settings.compute_frameworks == ("PandasDataFrame", "PythonDictFramework")
    assert settings.max_rows == 1000
    assert settings.deny_feature_groups == ("SecretFeatureGroup",)
    assert settings.preview_allows == ("public",)
    assert settings.max_preview_rows == 5
    assert settings.plan_ttl_seconds == 1800
    assert dict(settings.alternatives) == {"ssn": ("ssn_hash", "ssn_last4")}
    assert dict(settings.classification) == {"email": "personal"}
    assert set(settings.sources) == {"sales"}
    source = settings.sources["sales"]
    assert source.alias == "sales"
    assert len(source.files) == 1
    assert Path(source.files[0]).is_absolute()
    assert Path(source.files[0]).resolve() == (tmp_path / "data" / "sales.csv").resolve()
    assert len(source.folders) == 1
    assert Path(source.folders[0]).resolve() == abs_folder.resolve()


def test_load_accepts_str_path(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\nmode = "execute"\n')
    assert load_settings(str(path), env={}).mode == "execute"


def test_relative_source_path_resolves_against_config_dir_not_cwd(tmp_path: Path) -> None:
    config_dir = tmp_path / "nested" / "cfg"
    config_dir.mkdir(parents=True)
    path = config_dir / "mloda-mcp.toml"
    path.write_text('[mcp.sources.s]\nfolders = ["../data"]\n', encoding="utf-8")

    folder = load_settings(path, env={}).sources["s"].folders[0]

    assert Path(folder).is_absolute()
    assert Path(folder).resolve() == (tmp_path / "nested" / "data").resolve()


@pytest.mark.parametrize(("ttl", "seconds"), [("24h", 86400), ("30m", 1800), ("90s", 90)])
def test_plan_ttl_units(tmp_path: Path, ttl: str, seconds: int) -> None:
    path = _write(tmp_path, f'[mcp]\nplan_ttl = "{ttl}"\n')
    assert load_settings(path, env={}).plan_ttl_seconds == seconds


@pytest.mark.parametrize("ttl", ["abc", "5x", "h", "30 minutes"])
def test_bad_plan_ttl_is_config_error(tmp_path: Path, ttl: str) -> None:
    path = _write(tmp_path, f'[mcp]\nplan_ttl = "{ttl}"\n')
    with pytest.raises(ConfigError):
        load_settings(path, env={})


def test_unknown_key_in_mcp_table_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\nmood = "plan"\n')
    with pytest.raises(ConfigError, match="mood"):
        load_settings(path, env={})


def test_unknown_key_in_source_table_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp.sources.sales]\nfilez = ["a.csv"]\n')
    with pytest.raises(ConfigError, match="filez"):
        load_settings(path, env={})


def test_invalid_mode_in_file_is_config_error(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\nmode = "write"\n')
    with pytest.raises(ConfigError):
        load_settings(path, env={})


@pytest.mark.parametrize("rows", [0, -1])
def test_non_positive_max_preview_rows_is_config_error(tmp_path: Path, rows: int) -> None:
    path = _write(tmp_path, f"[mcp]\nmax_preview_rows = {rows}\n")
    with pytest.raises(ConfigError):
        load_settings(path, env={})


def test_malformed_toml_is_config_error_naming_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp\nmode = "plan"\n')
    with pytest.raises(ConfigError) as excinfo:
        load_settings(path, env={})
    assert path.name in str(excinfo.value)


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "does-not-exist.toml", env={})


def test_env_vars_are_applied() -> None:
    env = {
        "MLODA_MCP_MODE": "execute",
        "MLODA_MCP_MAX_ROWS": "100",
        "MLODA_MCP_COMPUTE_FRAMEWORKS": "PandasDataFrame,PolarsDataFrame",
    }
    settings = load_settings(None, env=env)
    assert settings.mode == "execute"
    assert settings.max_rows == 100
    assert settings.compute_frameworks == ("PandasDataFrame", "PolarsDataFrame")


def test_unrelated_env_vars_are_ignored() -> None:
    assert load_settings(None, env={"PATH": "/usr/bin", "MLODA_OTHER": "x"}) == Settings()


def test_env_wins_over_file(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\nmode = "plan"\nmax_rows = 10\n')
    settings = load_settings(path, env={"MLODA_MCP_MODE": "execute", "MLODA_MCP_MAX_ROWS": "20"})
    assert settings.mode == "execute"
    assert settings.max_rows == 20


def test_overrides_win_over_env_and_file(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\nmode = "plan"\nmax_rows = 10\n')
    settings = load_settings(
        path,
        env={"MLODA_MCP_MODE": "execute", "MLODA_MCP_MAX_ROWS": "20"},
        overrides={"mode": "plan", "max_rows": 30},
    )
    assert settings.mode == "plan"
    assert settings.max_rows == 30


def test_invalid_env_mode_is_config_error() -> None:
    with pytest.raises(ConfigError):
        load_settings(None, env={"MLODA_MCP_MODE": "write"})


def test_non_integer_env_max_rows_is_config_error() -> None:
    with pytest.raises(ConfigError):
        load_settings(None, env={"MLODA_MCP_MAX_ROWS": "many"})


def test_config_hash_is_hex_sha256() -> None:
    digest = Settings().config_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_hash_equal_for_identical_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, FULL_TOML.format(abs_folder=(tmp_path / "abs_folder").as_posix()))
    assert load_settings(path, env={}).config_hash() == load_settings(path, env={}).config_hash()


def test_config_hash_changes_with_mode() -> None:
    assert Settings().config_hash() != Settings(mode="execute").config_hash()


def test_config_hash_changes_with_source() -> None:
    one = Settings(sources={"sales": SourceConfig(alias="sales", files=("/data/a.csv",))})
    other = Settings(sources={"sales": SourceConfig(alias="sales", files=("/data/b.csv",))})
    assert one.config_hash() != other.config_hash()
    assert one.config_hash() != Settings().config_hash()


def test_config_hash_changes_with_classification() -> None:
    assert Settings().config_hash() != Settings(classification={"email": "personal"}).config_hash()


def test_config_hash_changes_with_inline_data() -> None:
    assert Settings().config_hash() != Settings(inline_data=True).config_hash()


def test_inline_data_toml_true_loads(tmp_path: Path) -> None:
    assert load_settings(_write(tmp_path, "[mcp]\ninline_data = true\n"), env={}).inline_data is True


@pytest.mark.parametrize("value", ['"yes"', "1", "[true]"])
def test_non_bool_inline_data_is_config_error(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="'inline_data' must be true or false"):
        load_settings(_write(tmp_path, f"[mcp]\ninline_data = {value}\n"), env={})


def test_trusted_caller_defaults_to_false() -> None:
    assert Settings().trusted_caller is False


def test_config_hash_changes_with_trusted_caller() -> None:
    assert Settings().config_hash() != Settings(trusted_caller=True).config_hash()


def test_trusted_caller_is_not_a_toml_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="trusted_caller"):
        load_settings(_write(tmp_path, "[mcp]\ntrusted_caller = true\n"), env={})


def test_trusted_caller_is_not_an_override() -> None:
    with pytest.raises(ConfigError, match="trusted_caller"):
        load_settings(None, env={}, overrides={"trusted_caller": True})


def test_default_classification_defaults_to_none() -> None:
    assert Settings().default_classification is None


def test_default_classification_toml_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\ndefault_classification = "internal"\n')
    assert load_settings(path, env={}).default_classification == "internal"


@pytest.mark.parametrize("value", ["1", "true", '["internal"]'])
def test_non_string_default_classification_is_config_error(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="default_classification"):
        load_settings(_write(tmp_path, f"[mcp]\ndefault_classification = {value}\n"), env={})


def test_config_hash_changes_with_default_classification() -> None:
    assert Settings().config_hash() != Settings(default_classification="internal").config_hash()


def test_allow_options_defaults_to_empty_tuple() -> None:
    assert Settings().allow_options == ()


def test_allow_options_toml_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\nallow_options = ["k", "other"]\n')
    assert load_settings(path, env={}).allow_options == ("k", "other")


@pytest.mark.parametrize("value", ['"k"', "1", "[1]", '["k", 2]', "{k = 1}"])
def test_bad_allow_options_is_config_error(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="allow_options"):
        load_settings(_write(tmp_path, f"[mcp]\nallow_options = {value}\n"), env={})


def test_config_hash_changes_with_allow_options() -> None:
    assert Settings().config_hash() != Settings(allow_options=("k",)).config_hash()


def test_config_hash_ignores_mapping_insertion_order() -> None:
    one = Settings(classification={"a": "personal", "b": "public"})
    other = Settings(classification={"b": "public", "a": "personal"})
    assert one.config_hash() == other.config_hash()


def test_removed_api_input_data_feature_in_deny_list_is_rejected_with_a_hint(tmp_path: Path) -> None:
    path = _write(tmp_path, '[mcp]\ndeny_feature_groups = ["ApiInputDataFeature"]\n')
    with pytest.raises(ConfigError, match="ApiDataFG"):
        load_settings(path, env={})


def test_default_options_default_to_empty() -> None:
    assert dict(Settings().options) == {}
    assert dict(Settings().context_options) == {}


def test_default_option_tables_load(tmp_path: Path) -> None:
    text = '[mcp.options]\nmodel = "m"\nretries = 3\nratio = 0.5\non = true\nbook = ["a", {b = [1, 2]}]\n'
    text += '[mcp.context_options]\ntimeout = 30\nnested = {x = {y = "z"}}\n'

    settings = load_settings(_write(tmp_path, text), env={})

    assert dict(settings.options) == {
        "model": "m",
        "retries": 3,
        "ratio": 0.5,
        "on": True,
        "book": ["a", {"b": [1, 2]}],
    }
    assert dict(settings.context_options) == {"timeout": 30, "nested": {"x": {"y": "z"}}}


@pytest.mark.parametrize("table", ["options", "context_options"])
@pytest.mark.parametrize(
    "body",
    ["when = 2020-01-01T00:00:00Z", "day = 2020-01-01", "deep = [{at = 12:30:00}]", 'in_features = "x"'],
)
def test_bad_default_option_values_are_config_errors(tmp_path: Path, table: str, body: str) -> None:
    with pytest.raises(ConfigError, match=table):
        load_settings(_write(tmp_path, f"[mcp.{table}]\n{body}\n"), env={})


@pytest.mark.parametrize("table", ["options", "context_options"])
@pytest.mark.parametrize("value", ['"x"', "1", '["a"]'])
def test_non_table_default_options_is_config_error(tmp_path: Path, table: str, value: str) -> None:
    with pytest.raises(ConfigError, match=table):
        load_settings(_write(tmp_path, f"[mcp]\n{table} = {value}\n"), env={})


def test_config_hash_changes_with_default_options() -> None:
    base = Settings().config_hash()
    assert base != Settings(options={"k": 1}).config_hash()
    assert base != Settings(context_options={"k": 1}).config_hash()
    assert Settings(options={"k": 1}).config_hash() != Settings(context_options={"k": 1}).config_hash()


@pytest.mark.parametrize("table", ["options", "context_options"])
def test_in_features_default_option_is_config_error(tmp_path: Path, table: str) -> None:
    with pytest.raises(ConfigError, match="in_features"):
        load_settings(_write(tmp_path, f'[mcp.{table}]\nin_features = "x"\n'), env={})


@pytest.mark.parametrize("key", ["max_inline_values", "max_inline_chars"])
class TestInlineLimits:
    def test_default_is_none(self, key: str) -> None:
        assert getattr(Settings(), key) is None

    def test_toml_loads(self, tmp_path: Path, key: str) -> None:
        assert getattr(load_settings(_write(tmp_path, f"[mcp]\n{key} = 5\n"), env={}), key) == 5

    @pytest.mark.parametrize("value", ["0", "-1", "true", '"5"', "1.5"])
    def test_bad_toml_value_is_config_error(self, tmp_path: Path, key: str, value: str) -> None:
        with pytest.raises(ConfigError, match=f"'{key}' must be"):
            load_settings(_write(tmp_path, f"[mcp]\n{key} = {value}\n"), env={})

    def test_env_loads_and_wins_over_file(self, tmp_path: Path, key: str) -> None:
        env = {f"MLODA_MCP_{key.upper()}": "7"}
        assert getattr(load_settings(None, env=env), key) == 7
        assert getattr(load_settings(_write(tmp_path, f"[mcp]\n{key} = 5\n"), env=env), key) == 7

    @pytest.mark.parametrize("value", ["0", "-3", "many", "1.5", ""])
    def test_bad_env_value_is_config_error(self, key: str, value: str) -> None:
        with pytest.raises(ConfigError):
            load_settings(None, env={f"MLODA_MCP_{key.upper()}": value})

    def test_config_hash_changes(self, key: str) -> None:
        changes: dict[str, Any] = {key: 5}
        assert Settings().config_hash() != replace(Settings(), **changes).config_hash()
