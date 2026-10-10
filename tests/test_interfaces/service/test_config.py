"""Settings loading for the MCP service layer: TOML file, env vars, overrides, strictness, config hash."""

from pathlib import Path

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


def test_config_hash_ignores_mapping_insertion_order() -> None:
    one = Settings(classification={"a": "personal", "b": "public"})
    other = Settings(classification={"b": "public", "a": "personal"})
    assert one.config_hash() == other.config_hash()
