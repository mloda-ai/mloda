"""ReadFileFG through the stdlib ``.toyfmtfile`` toy: shared file contract, framework loads, sample hook."""

from __future__ import annotations

import gc
import inspect
import os
from collections.abc import Collection
from pathlib import Path
from typing import Any, cast

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.provider import FormatFeatureGroup, ReadFileFG
from mloda.steward import ReaderProbeScope, reader_probe_scope
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    resolve_or_raise,
)
from mloda.user import DataAccessCollection, Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.lazy_format_group import load_document_group
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    FileLoadsIntoFrameworkMixin,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.file_format_feature_group_test_mixin import FileFormatFeatureGroupTestMixin
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_file_format_group import (
    SAMPLE_SIZE,
    TOY_SUFFIX,
    ToyFileFormatFG,
    toy_match,
    write_toy_file,
)


class TestToyFileFormatFeatureGroup(FileFormatFeatureGroupTestMixin):
    feature_group_class = ToyFileFormatFG
    present_column = "toyfmt_filecol"
    missing_column = "toyfmt_filemissing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_toy_file(path, columns)


class TestToyLoadsIntoPyArrowTable(PyArrowTableAdapter, FileLoadsIntoFrameworkMixin):
    file_group = ToyFileFormatFG

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_toy_file(path, columns)


class TestToyLoadsIntoPythonDict(PythonDictAdapter, FileLoadsIntoFrameworkMixin):
    file_group = ToyFileFormatFG

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_toy_file(path, columns)


class _Features:
    def get_all_names(self) -> set[str]:
        return {"toyfmt_b", "toyfmt_a"}


class TestReadFileFGBase:
    def test_a_template_base_may_implement_suffixes_from_an_unset_suffix(self) -> None:
        template: Any = type("RodTemplateFG", (ReadFileFG,), {"suffixes": classmethod(lambda cls: (cls.SUFFIX,))})
        assert inspect.isabstract(template)
        child: Any = type(
            "RodTemplateChildFG",
            (template,),
            {"SUFFIX": ".x", "column_names": classmethod(lambda cls, file_name: {"c"})},
        )
        assert child.suffixes() == (".x",)

    def test_is_an_abstract_format_group_exported_from_provider(self) -> None:
        assert provider.ReadFileFG is ReadFileFG
        assert "ReadFileFG" in provider.__all__
        assert issubclass(ReadFileFG, FormatFeatureGroup)
        assert inspect.isabstract(ReadFileFG)
        assert {"suffixes", "column_names"} <= set(ReadFileFG.__abstractmethods__)
        assert not {"find_sources", "load_neutral"} & set(ReadFileFG.__abstractmethods__)
        assert not inspect.isabstract(ToyFileFormatFG)

    def test_declares_checked_searched_file_and_folder_routes(self) -> None:
        assert ReadFileFG.CLAIM_ROUTES == (
            ClaimRoute("file", NamePolicy.CHECKED, True),
            ClaimRoute("folder", NamePolicy.CHECKED, True),
        )

    def test_property_mapping_declares_the_two_reader_options(self) -> None:
        handle = ReadFileFG.PROPERTY_MAPPING["data_access_handle"]
        assert handle.default is None
        assert handle.strict_validation is True
        assert handle.context is True
        assert handle.element_validator is not None
        assert handle.element_validator("a_handle")
        assert not handle.element_validator(5)
        assert handle.match_guard is not None
        assert handle.match_guard("a_handle")
        assert not handle.match_guard(["a_handle"])
        suffixes = ReadFileFG.PROPERTY_MAPPING["document_suffixes"]
        assert suffixes.default is None
        assert suffixes.context is True

    def test_file_format_is_the_first_suffix_without_dot_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert ToyFileFormatFG.file_format() == "toyfmtfile"
        monkeypatch.setattr(ToyFileFormatFG, "suffixes", classmethod(lambda cls: (".ToyFmtFile", ".other")))
        assert ToyFileFormatFG.file_format() == "toyfmtfile"

    def test_a_route_with_a_source_kind_other_than_file_or_folder_is_rejected(self) -> None:
        routes = (ClaimRoute("credentials", NamePolicy.CHECKED, True),)
        with pytest.raises(TypeError, match="BadKindFileFG"):
            type("BadKindFileFG", (ReadFileFG,), {"CLAIM_ROUTES": routes})

    @pytest.mark.parametrize("bad", ["toyfmtfile", (), (".ok", ""), (".ok", 1)])
    def test_suffixes_that_are_not_a_non_empty_tuple_of_non_empty_str_are_rejected(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="BadSuffixFileFG"):
            type(
                "BadSuffixFileFG",
                (ReadFileFG,),
                {"suffixes": classmethod(lambda cls: bad), "column_names": classmethod(lambda cls, path: [])},
            )

    def test_an_abstract_subclass_with_abstract_suffixes_defines_fine(self) -> None:
        child = type("AbstractSuffixFileFG", (ReadFileFG,), {"CLAIM_ROUTES": ReadFileFG.CLAIM_ROUTES})
        assert inspect.isabstract(child)

    def test_sample_listing_defaults_to_none_and_the_sample_size_is_an_int(self) -> None:
        assert ReadFileFG.sample_column_names("toyfmt_unused") is None
        assert isinstance(ReadFileFG.SAMPLE_SIZE_BYTES, int)

    def test_load_neutral_is_a_file_source_of_the_sorted_features(self, tmp_path: Path) -> None:
        path = tmp_path / f"neutral{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_a": [1], "toyfmt_b": [2]})
        neutral = ToyFileFormatFG.load_neutral(toy_match(path), _Features())
        assert neutral == FileSource(path=str(path), format="toyfmtfile", columns=("toyfmt_a", "toyfmt_b"))


class TestSampleListing:
    def _small(self, tmp_path: Path) -> Path:
        path = tmp_path / f"small{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_s_a": [1], "toyfmt_s_b": [2]})
        assert path.stat().st_size <= SAMPLE_SIZE
        return path

    def _wide(self, tmp_path: Path) -> Path:
        path = tmp_path / f"wide{TOY_SUFFIX}"
        write_toy_file(path, {f"toyfmt_wide_{i:02d}": [i] for i in range(20)})
        assert path.stat().st_size > SAMPLE_SIZE
        return path

    def _reads(self, path: Path) -> tuple[list[str], list[str]]:
        names = [p for p in ToyFileFormatFG.COLUMN_NAMES_CALLS if p == str(path)]
        samples = [p for p in ToyFileFormatFG.SAMPLE_CALLS if p == str(path)]
        return names, samples

    def test_a_sample_hit_needs_no_full_listing(self, tmp_path: Path) -> None:
        path = self._wide(tmp_path)
        assert ToyFileFormatFG.has_column(toy_match(path), "toyfmt_wide_00") is True
        names, samples = self._reads(path)
        assert samples and not names

    def test_a_sample_miss_on_a_file_larger_than_the_sample_falls_back_to_the_full_listing_once(
        self, tmp_path: Path
    ) -> None:
        path = self._wide(tmp_path)
        match = toy_match(path)
        with run_match_cache():
            assert ToyFileFormatFG.has_column(match, "toyfmt_wide_19") is True
            assert ToyFileFormatFG.has_column(match, "toyfmt_wide_18") is True
        names, _ = self._reads(path)
        assert len(names) == 1

    def test_a_sample_miss_on_a_file_within_the_sample_size_is_final(self, tmp_path: Path) -> None:
        path = self._small(tmp_path)
        assert ToyFileFormatFG.has_column(toy_match(path), "toyfmt_s_absent") is False
        names, samples = self._reads(path)
        assert samples
        assert not names

    def test_the_sample_listing_is_cached_separately_within_a_run(self, tmp_path: Path) -> None:
        path = self._small(tmp_path)
        match = toy_match(path)
        with run_match_cache():
            assert ToyFileFormatFG.has_column(match, "toyfmt_s_a") is True
            assert ToyFileFormatFG.has_column(match, "toyfmt_s_b") is True
            assert ToyFileFormatFG.columns(match) is not None
            assert ToyFileFormatFG.columns(match) is not None
        names, samples = self._reads(path)
        assert len(samples) == 1
        assert len(names) == 1


class _FirstCandidateRefusingExtender(Extender):
    """Runs the probe, then refuses the first file group probed; later groups pass."""

    never_fall_back = True

    def __init__(self) -> None:
        self.priority = 100
        self.name = "toyfmt_first_refuser"
        self.refused: list[str] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None and context.reader_class is not None and not self.refused:
            self.refused.append(context.reader_class.__name__)
            raise PermissionError("no clearance")
        return result


class TestPinnedFileOwnedBySomeFileGroup:
    """A pin to a file whose suffix no file format group owns is named in the resolution failure, once."""

    UNOWNED_FRAGMENT = "has a suffix no registered reader owns"

    def _dac(self, path: Path, column: str) -> DataAccessCollection:
        return DataAccessCollection(files={"pinned": str(path)}, column_to_file={column: "pinned"})

    def _unowned_pin_reasons(self, result: Any) -> list[str]:
        return [e.reason for e in result.eliminations.values() if self.UNOWNED_FRAGMENT in e.reason]

    def test_a_pin_to_a_file_group_suffix_records_no_unowned_pin(self, tmp_path: Path) -> None:
        path = tmp_path / f"pinned{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_g6_col": [1]})
        plugins: Any = {ToyFileFormatFG: {PyArrowTable}, CsvFG: {PyArrowTable}}

        result = IdentifyFeatureGroupClass.evaluate(
            Feature("toyfmt_g6_col"), plugins, None, self._dac(path, "toyfmt_g6_col")
        )

        assert ToyFileFormatFG in result.identified
        assert self._unowned_pin_reasons(result) == []

    @pytest.mark.parametrize(
        "module,name,suffix",
        [
            ("text_fg", "TextFG", ".TXT"),
            ("text_fg", "PyFG", ".py"),
            ("markdown_fg", "MarkdownFG", ".md"),
            ("yaml_fg", "YamlFG", ".yml"),
        ],
    )
    def test_a_pin_to_a_document_group_suffix_records_no_unowned_pin(
        self, tmp_path: Path, module: str, name: str, suffix: str
    ) -> None:
        group = load_document_group(module, name)
        path = tmp_path / f"pinned{suffix}"
        path.write_text("toyfmt_g6_doc\n", encoding="utf-8")
        plugins: Any = {group: {PyArrowTable}, ToyFileFormatFG: {PyArrowTable}}

        result = IdentifyFeatureGroupClass.evaluate(Feature(name), plugins, None, self._dac(path, name))

        assert self._unowned_pin_reasons(result) == []

    @pytest.mark.parametrize("refusing", [False, True], ids=["no_extender", "first_candidate_refused"])
    def test_a_pin_to_a_suffix_nobody_owns_is_named_once_in_the_resolution_failure(
        self, tmp_path: Path, refusing: bool
    ) -> None:
        path = tmp_path / "pinned.toyfmtunowned9"
        path.write_text("toyfmt_g6_other\n1\n")
        plugins: Any = {ToyFileFormatFG: {PyArrowTable}, CsvFG: {PyArrowTable}}
        extender = _FirstCandidateRefusingExtender() if refusing else None
        scope = ReaderProbeScope(extender, None, None, None, None) if extender is not None else None

        with reader_probe_scope(scope):
            with pytest.raises(FeatureResolutionError) as excinfo:
                evaluate_or_raise(Feature("toyfmt_g6_other"), plugins, None, self._dac(path, "toyfmt_g6_other"))

        if extender is not None:
            assert len(extender.refused) == 1
        assert str(excinfo.value).count(self.UNOWNED_FRAGMENT) == 1

    def test_the_unowned_pin_message_scrubs_uri_credentials(self) -> None:
        uri = "s3://AKIAKEY:supersecret@bucket/data.weirdext"
        dac = DataAccessCollection(files={"p": uri}, column_to_file={"col_x": "p"})
        plugins: Any = {CsvFG: {PyArrowTable}}

        with pytest.raises(FeatureResolutionError) as excinfo:
            evaluate_or_raise(Feature("col_x"), plugins, None, dac)

        text = str(excinfo.value)
        assert "supersecret" not in text
        assert self.UNOWNED_FRAGMENT in text


class TestUserinfoUriPlanIdentity:
    """A userinfo URI handle never shows its password in the plan step identity or the rendered plan."""

    def test_plan_step_identity_and_render_hide_the_password(self) -> None:
        column = "toyfmt_uri_plan_col"
        uri = f"s3://alice:hunter2@bucket/plan{TOY_SUFFIX}"  # nosec B105

        class ToyfmtUriPlanFG(ToyFileFormatFG):
            @classmethod
            def match_feature_group_criteria(
                cls, feature_name: Any, options: Options, data_access_collection: Any = None
            ) -> bool:
                if str(feature_name) != column:
                    return False
                return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))

            @classmethod
            def columns(cls, match: Any) -> Collection[str] | None:
                return [column]

            @classmethod
            def has_column(cls, match: Any, column_name: str) -> bool | None:
                return column_name == column

        explained = mloda.explain(
            [Feature(column)],
            compute_frameworks=[PyArrowTable],
            data_access_collection=DataAccessCollection(files={uri}),
            plugin_collector=PluginCollector.enabled_feature_groups({ToyfmtUriPlanFG}),
        )
        steps = [step for step in explained if step.step_kind == "compute"]
        assert len(steps) == 1
        assert steps[0].data_access_identity == f"s3://bucket/plan{TOY_SUFFIX}"
        assert "hunter2" not in repr(explained)
        assert "hunter2" not in str(explained)
        del explained, steps, ToyfmtUriPlanFG
        gc.collect()


class TestFileSourceUriProjection:
    """`_source` projects a URI path to a credential-free source and keeps the raw path as access."""

    def test_drops_userinfo_query_and_fragment_and_keeps_a_percent_encoded_path(self) -> None:
        source = ToyFileFormatFG._source
        uri = f"s3://alice:hunter2@bucket/my%20file{TOY_SUFFIX}?token=t#frag"  # nosec B105
        match = source(uri)
        assert match.source == f"s3://bucket/my%20file{TOY_SUFFIX}"
        assert match.access == uri
        first = source(f"s3://u:hunter2@bucket/a%2Fb{TOY_SUFFIX}").source  # nosec B105
        second = source(f"s3://u:hunter2@bucket/a%2Fc{TOY_SUFFIX}").source  # nosec B105
        assert first != second

    @pytest.mark.parametrize(
        "template,expected",
        [
            ("s3://AKIA:ab/hunter2@bucket/x{s}", "s3://bucket/x{s}"),  # nosec B105
            ("s3://u:hun#ter2@bucket/x{s}", "s3://bucket/x{s}"),  # nosec B105
            ("s3://u:hun?ter2@bucket/x{s}", "s3://bucket/x{s}"),  # nosec B105
            ("s3://tok@bucket/x{s}", "s3://bucket/x{s}"),
            ("s3://bucket/a@b{s}", "s3://bucket/a@b{s}"),
            ("s3://bucket:9000/x{s}", "s3://bucket:9000/x{s}"),
            ("abfss://cont@acct.dfs.core.windows.net/a%20b{s}", "abfss://cont@acct.dfs.core.windows.net/a%20b{s}"),
            ("simplecache::s3://tok@bucket/x{s}", "simplecache::s3://bucket/x{s}"),
            ("s3://bucket]/x{s}", "s3://bucket]/x{s}"),
            ("s3://[bad/x{s}", "s3://[bad/x{s}"),
        ],
        ids=[
            "slash-in-password",
            "hash-in-password",
            "question-in-password",
            "user-only",
            "at-in-path",
            "port",
            "azure",
            "chained",
            "bad-bracket-host",
            "open-bracket-host",
        ],
    )
    def test_userinfo_with_url_delimiters(self, template: str, expected: str) -> None:
        path = template.format(s=TOY_SUFFIX)
        match = ToyFileFormatFG._source(path)
        assert match.source == expected.format(s=TOY_SUFFIX)
        assert match.access == path


class TestUriSourceCollision:
    """Two handles with one credential-free source but different raw paths get `source@handle` sources."""

    COLUMN = "toyfmt_collision_col"

    def _group(self) -> type[ToyFileFormatFG]:
        column = self.COLUMN

        class ToyfmtCollisionFG(ToyFileFormatFG):
            @classmethod
            def match_feature_group_criteria(
                cls, feature_name: Any, options: Options, data_access_collection: Any = None
            ) -> bool:
                if str(feature_name) != column:
                    return False
                return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))

            @classmethod
            def columns(cls, match: Any) -> Collection[str] | None:
                return [column]

            @classmethod
            def has_column(cls, match: Any, column_name: str) -> bool | None:
                return column_name == column

        return ToyfmtCollisionFG

    def test_find_sources_suffixes_each_colliding_source_with_its_handle(self) -> None:
        group = self._group()
        dac = DataAccessCollection(
            files={"a": f"s3://alice:pw1@b/x{TOY_SUFFIX}", "b": f"s3://bob:pw2@b/x{TOY_SUFFIX}"}  # nosec B105
        )
        matches = group.find_sources(group.CLAIM_ROUTES[0], self.COLUMN, Options(), dac)
        assert sorted(match.source for match in matches) == [f"s3://b/x{TOY_SUFFIX}@a", f"s3://b/x{TOY_SUFFIX}@b"]
        assert {match.access for match in matches} == {
            f"s3://alice:pw1@b/x{TOY_SUFFIX}",  # nosec B105
            f"s3://bob:pw2@b/x{TOY_SUFFIX}",  # nosec B105
        }
        del matches, group
        gc.collect()

    def test_a_single_handle_keeps_the_unsuffixed_source(self) -> None:
        group = self._group()
        dac = DataAccessCollection(files={"a": f"s3://alice:pw1@b/x{TOY_SUFFIX}"})  # nosec B105
        matches = group.find_sources(group.CLAIM_ROUTES[0], self.COLUMN, Options(), dac)
        assert [match.source for match in matches] == [f"s3://b/x{TOY_SUFFIX}"]
        del matches, group
        gc.collect()

    def test_the_ambiguity_abort_names_both_suffixed_sources_without_the_passwords(self) -> None:
        group = self._group()
        dac = DataAccessCollection(
            files={"a": f"s3://alice:pw1@b/x{TOY_SUFFIX}", "b": f"s3://bob:pw2@b/x{TOY_SUFFIX}"}  # nosec B105
        )
        plugins: Any = {group: {PyArrowTable}}
        with pytest.raises(ValueError) as excinfo:
            evaluate_or_raise(Feature(self.COLUMN), plugins, None, dac)
        text = str(excinfo.value)
        assert "several sources" in text
        assert f"s3://b/x{TOY_SUFFIX}@a" in text
        assert f"s3://b/x{TOY_SUFFIX}@b" in text
        assert "pw1" not in text
        assert "pw2" not in text
        del excinfo, plugins, group
        gc.collect()


class TestAbstractReadFileFGWithAPin:
    """A column_to_file pin must not make the abstract ReadFileFG raise from its match hook."""

    def _reasons(self, result: Any) -> list[str]:
        return [e.reason for e in result.eliminations.values()]

    def test_probed_next_to_a_concrete_group_it_declines_without_a_match_hook_error(self, tmp_path: Path) -> None:
        path = tmp_path / f"pinned{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_abs_pin_col": [1]})
        dac = DataAccessCollection(files={"abs_pin": str(path)}, column_to_file={"toyfmt_abs_pin_col": "abs_pin"})
        plugins: Any = {ReadFileFG: {PyArrowTable}, ToyFileFormatFG: {PyArrowTable}}

        result = IdentifyFeatureGroupClass.evaluate(Feature("toyfmt_abs_pin_col"), plugins, None, dac)

        assert ToyFileFormatFG in result.identified
        assert not [r for r in self._reasons(result) if "raised TypeError" in r]
        assert ReadFileFG not in result.identified

    def test_a_pin_to_a_file_no_group_owns_declines_without_a_match_hook_error(self, tmp_path: Path) -> None:
        path = tmp_path / "pinned.toyfmtnobody"
        path.write_text("toyfmt_abs_pin_other\n1\n")
        dac = DataAccessCollection(files={"abs_pin": str(path)}, column_to_file={"toyfmt_abs_pin_other": "abs_pin"})
        plugins: Any = {ReadFileFG: {PyArrowTable}, ToyFileFormatFG: {PyArrowTable}}

        result = IdentifyFeatureGroupClass.evaluate(Feature("toyfmt_abs_pin_other"), plugins, None, dac)

        assert result.identified == {}
        assert not [r for r in self._reasons(result) if "raised TypeError" in r]


_CSVFG_COL = "sibling_sel_csvfg_col"


def _csv_file(tmp_path: Path) -> str:
    path = tmp_path / "sibling_sel_csvfg.csv"
    path.write_text(f"{_CSVFG_COL}\n1\n2\n")
    return str(path)


class TestClassKeyNormalizationOnCsvFG:
    """Groups B and E on the stock CsvFG: a format group class as an option key equals its class-name string."""

    def test_options_class_key_equals_string_key(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        class_keyed = Options(cast(dict[str, Any], {CsvFG: path}))
        string_keyed = Options({"CsvFG": path})
        assert class_keyed == string_keyed
        assert hash(class_keyed) == hash(string_keyed)
        assert class_keyed.get("CsvFG") == path

    def test_feature_class_key_equals_string_key(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        class_keyed = Feature(name=_CSVFG_COL, options=cast(dict[str, Any], {CsvFG: path}))
        string_keyed = Feature(name=_CSVFG_COL, options={"CsvFG": path})
        assert class_keyed == string_keyed
        assert hash(class_keyed) == hash(string_keyed)
        assert len({class_keyed, string_keyed}) == 1

    def test_class_key_resolves_like_the_string_form(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        feature = Feature(name=_CSVFG_COL, options=cast(dict[str, Any], {CsvFG: path}))

        result = IdentifyFeatureGroupClass.evaluate(feature, {CsvFG: {PyArrowTable}}, None)

        assert CsvFG in result.identified
        assert feature.input_data_match is not None
        assert feature.input_data_match[0] is CsvFG
        assert feature.input_data_match[1].source == os.path.abspath(path)

    def test_child_options_with_class_key_hashable(self, tmp_path: Path) -> None:
        feature = Feature(name="sibling_sel_csvfg_child_feat")
        feature.child_options = Options(
            cast(dict[str, Any], {CsvFG: _csv_file(tmp_path), "sibling_sel_csvfg_child_marker": True})
        )
        assert isinstance(hash(feature), int)

    def test_options_get_by_string_after_class_key_construction(self, tmp_path: Path) -> None:
        path = _csv_file(tmp_path)
        options = Options(cast(dict[str, Any], {CsvFG: path}))
        assert options.get(CsvFG.__name__) == path


class TestSecondCsvFormatGroupIsAmbiguous:
    """Two format groups owning the same suffix on the same file are ambiguous; no reader-level pick exists."""

    def test_a_gated_second_csv_group_next_to_csvfg_gives_multiple_feature_groups(self, tmp_path: Path) -> None:
        marker = "sibling_sel_second_csv_marker"

        class SiblingSelSecondCsvFG(ReadFileFG):
            """Owns .csv too; gated by a unique marker option so it never claims in other tests."""

            @classmethod
            def suffixes(cls) -> tuple[str, ...]:
                return (".csv",)

            @classmethod
            def column_names(cls, path: str) -> Collection[str]:
                with open(path, encoding="utf-8") as handle:
                    return handle.readline().strip().split(",")

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: DataAccessCollection | None = None,
            ) -> bool:
                if options.get(marker) is None:
                    return False
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        path = _csv_file(tmp_path)
        feature = Feature(_CSVFG_COL, Options(context={marker: True}))
        plugins: FeatureGroupEnvironmentMapping = {CsvFG: {PyArrowTable}, SiblingSelSecondCsvFG: {PyArrowTable}}

        with pytest.raises(ValueError, match="Multiple feature groups found") as excinfo:
            resolve_or_raise(feature, plugins, None, DataAccessCollection(files={path}))

        assert "CsvFG" in str(excinfo.value)
        assert "SiblingSelSecondCsvFG" in str(excinfo.value)
