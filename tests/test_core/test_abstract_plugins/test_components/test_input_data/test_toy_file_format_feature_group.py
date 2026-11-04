"""ReadFileFG through the stdlib ``.toyfmtfile`` toy: shared file contract, framework loads, sample hook."""

from __future__ import annotations

import inspect
import itertools
import os
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.match_rejection import CoreDeclineCode
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.api.plan_info import source_of
from mloda.provider import ReadFileFG, SourceFeatureGroup
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
from tests.mixins.reader_feature_groups.lazy_source_group import load_document_group
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    FileLoadsIntoFrameworkMixin,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from mloda._testing.source.read_file_suite import ReadFileFGTestMixin
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_file_format_group import (
    SAMPLE_SIZE,
    TOY_SUFFIX,
    ToyFileFormatFG,
    toy_match,
    write_toy_file,
)


class TestToyFileFormatFeatureGroup(ReadFileFGTestMixin):
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

    def test_is_an_abstract_source_group_exported_from_provider(self) -> None:
        assert provider.ReadFileFG is ReadFileFG
        assert "ReadFileFG" in provider.__all__
        assert issubclass(ReadFileFG, SourceFeatureGroup)
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
    """A pin to a file whose suffix no file source group owns is named in the resolution failure, once."""

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
        unowned = [e for e in excinfo.value.result.eliminations.values() if self.UNOWNED_FRAGMENT in e.reason]
        assert [e.code for e in unowned] == [CoreDeclineCode.NO_READER_FOR_SUFFIX]

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

    @pytest.mark.parametrize(
        "template,expected,secret,credentials",
        [
            ("s3://alice:hunter2@bucket/plan{s}", "s3://bucket/plan{s}", "hunter2", None),  # nosec B105
            (
                "s3://bucket/dir;jsessionid=SECRETSESSION/plan{s}",  # nosec B105
                "s3://bucket/dir;***/plan{s}",
                "SECRETSESSION",
                None,
            ),
            (
                "s3://bucket/dir;%74oken=SECRETENCODED/plan{s}",  # nosec B105
                "s3://bucket/dir;***/plan{s}",
                "SECRETENCODED",
                None,
            ),
            (
                "s3://bucket/botLITERALTOKEN1/plan{s}",
                "s3://bucket/bot***/plan{s}",
                "LITERALTOKEN1",  # nosec B105
                {"h": {"token": "LITERALTOKEN1"}},  # nosec B105
            ),
            (
                "s3://bucket/auth=LITERALTOKEN2/plan{s}",
                "s3://bucket/auth=***/plan{s}",
                "LITERALTOKEN2",  # nosec B105
                {"h": {"token": "LITERALTOKEN2"}},  # nosec B105
            ),
        ],
        ids=[
            "userinfo-password",
            "session-id-path-param",
            "encoded-path-param",
            "literal-path-segment",
            "literal-hive-segment",
        ],
    )
    def test_plan_step_identity_and_render_hide_the_password(
        self, template: str, expected: str, secret: str, credentials: dict[str, Any] | None
    ) -> None:
        column = "toyfmt_uri_plan_col"
        uri = template.format(s=TOY_SUFFIX)

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

        def collection() -> DataAccessCollection:
            if credentials is None:
                return DataAccessCollection(files={uri})
            return DataAccessCollection(files={uri}, credentials=credentials)

        session = mloda.prepare(
            [Feature(column)],
            compute_frameworks=[PyArrowTable],
            data_access_collection=collection(),
            plugin_collector=PluginCollector.enabled_feature_groups({ToyfmtUriPlanFG}),
        )
        explained = session.resolved_plan()
        steps = [step for step in explained if step.step_kind == "compute"]
        assert len(steps) == 1
        assert steps[0].data_access_identity == expected.format(s=TOY_SUFFIX)
        assert secret not in repr(explained)
        assert secret not in str(explained)
        if credentials is not None:
            assert secret in source_of(session._execution_plan(), steps[0])[1].source
        del session, explained, steps, ToyfmtUriPlanFG

    @pytest.mark.parametrize(
        ("key", "masked"),
        [("token", True), ("botref", False)],
        ids=["credential_shaped_key_masked", "non_shaped_key_kept"],
    )
    def test_a_provider_identity_override_masks_a_registered_token_only_under_a_credential_shaped_key(
        self, key: str, masked: bool
    ) -> None:
        column = "toyfmt_uri_override_col"
        token = "OVERRIDETOKEN9"  # nosec B105
        expected = "api://example/bot***/x.json" if masked else f"api://example/bot{token}/x.json"

        class ToyfmtUriOverrideFG(ToyFileFormatFG):
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

            @classmethod
            def data_access_identity(cls, match: Any) -> str:
                return f"api://example/bot{token}/x.json"

        plan = mloda.explain(
            [Feature(column)],
            compute_frameworks=[PyArrowTable],
            data_access_collection=DataAccessCollection(
                files={f"s3://bucket/override{TOY_SUFFIX}"}, credentials={"h": {key: token}}
            ),
            plugin_collector=PluginCollector.enabled_feature_groups({ToyfmtUriOverrideFG}),
        )
        steps = [step for step in plan if step.step_kind == "compute"]
        assert len(steps) == 1
        assert steps[0].data_access_identity == expected
        assert (token in repr(plan)) is (not masked)
        del plan, steps, ToyfmtUriOverrideFG


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

    def test_long_encoded_head_cuts_quickly(self) -> None:
        uri = "s3://b/" + "%25" * 8000 + ";x/x" + TOY_SUFFIX
        started = time.monotonic()
        match = ToyFileFormatFG._source(uri)
        assert time.monotonic() - started < 5
        assert match.source == "s3://b/" + "%25" * 8000 + ";***/x" + TOY_SUFFIX
        assert match.access == uri

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
            ("https://host/x{s};jsessionid=SECRET", "https://host/x{s};***"),  # nosec B105
            ("s3://bucket/dir;jsessionid=SECRET/x{s}", "s3://bucket/dir;***/x{s}"),  # nosec B105
            ("https://host/x{s};PHPSESSID=abc", "https://host/x{s};***"),  # nosec B105
            ("https://host/x{s};token=abc", "https://host/x{s};***"),  # nosec B105
            ("https://host/x{s};API-KEY=abc", "https://host/x{s};***"),  # nosec B105
            ("https://host/x{s};x-amz-signature=abc", "https://host/x{s};***"),  # nosec B105
            ("https://host/x{s};sessid=abc", "https://host/x{s};***"),  # nosec B105
            ("https://host/d;v=1;token=abc;lang=en/x{s}", "https://host/d;***/x{s}"),  # nosec B105
            ("https://host/d;token=a;v=1/x{s}", "https://host/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token=SE::CRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token::SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("https://host/x{s};token=", "https://host/x{s};***"),
            ("s3://u:pw@bucket/d;token=abc/x{s}?q=1#f", "s3://bucket/d;***/x{s}"),  # nosec B105
            ("https://host/x{s};v=1", "https://host/x{s};***"),
            ("https://host/x{s};type=a", "https://host/x{s};***"),
            ("https://host/d;tokens=3/x{s}", "https://host/d;***/x{s}"),
            ("https://host/d;session_id=1/x{s}", "https://host/d;***/x{s}"),
            ("https://host/clean/x{s}", "https://host/clean/x{s}"),
            ("d;token=abc/x{s}", os.path.abspath("d;token=abc/x{s}")),
            ("s3://b/d;aws.token=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;x~token=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;v=1,token=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;v=1&token=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token = SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token=a,b/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("zip://d;token=SECRET::s3://bucket/x{s}", "zip://d;***::s3://bucket/x{s}"),  # nosec B105
            ("s3://b/token=abc;v=1/x{s}", "s3://b/token=abc;***/x{s}"),  # nosec B105
            ("s3://b/my.token.dir;v=1/x{s}", "s3://b/my.token.dir;***/x{s}"),
            ("s3://b/d;%74oken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token%3DSECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d%3Btoken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;%2574oken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token%3DSE%2FCRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;%252525252525" + "74oken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("zip://d;%74oken=SECRET::s3://bucket/x{s}", "zip://d;***::s3://bucket/x{s}"),  # nosec B105
            ("s3://u:pw@b/d;%74oken=SECRET/x{s}?q=1", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/100%25{s}", "s3://b/100%25{s}"),
            ("s3://b/100%2541{s}", "s3://b/100%2541{s}"),
            ("s3://b/my%2520file{s}", "s3://b/my%2520file{s}"),
            ("s3://b/d;v%3D1/x{s}", "s3://b/d;***/x{s}"),
            ("s3://b/d%3Bv=1/x{s}", "s3://b/d;***/x{s}"),
            ("s3://b/token%3Dabc;v=1/x{s}", "s3://b/token%3Dabc;***/x{s}"),  # nosec B105
            ("s3://b/d;v=1%3Btoken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;%2F%74oken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;%252F%74oken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;v%3A%3A%74oken=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/my%25252520file{s}", "s3://b/my%25252520file{s}"),
            ("s3://b/a%2Ftoken=abc/x{s}", "s3://b/a%2Ftoken=abc/x{s}"),  # nosec B105
            ("s3://b/d;v=1%2Fx/y{s}", "s3://b/d;***/y{s}"),
            ("https://host/x{s};auth=SECRET", "https://host/x{s};***"),  # nosec B105
            ("s3://b/d;key=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;session=SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;token:SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d;SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/x%3Fauth=SECRET/x{s}", "s3://b/x;***/x{s}"),  # nosec B105
            ("s3://b/x%23auth=SECRET/x{s}", "s3://b/x;***/x{s}"),  # nosec B105
            ("s3://b/my%20dir;v=1/x{s}", "s3://b/my%20dir;***/x{s}"),
            ("s3://b/%3Bx/y{s}", "s3://b/;***/y{s}"),
            ("s3://b/d;;SECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("s3://b/d%3B%53ECRET/x{s}", "s3://b/d;***/x{s}"),  # nosec B105
            ("https://host/x{s};", "https://host/x{s};"),
            ("s3://b/d%3F/x{s}", "s3://b/d%3F/x{s}"),
            ("s3://b/d%23/x{s}", "s3://b/d%23/x{s}"),
            ("s3://b/year=2024/token=the/x{s}", "s3://b/year=2024/token=the/x{s}"),  # nosec B105
            ("s3://b/my%2525252520file{s}", "s3://b/my;***"),
            ("ftp://h/x{s};type=i", "ftp://h/x{s};***"),
            ("http://[::1]/d;v=1/x{s}", "http://[::1]/d;***/x{s}"),
            ("s3://bucket;x/y{s}", "s3://bucket;***/y{s}"),
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
            "jsessionid-trailing",
            "jsessionid-stops-at-slash",
            "phpsessid-uppercase",
            "token",
            "api-key-uppercase",
            "prefixed-signature",
            "sessid",
            "secret-among-others",
            "secret-first",
            "double-colon-in-value",
            "double-colon-assignment",
            "empty-value",
            "with-userinfo-query-fragment",
            "kept-v",
            "kept-type",
            "kept-tokens",
            "kept-session-underscore-id",
            "kept-clean-uri",
            "kept-local-path",
            "dotted-key",
            "tilde-key",
            "comma-separated-param",
            "ampersand-in-param",
            "whitespace-around-equals",
            "comma-in-value",
            "chain-separator-kept",
            "kept-secret-text-before-semicolon",
            "kept-dot-in-filename",
            "encoded-key",
            "encoded-equals",
            "encoded-semicolon",
            "double-encoded-key",
            "encoded-slash-in-value",
            "decode-cap-hit",
            "encoded-chain-kept",
            "encoded-with-userinfo-query",
            "kept-percent-literal",
            "kept-double-encoded-percent-digits",
            "kept-double-encoded-space",
            "kept-encoded-equals-param",
            "kept-encoded-semicolon-param",
            "kept-encoded-equals-before-param",
            "raw-masked-then-decoded",
            "encoded-slash-before-key",
            "double-encoded-slash-before-key",
            "encoded-chain-before-key",
            "kept-four-level-encoded-clean",
            "kept-encoded-slash-no-param",
            "kept-encoded-slash-clean-param",
            "unnamed-auth-key",
            "unnamed-key",
            "unnamed-session",
            "colon-assignment",
            "bare-region",
            "encoded-question-mark",
            "encoded-hash",
            "encoded-head-kept",
            "encoded-marker-first",
            "double-semicolon-head",
            "encoded-semicolon-encoded-tail",
            "kept-lone-trailing-semicolon",
            "kept-lone-trailing-encoded-question",
            "kept-lone-trailing-encoded-hash",
            "kept-hive-segments",
            "unmarked-over-decode-cap",
            "ftp-type",
            "ipv6-host",
            "host-part",
        ],
    )
    def test_uri_projection(self, template: str, expected: str) -> None:
        path = template.format(s=TOY_SUFFIX)
        match = ToyFileFormatFG._source(path)
        assert match.source == expected.format(s=TOY_SUFFIX)
        assert match.access == path

    def test_generated_matrix(self) -> None:
        secret = "S3CR3TVAL"  # nosec B105
        plain = {
            "mid": "s3://bucket/year=2024/dirA{r}/my%20dir/data" + TOY_SUFFIX,
            "file": "s3://bucket/year=2024/my%20dir/data" + TOY_SUFFIX + "{r}",
            "first": "s3://bucket/dirA{r}/year=2024/my%20dir/data" + TOY_SUFFIX,
        }
        zipped = {
            "mid": "zip://year=2024/dirA{r}/my%20dir/a.zip::s3://bucket/inner" + TOY_SUFFIX,
            "file": "zip://year=2024/my%20dir/a.zip{r}::s3://bucket/inner" + TOY_SUFFIX,
            "first": "zip://dirA{r}/year=2024/my%20dir/a.zip::s3://bucket/inner" + TOY_SUFFIX,
        }
        chains = {"plain": plain, "zip": zipped}

        def encode(sep: str, depth: int) -> str:
            for _ in range(depth):
                sep = quote(sep, safe="")
            return sep

        def fixpoint(text: str) -> str:
            for _ in range(20):
                nxt = unquote(text)
                if nxt == text:
                    break
                text = nxt
            return text

        failures: list[str] = []
        cases = 0
        for key, sep, assign, depth, position, chain in itertools.product(
            ["token", "auth", "key", "session", "jsessionid", ""],
            [";", "?", "#"],
            ["=", ":", "::", ""],
            range(6),
            ["mid", "file", "first"],
            ["plain", "zip"],
        ):
            cases += 1
            template = chains[chain][position]
            region = encode(sep, depth) + key + assign + secret
            uri = template.format(r=region)
            if depth == 0 and sep in "?#":
                expected = uri[: uri.index(sep)]
            else:
                expected = template.format(r=";***")
            label = f"key={key!r} sep={sep!r} assign={assign!r} depth={depth} position={position} chain={chain}"
            match = ToyFileFormatFG._source(uri)
            if match.source != expected:
                failures.append(f"{label}: source {match.source!r} != {expected!r}")
            if secret in match.source or secret in fixpoint(match.source):
                failures.append(f"{label}: secret leaked in {match.source!r}")
            if match.access != uri:
                failures.append(f"{label}: access {match.access!r} != raw {uri!r}")

        assert cases == 2592
        assert not failures, f"{len(failures)} failures, first 20:\n" + "\n".join(failures[:20])


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

    def test_handles_differing_only_by_a_session_id_param_get_suffixed_masked_sources(self) -> None:
        group = self._group()
        dac = DataAccessCollection(
            files={
                "a": f"s3://b/d;jsessionid=AAA/x{TOY_SUFFIX}",  # nosec B105
                "b": f"s3://b/d;jsessionid=BBB/x{TOY_SUFFIX}",  # nosec B105
            }
        )
        matches = group.find_sources(group.CLAIM_ROUTES[0], self.COLUMN, Options(), dac)
        assert sorted(match.source for match in matches) == [
            f"s3://b/d;***/x{TOY_SUFFIX}@a",
            f"s3://b/d;***/x{TOY_SUFFIX}@b",
        ]
        del matches, group

    def test_a_single_handle_keeps_the_unsuffixed_source(self) -> None:
        group = self._group()
        dac = DataAccessCollection(files={"a": f"s3://alice:pw1@b/x{TOY_SUFFIX}"})  # nosec B105
        matches = group.find_sources(group.CLAIM_ROUTES[0], self.COLUMN, Options(), dac)
        assert [match.source for match in matches] == [f"s3://b/x{TOY_SUFFIX}"]
        del matches, group

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
    """Groups B and E on the stock CsvFG: a source group class as an option key equals its class-name string."""

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


class TestSecondCsvSourceGroupIsAmbiguous:
    """Two source groups owning the same suffix on the same file are ambiguous; no reader-level pick exists."""

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
