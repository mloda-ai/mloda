"""ReadDocumentFG base behaviour through small test-local document groups on the ``.toydoc*`` suffixes."""

from __future__ import annotations

import gc
import importlib
import inspect
import os
from pathlib import Path
from typing import Any

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.utils import escalate_match_abort, is_match_abort
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass, resolve_or_raise
from mloda.provider import FormatFeatureGroup, ReadDocumentFG, ReadFileFG
from mloda.user import DataAccessCollection, DataType, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class ToyDocFG(ReadDocumentFG):
    """Declares only its suffixes; every other hook is the base default."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".toydoc", ".TOYDOC")


class ToyReadTextDocFG(ReadDocumentFG):
    """Overrides the text hook only."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".toyreadtext",)

    @classmethod
    def read_text(cls, path: str) -> str:
        return "toy::" + Path(path).read_text(encoding="utf-8").upper()


class ToyHandoverDocFG(ReadDocumentFG):
    """Hands its suffix over: it claims only when a feature lists the suffix in document_suffixes."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".toyhandover",)

    @classmethod
    def handover_suffixes(cls) -> tuple[str, ...]:
        return (".toyhandover",)


class _Features:
    def get_all_names(self) -> set[str]:
        return {"ToyDocFG", "ToyDocFG~source", "ToyDocFG~file_type"}


def _write(tmp_path: Path, name: str, text: str = "toy body\n") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _claims(group: type[Any], feature: Feature, dac: DataAccessCollection | None) -> bool:
    result = IdentifyFeatureGroupClass.evaluate(feature, {group: {PythonDictFramework}}, None, dac)
    return group in result.identified


class TestReadDocumentFGBase:
    def test_is_an_abstract_format_group_exported_from_provider_asking_only_for_suffixes(self) -> None:
        assert provider.ReadDocumentFG is ReadDocumentFG
        assert "ReadDocumentFG" in provider.__all__
        assert issubclass(ReadDocumentFG, FormatFeatureGroup)
        assert inspect.isabstract(ReadDocumentFG)
        assert set(ReadDocumentFG.__abstractmethods__) == {"suffixes"}
        assert not inspect.isabstract(ToyDocFG)

    def test_declares_declared_searched_file_and_folder_routes(self) -> None:
        assert ReadDocumentFG.CLAIM_ROUTES == (
            ClaimRoute("file", NamePolicy.DECLARED, True),
            ClaimRoute("folder", NamePolicy.DECLARED, True),
        )

    def test_declares_the_same_reader_options_as_the_file_groups(self) -> None:
        assert set(ReadDocumentFG.PROPERTY_MAPPING) == set(ReadFileFG.PROPERTY_MAPPING)
        assert set(ReadDocumentFG.PROPERTY_MAPPING) == {"data_access_handle", "document_suffixes"}

    def test_the_docstring_points_third_party_groups_at_the_contract_mixins(self) -> None:
        doc = ReadDocumentFG.__doc__ or ""
        assert "tests/mixins/reader_feature_groups" in doc
        assert "document_format_feature_group_test_mixin" in doc

    def test_the_abstract_base_never_claims_a_feature_named_after_it(self, tmp_path: Path) -> None:
        dac = DataAccessCollection(files={"toy": str(_write(tmp_path, "a.toydoc"))})
        assert not _claims(ReadDocumentFG, Feature("ReadDocumentFG"), dac)

    def test_the_default_hooks(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "unicode.toydoc", "héllo 世界\n")
        assert ToyDocFG.handover_suffixes() == ()
        assert ToyDocFG.read_text(str(path)) == "héllo 世界\n"
        assert ToyDocFG.file_type("some/dir.d/UPPER.TOYDOC") == "toydoc"
        assert ToyDocFG.file_type("some/dir.d/lower.toydoc") == "toydoc"

    def test_describe_columns_lists_three_string_outputs_and_count_rows_is_one(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.toydoc")
        match = SourceMatch(source=str(path), access=str(path))
        assert ToyDocFG.describe_columns(match) == {
            "ToyDocFG": DataType.STRING,
            "ToyDocFG~source": DataType.STRING,
            "ToyDocFG~file_type": DataType.STRING,
        }
        assert ToyDocFG.count_rows(match, PythonDictFramework) == 1

    def test_the_neutral_form_is_a_columnar_dict_of_one_row(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.TOYDOC", "neutral body\n")
        neutral = ToyDocFG.load_neutral(SourceMatch(source=str(path), access=str(path)), _Features())
        assert neutral == {
            "ToyDocFG": ["neutral body\n"],
            "ToyDocFG~source": [str(path)],
            "ToyDocFG~file_type": ["toydoc"],
        }


class TestProviderHooks:
    def test_a_group_declaring_only_suffixes_claims_and_loads_end_to_end(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.toydoc", "end to end\n")

        result = mloda.run_all(
            ["ToyDocFG", "ToyDocFG~source", "ToyDocFG~file_type"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyDocFG}),
            data_access_collection=DataAccessCollection(files={"toy": str(path)}),
        )

        assert result == [
            {"ToyDocFG": ["end to end\n"], "ToyDocFG~source": [str(path)], "ToyDocFG~file_type": ["toydoc"]}
        ]

    def test_an_overridden_read_text_supplies_the_content(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a.toyreadtext", "shout\n")

        result = mloda.run_all(
            ["ToyReadTextDocFG"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyReadTextDocFG}),
            data_access_collection=DataAccessCollection(files={"toy": str(path)}),
        )

        assert result == [{"ToyReadTextDocFG": ["toy::SHOUT\n"]}]

    def test_a_handed_over_suffix_is_claimed_only_when_the_feature_lists_it(self, tmp_path: Path) -> None:
        dac = DataAccessCollection(files={"toy": str(_write(tmp_path, "a.toyhandover"))})
        plain = Feature("ToyHandoverDocFG")
        listed = Feature("ToyHandoverDocFG", Options(context={"document_suffixes": frozenset({".toyhandover"})}))
        other = Feature("ToyHandoverDocFG", Options(context={"document_suffixes": frozenset({".toyother"})}))

        assert not _claims(ToyHandoverDocFG, plain, dac)
        assert _claims(ToyHandoverDocFG, listed, dac)
        assert not _claims(ToyHandoverDocFG, other, dac)

    def test_a_handed_over_suffix_is_declined_for_a_pointer_and_a_folder_without_the_listing(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, "a.toyhandover")
        pointed = Feature("ToyHandoverDocFG", Options({"ToyHandoverDocFG": str(path)}))
        folder = DataAccessCollection(folders={"toy_dir": str(tmp_path)})

        assert not _claims(ToyHandoverDocFG, pointed, None)
        assert not _claims(ToyHandoverDocFG, Feature("ToyHandoverDocFG"), folder)

    @pytest.mark.parametrize("source_kind", ["file_handle", "folder_entry"])
    def test_a_collection_file_with_only_a_handed_over_suffix_names_document_suffixes_and_drops_the_prefix_hint(
        self, tmp_path: Path, source_kind: str
    ) -> None:
        path = _write(tmp_path, "a.toyhandover")
        dac = (
            DataAccessCollection(files={"toy": str(path)})
            if source_kind == "file_handle"
            else DataAccessCollection(folders={"toy_dir": str(tmp_path)})
        )
        feature = Feature("ToyHandoverDocFG")
        plugins: FeatureGroupEnvironmentMapping = {ToyHandoverDocFG: {PythonDictFramework}}

        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None, dac)
        assert "document_suffixes" in result.eliminations[ToyHandoverDocFG].reason

        with pytest.raises(ValueError) as exc_info:
            resolve_or_raise(feature, plugins, None, dac)
        message = str(exc_info.value)
        assert "document_suffixes" in message
        assert "ToyHandoverDocFG_" not in message

    def test_the_handed_over_hint_skips_files_outside_the_named_data_access_handle(self, tmp_path: Path) -> None:
        unrelated = _write(tmp_path, "unrelated_handed.toyhandover")
        named = _write(tmp_path, "named.unrelatedsuffix")
        dac = DataAccessCollection(files={"other_h": str(unrelated), "mine_h": str(named)})
        feature = Feature("ToyHandoverDocFG", Options({"data_access_handle": "mine_h"}))
        plugins: FeatureGroupEnvironmentMapping = {ToyHandoverDocFG: {PythonDictFramework}}

        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None, dac)

        assert ToyHandoverDocFG not in result.identified
        elimination = result.eliminations.get(ToyHandoverDocFG)
        assert elimination is None or "unrelated_handed" not in elimination.reason

    def test_the_handed_over_hint_skips_a_folder_entry_that_is_not_a_regular_file(self, tmp_path: Path) -> None:
        (tmp_path / "dir_entry.toyhandover").mkdir()
        dac = DataAccessCollection(folders={"toy_dir": str(tmp_path)})
        plugins: FeatureGroupEnvironmentMapping = {ToyHandoverDocFG: {PythonDictFramework}}

        result = IdentifyFeatureGroupClass.evaluate(Feature("ToyHandoverDocFG"), plugins, None, dac)

        elimination = result.eliminations.get(ToyHandoverDocFG)
        assert elimination is None or "dir_entry" not in elimination.reason

    @pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="symlinks need POSIX")
    def test_the_handed_over_hint_skips_a_folder_entry_linking_outside_its_folder(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "elsewhere_target.toyhandover")
        folder = tmp_path / "hint_folder"
        folder.mkdir()
        os.symlink(target, folder / "escape_link.toyhandover")
        dac = DataAccessCollection(folders={"toy_dir": str(folder)})
        plugins: FeatureGroupEnvironmentMapping = {ToyHandoverDocFG: {PythonDictFramework}}

        result = IdentifyFeatureGroupClass.evaluate(Feature("ToyHandoverDocFG"), plugins, None, dac)

        elimination = result.eliminations.get(ToyHandoverDocFG)
        assert elimination is None or "escape_link" not in elimination.reason

    def test_the_matched_but_rejection_masks_a_credential_in_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "token=toyfmt-secret-1234.toydoc"
        dac = DataAccessCollection(files={"toy": str(missing)})

        result = IdentifyFeatureGroupClass.evaluate(Feature("ToyDocFG"), {ToyDocFG: {PythonDictFramework}}, None, dac)

        reason = result.eliminations[ToyDocFG].reason
        assert "matched" in reason
        assert "toyfmt-secret-1234" not in reason
        assert "token=***" in reason

    def test_a_marked_abort_from_suffixes_escapes_matching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Function-local group: a module-level one raising from suffixes() would break every other test's discovery."""

        def suffixes(cls: Any) -> tuple[str, ...]:
            raise escalate_match_abort(NotImplementedError("toy doc marked abort"))

        group: Any = type(
            "ToyMarkedAbortSuffixDocFG", (ReadDocumentFG,), {"suffixes": classmethod(lambda cls: (".toy",))}
        )
        monkeypatch.setattr(group, "suffixes", classmethod(suffixes))
        dac = DataAccessCollection(files={"toy": "/path/to/doc.toy"})

        with pytest.raises(NotImplementedError) as excinfo:
            group.match_feature_group_criteria("ToyMarkedAbortSuffixDocFG", Options(), dac)

        assert is_match_abort(excinfo.value)
        del excinfo, group
        gc.collect()


class TestStockFormats:
    def test_stock_formats_exports_every_document_group(self) -> None:
        stock = importlib.import_module("mloda_plugins.feature_group.input_data.file_formats.stock_formats")
        for name in ("TextFG", "PyFG", "MarkdownFG", "YamlFG", "JsonDocumentFG"):
            assert name in stock.__all__
            assert getattr(stock, name).get_class_name() == name


class TestSiblingsUnderASharedFileSourceBase:
    def test_a_document_group_is_not_a_file_group(self) -> None:
        assert not issubclass(ReadDocumentFG, ReadFileFG)

    def test_both_share_an_abstract_format_base_that_is_neither_of_them(self) -> None:
        common = [
            base
            for base in ReadDocumentFG.__mro__
            if base in ReadFileFG.__mro__ and base not in (object, FormatFeatureGroup)
        ]
        assert common
        shared = common[0]
        assert issubclass(shared, FormatFeatureGroup)
        assert shared not in (FormatFeatureGroup, ReadFileFG, ReadDocumentFG)
        assert inspect.isabstract(shared)

    def test_a_document_group_has_no_column_names_listing(self) -> None:
        assert not hasattr(ReadDocumentFG, "column_names")

    def test_a_document_group_does_not_inherit_the_file_format_ambiguity_fix(self) -> None:
        definer = next(base for base in ReadDocumentFG.__mro__ if "ambiguity_fix" in vars(base))
        assert definer is not ReadFileFG
