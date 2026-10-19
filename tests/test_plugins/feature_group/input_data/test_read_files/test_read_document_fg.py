"""ReadDocumentFG base behaviour through small test-local document groups on the ``.toydoc*`` suffixes."""

from __future__ import annotations

import gc
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.utils import escalate_match_abort, is_match_abort
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
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

    def test_a_marked_abort_from_suffixes_escapes_matching(self) -> None:
        """Function-local group: a module-level one raising from suffixes() would break every other test's discovery."""

        def suffixes(cls: Any) -> tuple[str, ...]:
            raise escalate_match_abort(NotImplementedError("toy doc marked abort"))

        group: Any = type("ToyMarkedAbortSuffixDocFG", (ReadDocumentFG,), {"suffixes": classmethod(suffixes)})
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
