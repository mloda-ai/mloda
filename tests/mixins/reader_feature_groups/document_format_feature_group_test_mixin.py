"""Shared contract tests for ReadDocumentFG implementations (one document format per group).

A concrete test class sets the class attributes below. Not collected on its own.
Every test-local subclass of the group is gated by an explicit pointer on its parent key.
"""

from __future__ import annotations

import gc
import itertools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import CHAIN_SEPARATOR, FormatFeatureGroup
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.mixins.reader_feature_groups.format_feature_group_test_mixin import (
    FormatFeatureGroupTestMixin,
    _LoadCapture,
)

if TYPE_CHECKING:
    from mloda.provider import ReadDocumentFG

FOREIGN_SUFFIX = ".notmyformat"
HANDLE_OPTION = "data_access_handle"
DOCUMENT_SUFFIXES_OPTION = "document_suffixes"
SOURCE = "~source"
FILE_TYPE = "~file_type"


class DocumentFormatFeatureGroupTestMixin(FormatFeatureGroupTestMixin):
    """Contract of every ReadDocumentFG: file, folder and pointer routes, exact names, the three outputs."""

    feature_group_class: type[ReadDocumentFG]
    expected_suffixes: frozenset[str]
    sample_text: str = "document body\n"
    expected_content: str | None = None
    handed_over: bool = False
    load_framework = PythonDictFramework
    load_loader_names = ("neutral", "PythonDictFramework")
    tmp_path: Path

    @pytest.fixture(autouse=True)
    def _document_setup(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.own_path = self._make("own_file")
        self.foreign_path = tmp_path / f"foreign_file{FOREIGN_SUFFIX}"
        self.foreign_path.write_text(self.sample_text, encoding="utf-8")
        self.expected_source = os.path.abspath(self.own_path)

    def own_pointer(self) -> Any:
        return str(self.own_path)

    def other_pointer(self) -> tuple[Any, str]:
        other = self._make("other")
        return str(other), os.path.abspath(other)

    def foreign_handle_dac(self) -> tuple[DataAccessCollection, str]:
        dac = DataAccessCollection(files={"own_handle": str(self.own_path)}, credentials={"cred_h": {"k": "v"}})
        return dac, "cred_h"

    def known_handles_dac(self) -> tuple[DataAccessCollection, list[str]]:
        dac = DataAccessCollection(
            files={"known_file_handle": str(self.own_path)}, folders={"known_folder_handle": str(self.tmp_path)}
        )
        return dac, ["known_file_handle", "known_folder_handle"]

    def own_dac(self) -> DataAccessCollection:
        return DataAccessCollection(files={"own_handle": str(self.own_path)})

    def foreign_dac(self) -> DataAccessCollection:
        return DataAccessCollection(files={"foreign_handle": str(self.foreign_path)})

    # helpers

    def _suffix(self) -> str:
        return self.feature_group_class.suffixes()[0]

    def _make(self, name: str, folder: Path | None = None, suffix: str | None = None) -> Path:
        parent = folder if folder is not None else self.tmp_path
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"{name}{self._suffix() if suffix is None else suffix}"
        path.write_text(self.sample_text, encoding="utf-8")
        return path

    def _declared(self) -> list[str]:
        name = self._group_name()
        return [name, f"{name}{SOURCE}", f"{name}{FILE_TYPE}"]

    def _content(self) -> str:
        return self.sample_text if self.expected_content is None else self.expected_content

    def _matched_source(self, name: str, dac: DataAccessCollection | None, group: dict[str, Any] | None = None) -> Any:
        feature = self._feature(name, group)
        assert self._claims(feature, dac)
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is self.feature_group_class
        return cast(SourceMatch, pair[1])

    def _run(
        self,
        names: list[str],
        dac: DataAccessCollection | None,
        group: dict[str, Any] | None = None,
        extender: Any = None,
    ) -> list[Any]:
        return cast(
            list[Any],
            mloda.run_all(
                [self._feature(name, group) for name in names],
                compute_frameworks=[PythonDictFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
                data_access_collection=dac,
                function_extender=None if extender is None else {extender},
            ),
        )

    def _file_type(self, suffix: str) -> str:
        return suffix.lstrip(".").lower()

    def _record(self, path: Path | str, suffix: str | None = None, content: str | None = None) -> dict[str, list[str]]:
        name = self._group_name()
        return {
            name: [self._content() if content is None else content],
            f"{name}{SOURCE}": [str(path)],
            f"{name}{FILE_TYPE}": [self._file_type(self._suffix() if suffix is None else suffix)],
        }

    def _make_ambiguous_folder(self, name: str = "ambig") -> Path:
        folder = self.tmp_path / name
        self._make("one", folder)
        self._make("two", folder)
        return folder

    # contract shape

    def test_doc_declares_searched_declared_file_and_folder_routes(self) -> None:
        assert self.feature_group_class.CLAIM_ROUTES == (
            ClaimRoute("file", NamePolicy.DECLARED, True),
            ClaimRoute("folder", NamePolicy.DECLARED, True),
        )

    def test_doc_declared_names_are_exactly_the_three_outputs(self) -> None:
        assert self.feature_group_class.declared_names() == frozenset(self._declared())

    def test_doc_owns_exactly_the_expected_suffixes(self) -> None:
        assert frozenset(self.feature_group_class.suffixes()) == self.expected_suffixes

    # claiming

    def test_doc_claims_every_declared_name_via_a_file_handle_keeping_the_path_as_given(self) -> None:
        for name in self._declared():
            match = self._matched_source(name, self.own_dac())
            assert match.source == os.path.abspath(self.own_path)
            assert match.access == str(self.own_path)

    def test_doc_claims_every_declared_name_via_a_folder_entry(self) -> None:
        folder = self.tmp_path / "claim_folder"
        path = self._make("entry", folder)
        self._make("entry_other", folder, FOREIGN_SUFFIX)
        for name in self._declared():
            match = self._matched_source(name, DataAccessCollection(folders={"folder_handle": str(folder)}))
            assert match.source == os.path.abspath(path)

    def test_doc_claims_via_a_pointer_to_a_file_as_str_and_path(self) -> None:
        for value in (str(self.own_path), self.own_path):
            for name in self._declared():
                match = self._matched_source(name, None, {self._group_name(): value})
                assert match.source == os.path.abspath(self.own_path)

    def test_doc_claims_via_a_pointer_to_a_folder(self) -> None:
        folder = self.tmp_path / "pointer_folder"
        path = self._make("entry", folder)
        match = self._matched_source(self._group_name(), None, {self._group_name(): str(folder)})
        assert match.source == os.path.abspath(path)

    def test_doc_a_pointer_to_a_foreign_suffix_never_claims(self) -> None:
        feature = self._feature(self._group_name(), {self._group_name(): str(self.foreign_path)})
        assert not self._claims(feature, self.own_dac())

    def test_doc_foreign_suffix_files_in_a_folder_never_claim(self) -> None:
        folder = self.tmp_path / "foreign_folder"
        self._make("decoy", folder, FOREIGN_SUFFIX)
        assert not self._claims(
            self._feature(self._group_name()), DataAccessCollection(folders={"foreign_dir": str(folder)})
        )

    def test_doc_a_nonexistent_folder_handle_declines_without_raising(self) -> None:
        dac = DataAccessCollection(folders={"gone_dir": str(self.tmp_path / "no_such_folder")})
        assert not self._claims(self._feature(self._group_name()), dac)

    def test_doc_a_missing_file_declines_unpointed_with_a_reason_and_aborts_when_pointed(self) -> None:
        gone = str(self.tmp_path / f"gone{self._suffix()}")
        name = self._group_name()
        for dac in (DataAccessCollection(files={"gone_file": gone}), None):
            group = None if dac is not None else {name: gone}
            result = self._evaluate(self._feature(name, group), dac)
            assert self.feature_group_class not in result.identified
            if dac is not None:
                assert "not a regular file" in result.eliminations[self.feature_group_class].reason
        with pytest.raises(ValueError, match="not a regular file"):
            self._resolve(self._feature(name, {name: gone}), None)

    def test_doc_a_pointer_to_a_handed_over_suffix_names_document_suffixes(self) -> None:
        sub = self._gated_subclass()
        sub.handover_suffixes = classmethod(lambda cls: cls.suffixes())
        mapping: FeatureGroupEnvironmentMapping = {cast(type[FormatFeatureGroup], sub): {PyArrowTable}}
        feature = Feature(self._group_name(), Options({self._group_name(): str(self.own_path)}))
        result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, None)
        assert sub not in result.identified
        assert DOCUMENT_SUFFIXES_OPTION in result.eliminations[sub].reason, result.eliminations[sub]
        del sub, mapping, result, feature
        gc.collect()

    def test_doc_a_folder_with_one_fitting_and_one_other_file_resolves(self) -> None:
        folder = self.tmp_path / "mixed_folder"
        fitting = self._make("fits", folder)
        self._make("other", folder, FOREIGN_SUFFIX)
        match = self._matched_source(self._group_name(), DataAccessCollection(folders={"mixed_handle": str(folder)}))
        assert match.source == os.path.abspath(fitting)

    def test_doc_same_file_as_handle_and_in_a_folder_is_one_source(self) -> None:
        folder = self.tmp_path / "dup_folder"
        path = self._make("dup", folder)
        dac = DataAccessCollection(files={"dup_file": str(path)}, folders={"dup_dir": str(folder)})
        assert self._matched_source(self._group_name(), dac).source == os.path.abspath(path)

    def test_doc_relative_file_handle_and_absolute_folder_are_not_ambiguous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = self.tmp_path / "rel_folder"
        path = self._make("rel", folder)
        monkeypatch.chdir(self.tmp_path)
        relative = os.path.relpath(path, self.tmp_path)
        assert not os.path.isabs(relative)
        dac = DataAccessCollection(files={"rel_file": relative}, folders={"rel_dir": str(folder)})
        assert self._matched_source(self._group_name(), dac).source == os.path.abspath(path)

    # one file per request

    def test_doc_folder_with_two_files_of_the_suffix_aborts_for_every_declared_name(self) -> None:
        folder = self._make_ambiguous_folder()
        dac = DataAccessCollection(
            folders={"ambig_folder": str(folder)},
            credentials={"secret_handle": {"password": "docfmt-secret-value"}},  # nosec B105
        )
        for name in self._declared():
            with pytest.raises(ValueError) as exc_info:
                self._resolve(self._feature(name), dac)
            message = str(exc_info.value)
            assert os.path.abspath(folder / f"one{self._suffix()}") in message
            assert os.path.abspath(folder / f"two{self._suffix()}") in message
            assert self._group_name() in message
            assert "docfmt-secret-value" not in message

    def test_doc_two_file_handles_of_the_suffix_abort_and_a_handle_resolves_it(self) -> None:
        other = self._make("handle_other")
        dac = DataAccessCollection(files={"hand_a": str(self.own_path), "hand_b": str(other)})
        with pytest.raises(ValueError) as exc_info:
            self._resolve(self._feature(self._group_name()), dac)
        assert os.path.abspath(other) in str(exc_info.value)
        feature = Feature(self._group_name(), Options(context={**self.context_options, HANDLE_OPTION: "hand_b"}))
        assert self._claims(feature, dac)
        assert cast(SourceMatch, feature.input_data_match[1]).source == os.path.abspath(other)  # type: ignore[index]

    def test_doc_a_column_to_file_pin_selects_only_that_file(self) -> None:
        other = self._make("pin_other")
        dac = DataAccessCollection(
            files={"pin_a": str(self.own_path), "pin_b": str(other)}, column_to_file={self._group_name(): "pin_b"}
        )
        assert self._matched_source(self._group_name(), dac).source == os.path.abspath(other)

    def test_doc_a_pin_to_a_foreign_suffix_never_falls_back_to_an_unpinned_own_file(self) -> None:
        dac = DataAccessCollection(
            files={"pin_own": str(self.own_path), "pin_foreign": str(self.foreign_path)},
            column_to_file={self._group_name(): "pin_foreign"},
        )
        assert not self._claims(self._feature(self._group_name()), dac)

    # data_access_handle

    def test_doc_handle_names_one_folder(self) -> None:
        first = self._make("entry", self.tmp_path / "dir_one")
        second = self._make("entry", self.tmp_path / "dir_two")
        dac = DataAccessCollection(folders={"dir_a": str(first.parent), "dir_b": str(second.parent)})
        feature = Feature(self._group_name(), Options(context={**self.context_options, HANDLE_OPTION: "dir_b"}))
        assert self._claims(feature, dac)
        assert cast(SourceMatch, feature.input_data_match[1]).source == os.path.abspath(second)  # type: ignore[index]

    # exact names

    def _undeclared_names(self) -> list[str]:
        name = self._group_name()
        return [
            f"{name}_anything",
            f"{name}__x",
            f"{name}{CHAIN_SEPARATOR}sum_aggr",
            f"{name}~other",
            f"{name}~0",
            f"{name}{SOURCE}~file_type",
            f"{name}{SOURCE}_x",
            f"prefix_{name}",
            name.lower(),
            "source",
            "file_type",
            self.missing_column,
        ]

    def test_doc_declines_every_undeclared_name_with_the_own_file_present(self) -> None:
        for name in self._undeclared_names():
            assert not self._claims(self._feature(name), self.own_dac()), name
            assert not self._claims(self._feature(name), DataAccessCollection(folders={"d": str(self.tmp_path)})), name

    def test_doc_declines_every_undeclared_name_when_pointed_and_records_why(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = self._options({self._group_name(): str(self.own_path)})
        for name in self._undeclared_names():
            rejection_window.clear()
            assert not self.feature_group_class.match_feature_group_criteria(name, options, None), name
            assert name in rejection_window[self._group_name()].reason

    def test_doc_the_prefix_shortcut_is_gone_for_a_name_extending_the_group_name(self) -> None:
        name = f"{self._group_name()}_extra"
        feature = self._feature(name, {self._group_name(): str(self.own_path)})
        assert not self._claims(feature, None)

    # the three outputs end to end

    def test_doc_all_three_outputs_end_to_end(self) -> None:
        result = self._run(self._declared(), self.own_dac())
        assert result == [self._record(self.own_path)]

    def test_doc_every_requested_subset_returns_exactly_those_columns(self) -> None:
        full = self._record(self.own_path)
        declared = self._declared()
        for size in (1, 2, 3):
            for subset in itertools.combinations(declared, size):
                result = self._run(list(subset), self.own_dac())
                assert result == [{name: full[name] for name in subset}], subset

    def test_doc_file_type_is_the_actual_lowercased_suffix_of_each_owned_suffix(self) -> None:
        for index, suffix in enumerate(sorted(self.expected_suffixes)):
            path = self._make(f"typed_{index}", self.tmp_path / f"typed_{index}", suffix)
            result = self._run([f"{self._group_name()}{FILE_TYPE}"], DataAccessCollection(files={"typed": str(path)}))
            assert result == [{f"{self._group_name()}{FILE_TYPE}": [self._file_type(suffix)]}], suffix

    def test_doc_source_is_the_path_as_given_while_the_load_identity_is_absolute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(self.tmp_path)
        relative = os.path.relpath(self.own_path, self.tmp_path)
        assert not os.path.isabs(relative)
        capture = _LoadCapture()
        dac = DataAccessCollection(files={"rel_handle": relative})
        result = self._run([f"{self._group_name()}{SOURCE}"], dac, extender=capture)
        assert result == [{f"{self._group_name()}{SOURCE}": [relative]}]
        assert [context.data_access_identity for context in capture.contexts] == [os.path.abspath(self.own_path)]

    def test_doc_a_pointed_run_loads_the_pointed_file(self) -> None:
        other = self._make("pointed_other")
        result = self._run(self._declared(), None, {self._group_name(): str(other)})
        assert result == [self._record(other)]

    # document_suffixes handover

    def test_doc_handover_default_matches_the_group_kind(self) -> None:
        plain = Feature(self._group_name())
        result = IdentifyFeatureGroupClass.evaluate(plain, self._plugins(), None, self.own_dac())
        assert (self.feature_group_class in result.identified) is (not self.handed_over)

    def test_doc_listing_its_own_suffixes_in_document_suffixes_claims(self) -> None:
        suffixes = frozenset(self.feature_group_class.suffixes())
        feature = Feature(self._group_name(), Options(context={DOCUMENT_SUFFIXES_OPTION: suffixes}))
        result = IdentifyFeatureGroupClass.evaluate(feature, self._plugins(), None, self.own_dac())
        assert self.feature_group_class in result.identified

    def test_doc_listing_an_unrelated_suffix_in_document_suffixes_changes_nothing(self) -> None:
        feature = Feature(self._group_name(), Options(context={DOCUMENT_SUFFIXES_OPTION: frozenset({".docfmtother"})}))
        result = IdentifyFeatureGroupClass.evaluate(feature, self._plugins(), None, self.own_dac())
        assert (self.feature_group_class in result.identified) is (not self.handed_over)

    # opt-outs of shared contract cases that need a column listing

    def test_pointed_with_a_missing_column_aborts_naming_source_and_columns(self) -> None:
        pytest.skip("document names are declared, so an undeclared name declines even when pointed")

    def test_unpointed_missing_column_declines_with_a_rejection(self) -> None:
        pytest.skip("document names are declared, so an undeclared name has no column listing to cite")

    def test_chain_and_column_separated_names_that_are_not_columns_decline(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        pytest.skip("undeclared document names decline without a column listing; the undeclared-name tests cover them")

    # subclass takeover

    def test_doc_a_subclass_takes_over_the_other_declared_names_keyed_on_its_parent(self) -> None:
        sub = self._gated_subclass()
        mapping: FeatureGroupEnvironmentMapping = {cast(type[FormatFeatureGroup], sub): {PyArrowTable}}
        for name in self._declared()[1:]:
            feature = self._feature(name, {self._group_name(): str(self.own_path)})
            result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, None)
            assert sub in result.identified, name
            del result, feature
        del sub, mapping
        gc.collect()

    def test_doc_a_subclass_loads_the_requested_columns_under_the_parent_names(self) -> None:
        sub = self._gated_subclass()
        name = self._group_name()
        pointer = {name: str(self.own_path)}
        for names in ([name], [f"{name}{SOURCE}"], self._declared()):
            result = mloda.run_all(
                [self._feature(requested, pointer) for requested in names],
                compute_frameworks=[PythonDictFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({cast(type[FormatFeatureGroup], sub)}),
            )
            full = self._record(self.own_path)
            assert result == [{requested: full[requested] for requested in names}], names
            del result
        del sub
        gc.collect()
