"""Shared contract tests for ReadFileFG implementations (one file format per group).

A concrete test class sets the FormatFeatureGroup attributes plus ``write_file``. Not collected on its own.
Every test-local subclass of the group is gated by an explicit plugin mapping.
"""

from __future__ import annotations

import gc
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.provider import CHAIN_SEPARATOR, FormatFeatureGroup
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup
from tests.mixins.reader_feature_groups.format_feature_group_test_mixin import FormatFeatureGroupTestMixin

if TYPE_CHECKING:
    from mloda.provider import ReadFileFG

FOREIGN_SUFFIX = ".notmyformat"
HANDLE_OPTION = "data_access_handle"


class FileFormatFeatureGroupTestMixin(FormatFeatureGroupTestMixin):
    """Contract of every ReadFileFG; adds file, folder, pin, handle, cache and failure checks."""

    feature_group_class: type[ReadFileFG]
    tmp_path: Path

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        """Write a readable file of the group's format holding exactly these columns."""
        raise NotImplementedError

    def write_corrupt_file(self, path: Path) -> None:
        """Write a file with the group's suffix that its parser rejects (default: invalid bytes)."""
        path.write_bytes(b"\xff\xfe\x00\x81" * 64)

    @pytest.fixture(autouse=True)
    def _file_format_setup(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.own_path = self._make("own_file", {self.present_column: [1, 2]})
        self.foreign_path = tmp_path / "foreign_file{}".format(FOREIGN_SUFFIX)
        self.foreign_path.write_text(f"{self.present_column}\n1\n2\n")
        self.expected_source = os.path.abspath(self.own_path)

    def own_pointer(self) -> Any:
        return str(self.own_path)

    def other_pointer(self) -> tuple[Any, str]:
        other = self._make("other", {self.present_column: [9]})
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

    def _make(self, name: str, columns: dict[str, list[Any]], folder: Path | None = None) -> Path:
        parent = folder if folder is not None else self.tmp_path
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"{name}{self._suffix()}"
        self.write_file(path, columns)
        return path

    def _spy_reads(self, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
        """(full listing paths, sample listing paths) recorded from now on; the real listings still run."""
        cls = self.feature_group_class
        names: list[str] = []
        samples: list[str] = []
        original_names = cls.column_names
        original_sample = cls.sample_column_names

        def column_names(klass: Any, path: str) -> Any:
            names.append(str(path))
            return original_names(path)

        def sample_column_names(klass: Any, path: str) -> Any:
            samples.append(str(path))
            return original_sample(path)

        monkeypatch.setattr(cls, "column_names", classmethod(column_names))
        monkeypatch.setattr(cls, "sample_column_names", classmethod(sample_column_names))
        return names, samples

    def _make_ambiguous_folder(self, name: str = "ambig") -> Path:
        folder = self.tmp_path / name
        self._make("one", {self.present_column: [1]}, folder)
        self._make("two", {self.present_column: [2]}, folder)
        return folder

    # claiming

    def test_file_claims_via_file_handle_with_an_absolute_source(self) -> None:
        feature = Feature(self.present_column)
        assert self._claims(feature, self.own_dac())
        match = self._claimed_match(feature)
        assert match.source == os.path.abspath(self.own_path)
        assert match.access == str(self.own_path)

    def test_file_claims_via_a_folder_entry(self) -> None:
        folder = self.tmp_path / "claim_folder"
        path = self._make("entry", {self.present_column: [1]}, folder)
        feature = Feature(self.present_column)
        assert self._claims(feature, DataAccessCollection(folders={"folder_handle": str(folder)}))
        assert self._claimed_match(feature).source == os.path.abspath(path)

    def test_file_claims_via_a_pointer_to_a_file_as_str_and_path(self) -> None:
        for value in (str(self.own_path), self.own_path):
            feature = Feature(self.present_column, Options({self._group_name(): value}))
            assert self._claims(feature, None)
            assert self._claimed_match(feature).source == os.path.abspath(self.own_path)

    def test_file_claims_via_a_pointer_to_a_folder(self) -> None:
        folder = self.tmp_path / "pointer_folder"
        path = self._make("entry", {self.present_column: [1]}, folder)
        feature = Feature(self.present_column, Options({self._group_name(): str(folder)}))
        assert self._claims(feature, None)
        assert self._claimed_match(feature).source == os.path.abspath(path)

    # missing columns and ambiguity

    def test_file_folder_with_two_fitting_files_aborts_naming_both_and_the_fix(self) -> None:
        folder = self._make_ambiguous_folder()
        dac = DataAccessCollection(
            folders={"ambig_folder": str(folder)},
            credentials={"secret_handle": {"password": "toyfmt-secret-value"}},  # nosec B105
        )
        with pytest.raises(ValueError) as exc_info:
            self._resolve(Feature(self.present_column), dac)
        message = str(exc_info.value)
        assert os.path.abspath(folder / f"one{self._suffix()}") in message
        assert os.path.abspath(folder / f"two{self._suffix()}") in message
        assert "column_to_file" in message
        assert self._group_name() in message
        assert "narrow with a data_access_handle or column_to_file." not in message
        assert "data_access_handle" not in message  # a folder handle still holds both files, so it cannot pick one
        assert "file handles" not in message
        assert "toyfmt-secret-value" not in message

    def test_file_folder_with_one_fitting_and_one_other_file_resolves(self) -> None:
        folder = self.tmp_path / "mixed_folder"
        fitting = self._make("fits", {self.present_column: [1]}, folder)
        self._make("other", {"toyfmt_unrelated_column": [1]}, folder)
        feature = Feature(self.present_column)
        assert self._claims(feature, DataAccessCollection(folders={"mixed_handle": str(folder)}))
        assert self._claimed_match(feature).source == os.path.abspath(fitting)

    def test_file_a_nonexistent_folder_handle_declines_without_raising(self) -> None:
        dac = DataAccessCollection(folders={"gone_dir": str(self.tmp_path / "no_such_folder")})
        assert not self._claims(Feature(self.present_column), dac)

    def test_file_same_file_as_handle_and_in_a_folder_is_one_source(self) -> None:
        folder = self.tmp_path / "dup_folder"
        path = self._make("dup", {self.present_column: [1]}, folder)
        dac = DataAccessCollection(files={"dup_file": str(path)}, folders={"dup_dir": str(folder)})
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == os.path.abspath(path)

    def test_file_relative_file_handle_and_absolute_folder_are_not_ambiguous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = self.tmp_path / "rel_folder"
        path = self._make("rel", {self.present_column: [1]}, folder)
        monkeypatch.chdir(self.tmp_path)
        relative = os.path.relpath(path, self.tmp_path)
        assert not os.path.isabs(relative)
        dac = DataAccessCollection(files={"rel_file": relative}, folders={"rel_dir": str(folder)})
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == os.path.abspath(path)

    # column_to_file pins

    def test_file_column_to_file_pin_selects_only_that_file(self) -> None:
        other = self._make("pin_other", {self.present_column: [2]})
        dac = DataAccessCollection(
            files={"pin_a": str(self.own_path), "pin_b": str(other)}, column_to_file={self.present_column: "pin_b"}
        )
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == os.path.abspath(other)

    def test_file_column_to_file_pin_to_a_file_without_the_column_aborts(self) -> None:
        lacking = self._make("pin_lacking", {"toyfmt_unrelated_column": [1]})
        dac = DataAccessCollection(
            files={"pin_a": str(self.own_path), "pin_b": str(lacking)}, column_to_file={self.present_column: "pin_b"}
        )
        with pytest.raises(ValueError) as exc_info:
            self._resolve(Feature(self.present_column), dac)
        assert os.path.abspath(lacking) in str(exc_info.value)

    def test_file_column_to_file_pin_to_a_foreign_suffix_declines_without_reading_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names, samples = self._spy_reads(monkeypatch)
        dac = DataAccessCollection(
            files={"pin_foreign": str(self.foreign_path)}, column_to_file={self.present_column: "pin_foreign"}
        )
        assert not self._claims(Feature(self.present_column), dac)
        assert not [p for p in names + samples if p.endswith(FOREIGN_SUFFIX)]

    def test_file_column_to_file_pin_to_a_foreign_suffix_never_falls_back_to_an_unpinned_own_file(self) -> None:
        dac = DataAccessCollection(
            files={"pin_own": str(self.own_path), "pin_foreign": str(self.foreign_path)},
            column_to_file={self.present_column: "pin_foreign"},
        )
        assert not self._claims(Feature(self.present_column), dac)

    def test_file_column_to_file_pin_wins_over_a_data_access_handle_pointing_elsewhere(self) -> None:
        dac = DataAccessCollection(
            files={"pin_foreign": str(self.foreign_path), "pin_own": str(self.own_path)},
            column_to_file={self.present_column: "pin_foreign"},
        )
        feature = Feature(self.present_column, Options(context={HANDLE_OPTION: "pin_own"}))
        assert not self._claims(feature, dac)

    def test_file_column_to_file_pin_to_a_corrupt_file_aborts_despite_a_good_sibling(self) -> None:
        corrupt = self.tmp_path / f"pin_corrupt{self._suffix()}"
        self.write_corrupt_file(corrupt)
        dac = DataAccessCollection(
            files={"pin_corrupt": str(corrupt), "pin_sibling": str(self.own_path)},
            column_to_file={self.present_column: "pin_corrupt"},
        )
        with pytest.raises(ValueError) as exc_info:
            self._resolve(Feature(self.present_column), dac)
        assert os.path.abspath(corrupt) in str(exc_info.value)
        assert "could not read its columns" in str(exc_info.value)

    def test_file_pinned_chain_shaped_real_column_resolves_while_an_unpinned_chain_shaped_name_declines(self) -> None:
        name = f"toyfmt_pinchain{CHAIN_SEPARATOR}real"
        path = self._make("pin_chain", {name: [1], self.present_column: [2]})
        pinned = DataAccessCollection(files={"pin_chain": str(path)}, column_to_file={name: "pin_chain"})
        feature = Feature(name)
        assert self._claims(feature, pinned)
        assert self._claimed_match(feature).source == os.path.abspath(path)
        assert not self._claims(Feature(f"toyfmt_pinother{CHAIN_SEPARATOR}missing"), pinned)

    # data_access_handle

    def test_file_handle_names_one_file(self) -> None:
        other = self._make("handle_other", {self.present_column: [2]})
        dac = DataAccessCollection(files={"hand_a": str(self.own_path), "hand_b": str(other)})
        feature = Feature(self.present_column, Options(context={HANDLE_OPTION: "hand_b"}))
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == os.path.abspath(other)

    def test_file_handle_names_one_folder(self) -> None:
        first = self._make("entry", {self.present_column: [1]}, self.tmp_path / "dir_one")
        second = self._make("entry", {self.present_column: [2]}, self.tmp_path / "dir_two")
        dac = DataAccessCollection(folders={"dir_a": str(first.parent), "dir_b": str(second.parent)})
        feature = Feature(self.present_column, Options(context={HANDLE_OPTION: "dir_b"}))
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == os.path.abspath(second)

    def test_file_a_non_str_handle_is_no_narrowing_of_the_file_sources(self) -> None:
        cls = self.feature_group_class
        route = ClaimRoute("file", NamePolicy.CHECKED, True)
        dac = self.own_dac()
        plain = cls.find_sources(route, self.present_column, Options(), dac)
        listed = cls.find_sources(route, self.present_column, Options(context={HANDLE_OPTION: ["own_handle"]}), dac)
        assert plain
        assert [m.source for m in listed] == [m.source for m in plain]

    # document_suffixes and foreign files

    def test_file_document_suffixes_with_the_own_suffix_declines_every_source_kind(self) -> None:
        folder = self.tmp_path / "doc_folder"
        self._make("entry", {self.present_column: [1]}, folder)
        suffixes = frozenset({self._suffix()})
        options = Options({self._group_name(): str(self.own_path)}, context={"document_suffixes": suffixes})
        assert not self._claims(Feature(self.present_column, options), None)
        for dac in (self.own_dac(), DataAccessCollection(folders={"doc_dir": str(folder)})):
            feature = Feature(self.present_column, Options(context={"document_suffixes": suffixes}))
            assert not self._claims(feature, dac)

    def test_file_foreign_suffix_files_are_never_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names, samples = self._spy_reads(monkeypatch)
        folder = self.tmp_path / "foreign_folder"
        folder.mkdir()
        (folder / f"decoy{FOREIGN_SUFFIX}").write_text(f"{self.present_column}\n1\n")
        dac = DataAccessCollection(
            files={"own_handle": str(self.own_path), "foreign_handle": str(self.foreign_path)},
            folders={"foreign_dir": str(folder)},
        )
        assert self._claims(Feature(self.present_column), dac)
        assert not [p for p in names + samples if p.endswith(FOREIGN_SUFFIX)]

    # per-run cache

    def test_file_one_run_lists_the_file_at_most_once_per_listing_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        columns = [f"{self.present_column}_cache_{n}" for n in ("a", "b", "c")]
        path = self._make("cache_file", {name: [1, 2, 3] for name in columns})
        names, samples = self._spy_reads(monkeypatch)

        mloda.run_all(
            [*columns, f"{columns[0]}{CHAIN_SEPARATOR}sum_aggr"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {self.feature_group_class, PyArrowAggregatedFeatureGroup}
            ),
            data_access_collection=DataAccessCollection(files={"cache_handle": str(path)}),
        )

        mine = str(path)
        assert names.count(mine) <= 1
        assert samples.count(mine) <= 1
        assert names.count(mine) + samples.count(mine) >= 1

    def test_file_cached_listing_follows_size_and_mtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cls = self.feature_group_class
        path = self._make("stat_file", {"toyfmt_stat_a": [1]})
        match = SourceMatch(source=os.path.abspath(path), access=str(path))
        names, _ = self._spy_reads(monkeypatch)

        with run_match_cache():
            assert cls.columns(match) is not None
            assert cls.columns(match) is not None
            assert len(names) == 1

            self.write_file(path, {"toyfmt_stat_a": [1], "toyfmt_stat_longer_b": [2]})
            listing = cls.columns(match)
            assert listing is not None
            assert "toyfmt_stat_longer_b" in listing
            assert len(names) == 2

            stat_result = os.stat(path)
            os.utime(path, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 10**9))
            assert cls.columns(match) is not None
            assert len(names) == 3

    # listing failures

    def test_file_unreadable_file_listing_is_none_with_a_rejection_naming_the_path(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        path = self.tmp_path / f"corrupt_direct{self._suffix()}"
        self.write_corrupt_file(path)
        match = SourceMatch(source=os.path.abspath(path), access=str(path))
        assert self.feature_group_class.columns(match) is None
        assert os.path.abspath(path) in rejection_window[self._group_name()].reason

    def test_file_non_regular_file_is_never_opened(
        self, monkeypatch: pytest.MonkeyPatch, rejection_window: dict[str, MatchRejection]
    ) -> None:
        names, samples = self._spy_reads(monkeypatch)
        directory = self.tmp_path / f"not_a_file{self._suffix()}"
        directory.mkdir()
        assert not stat.S_ISREG(os.stat(directory).st_mode)
        match = SourceMatch(source=os.path.abspath(directory), access=str(directory))
        assert self.feature_group_class.columns(match) is None
        assert not names
        assert os.path.abspath(directory) in rejection_window[self._group_name()].reason

    @pytest.mark.parametrize("error", [OSError("toyfmt os failure"), ValueError("toyfmt value failure")])
    def test_file_read_failures_are_none_with_a_rejection(
        self, error: Exception, monkeypatch: pytest.MonkeyPatch, rejection_window: dict[str, MatchRejection]
    ) -> None:
        def failing(klass: Any, path: str) -> Any:
            raise error

        monkeypatch.setattr(self.feature_group_class, "column_names", classmethod(failing))
        match = SourceMatch(source=os.path.abspath(self.own_path), access=str(self.own_path))
        assert self.feature_group_class.columns(match) is None
        assert str(error) in rejection_window[self._group_name()].reason

    def test_file_missing_backend_listing_is_none_with_a_rejection(
        self, monkeypatch: pytest.MonkeyPatch, rejection_window: dict[str, MatchRejection]
    ) -> None:
        def missing(klass: Any, path: str) -> Any:
            raise ImportError("toyfmt backend is not installed")

        monkeypatch.setattr(self.feature_group_class, "column_names", classmethod(missing))
        match = SourceMatch(source=os.path.abspath(self.own_path), access=str(self.own_path))
        assert self.feature_group_class.columns(match) is None
        assert "toyfmt backend is not installed" in rejection_window[self._group_name()].reason

    def test_file_a_type_error_from_column_names_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def defect(klass: Any, path: str) -> Any:
            raise TypeError("toyfmt code defect")

        monkeypatch.setattr(self.feature_group_class, "column_names", classmethod(defect))
        match = SourceMatch(source=os.path.abspath(self.own_path), access=str(self.own_path))
        with pytest.raises(TypeError, match="toyfmt code defect"):
            self.feature_group_class.columns(match)

    def test_file_a_marked_match_abort_from_column_names_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def abort(klass: Any, path: str) -> Any:
            raise escalate_match_abort(ValueError("toyfmt marked abort"))

        monkeypatch.setattr(self.feature_group_class, "column_names", classmethod(abort))
        match = SourceMatch(source=os.path.abspath(self.own_path), access=str(self.own_path))
        with pytest.raises(ValueError, match="toyfmt marked abort"):
            self.feature_group_class.columns(match)

    def test_file_corrupt_file_declines_unpointed_and_aborts_pointed(self) -> None:
        path = self.tmp_path / f"corrupt_run{self._suffix()}"
        self.write_corrupt_file(path)

        result = self._evaluate(Feature(self.present_column), DataAccessCollection(files={"corrupt": str(path)}))
        assert self.feature_group_class not in result.identified
        reason = result.eliminations[self.feature_group_class].reason
        assert os.path.abspath(path) in reason
        assert "could not read" in reason

        pointed = Feature(self.present_column, Options({self._group_name(): str(path)}))
        with pytest.raises(ValueError) as exc_info:
            self._resolve(pointed, None)
        assert os.path.abspath(path) in str(exc_info.value)
        assert "could not read its columns" in str(exc_info.value)

    def test_file_pointed_at_a_missing_file_aborts_with_the_listing_reason(self) -> None:
        missing = self.tmp_path / f"pointed_missing{self._suffix()}"
        pointed = Feature(self.present_column, Options({self._group_name(): str(missing)}))
        with pytest.raises(ValueError) as exc_info:
            self._resolve(pointed, None)
        message = str(exc_info.value)
        assert os.path.abspath(missing) in message
        assert "not a regular file" in message

    # pointer of the wrong type

    # loader versus a load_neutral override

    def _load_through_gated_subclass(self, loader_value: int | None, neutral_value: int) -> Any:
        """Run a gated subclass overriding load_neutral (and optionally registering its own loader) under PyArrowTable."""
        parent = self.feature_group_class
        column = self.present_column
        sub_name = f"{parent.__name__}ToyfmtLoaderProbe{neutral_value}"

        def gated(cls: Any, feature_name: Any, options: Options, data_access_collection: Any = None) -> bool:
            if sub_name not in options:
                return False
            return bool(
                getattr(super(cls, cls), "match_feature_group_criteria")(feature_name, options, data_access_collection)
            )

        def load_neutral(cls: Any, match: SourceMatch, features: Any) -> Any:
            return pa.table({column: [neutral_value]})

        sub = type(
            sub_name,
            (parent,),
            {"match_feature_group_criteria": classmethod(gated), "load_neutral": classmethod(load_neutral)},
        )
        if loader_value is not None:
            getattr(sub, "register_loader")(PyArrowTable, lambda match, features: pa.table({column: [loader_value]}))
        result = mloda.run_all(
            [Feature(column, Options({sub_name: str(self.own_path)}))],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({cast(type[FormatFeatureGroup], sub)}),
        )
        values = result[0].column(column).to_pylist()
        del sub, result, gated, load_neutral
        gc.collect()
        return values

    def test_file_a_load_neutral_override_is_used_under_pyarrow_table(self) -> None:
        assert self._load_through_gated_subclass(None, 7101) == [7101]

    def test_file_a_subclass_registering_its_own_loader_keeps_that_loader_over_its_load_neutral(self) -> None:
        assert self._load_through_gated_subclass(7202, 7201) == [7202]

    # names

    def test_file_a_nonexistent_file_handle_declines(self) -> None:
        dac = DataAccessCollection(files={"gone_file": str(self.tmp_path / f"no_such_file{self._suffix()}")})
        assert not self._claims(Feature(self.present_column), dac)

    # folders: FIFOs and symlinks

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs need POSIX")
    def test_file_a_fifo_in_a_folder_is_skipped_without_blocking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        names, samples = self._spy_reads(monkeypatch)
        folder = self.tmp_path / "fifo_folder"
        folder.mkdir()
        fifo = folder / f"a_fifo{self._suffix()}"
        os.mkfifo(fifo)
        regular = self._make("b_regular", {self.present_column: [1]}, folder)
        feature = Feature(self.present_column)
        assert self._claims(feature, DataAccessCollection(folders={"fifo_dir": str(folder)}))
        assert self._claimed_match(feature).source == os.path.abspath(regular)
        assert str(fifo) not in names + samples

    @pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="symlinks need POSIX")
    def test_file_a_symlink_to_a_regular_file_in_a_folder_is_read(self) -> None:
        real = self._make("symlink_target", {self.present_column: [1]})
        folder = self.tmp_path / "link_folder"
        folder.mkdir()
        link = folder / f"link{self._suffix()}"
        os.symlink(real, link)
        feature = Feature(self.present_column)
        assert self._claims(feature, DataAccessCollection(folders={"link_dir": str(folder)}))
        assert os.path.basename(self._claimed_match(feature).source) == link.name

    # subclass takeover
