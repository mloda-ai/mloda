"""Pins that the explicit-``file_paths`` branch of ``ConcatenatedFileContent`` honours its declarations.

``input_features`` has two source branches, and today they disagree:

* the ``target_folder`` branch filters through ``find_file_paths(..., not_allowed_files_names=...)``, so
  the declared ``disallowed_files`` option really removes files;
* the ``file_paths`` branch contains a duplicated ``if options.get("file_paths"):`` check whose body
  builds a ``new_file_paths`` list, applies the ``disallowed_files`` filter and strips newlines out of
  each entry, and then never assigns it back. Every file survives and every newline stays.

``disallowed_files`` is a declared option with a declared default of ``("__init__.py",)``. A declaration
that one code path silently ignores is exactly the untruthful-metadata problem this work is fixing, so
the intended behavior is pinned here: the two branches must agree, and the newline stripping the dead
body performs must actually happen. ruff cannot see the defect because ``new_file_paths.append(...)``
counts as a use of the list.

The reads are observed the same way as in the sibling declaration tests: call ``_create_join_class``
first, then read ``DefaultOptionKeys.in_features`` off the produced feature and inspect each
``SourceTuple``.

Filtering that removes EVERY named file is the sharp edge of the fix and is pinned here too: an
empty result must be a loud error, not a join child whose ``in_features`` is an empty frozenset.

Subclass-leak policy: this module defines no ``FeatureGroup`` or ``BaseInputData`` subclass at all,
so it has nothing to leak that way; ``_create_join_class`` still registers
``ConcatenatedFileContent.join_feature_name`` in ``DynamicFeatureGroupCreator._created_classes``, a
permanent strong reference no ``gc.collect()`` reclaims, so the autouse fixture below pops it back out.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.provider import DefaultOptionKeys
from mloda.user import FeatureName, Options
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.experimental.source_input_feature import SourceTuple
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader


PROBE_FEATURE = FeatureName("rcff_file_paths_probe")


@pytest.fixture(autouse=True)
def _cleanup_dynamic_feature_groups() -> Iterator[None]:
    yield
    DynamicFeatureGroupCreator._created_classes.pop(ConcatenatedFileContent.join_feature_name, None)


def _source_tuples(instance: ConcatenatedFileContent, options: Options) -> set[SourceTuple]:
    """Every SourceTuple ``input_features`` produced for ``options``."""
    instance._create_join_class(instance.join_feature_name)
    features = instance.input_features(options, PROBE_FEATURE)
    assert features is not None
    feature = next(iter(features))
    source_tuples = feature.options.get(DefaultOptionKeys.in_features)
    assert source_tuples is not None
    return set(source_tuples)


def _names(instance: ConcatenatedFileContent, options: Options) -> set[str]:
    """The SourceTuple feature names, which are the short file names."""
    return {source_tuple.feature_name for source_tuple in _source_tuples(instance, options)}


def _values(instance: ConcatenatedFileContent, options: Options) -> set[str | None]:
    """The SourceTuple source values, which are the file paths handed to the reader."""
    return {source_tuple.source_value for source_tuple in _source_tuples(instance, options)}


def _write(tmp_path: Path, *names: str) -> None:
    """Create the named python files so the fixtures mirror a real directory."""
    for name in names:
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")


def _options(tmp_path: Path, file_names: list[str], **extra: object) -> Options:
    """Options selecting the given files explicitly, i.e. driving the ``file_paths`` branch."""
    return Options(
        {
            "file_paths": [str(tmp_path / name) for name in file_names],
            "document_reader_class": PyFileReader.get_class_name(),
            **extra,
        }
    )


class TestDisallowedFilesAppliesToExplicitFilePaths:
    """``disallowed_files`` must remove files on the ``file_paths`` branch, not only under ``target_folder``."""

    def test_declared_default_excludes_dunder_init(self, tmp_path: Path) -> None:
        """With no ``disallowed_files`` option, the declared ``("__init__.py",)`` default still applies."""
        _write(tmp_path, "a.py", "__init__.py")
        options = _options(tmp_path, ["a.py", "__init__.py"])

        instance = ConcatenatedFileContent()

        assert _names(instance, options) == {"a.py"}
        assert _values(instance, options) == {str(tmp_path / "a.py")}

    def test_explicit_disallowed_files_excludes_the_named_file(self, tmp_path: Path) -> None:
        """An explicitly passed ``disallowed_files`` removes exactly the files it names."""
        _write(tmp_path, "keep.py", "skip.py")
        options = _options(tmp_path, ["keep.py", "skip.py"], disallowed_files=("skip.py",))

        instance = ConcatenatedFileContent()

        assert _names(instance, options) == {"keep.py"}
        assert _values(instance, options) == {str(tmp_path / "keep.py")}

    def test_both_source_branches_exclude_the_same_files(self, tmp_path: Path) -> None:
        """Naming the files explicitly must select the same set as scanning the folder that holds them."""
        _write(tmp_path, "a.py", "b.py", "__init__.py")
        explicit = _options(tmp_path, ["a.py", "b.py", "__init__.py"])
        scanned = Options(
            {
                "target_folder": [str(tmp_path)],
                "document_reader_class": PyFileReader.get_class_name(),
            }
        )

        instance = ConcatenatedFileContent()

        assert _names(instance, explicit) == _names(instance, scanned)

    def test_control_nothing_is_dropped_when_nothing_is_disallowed(self, tmp_path: Path) -> None:
        """Control: an empty ``disallowed_files`` keeps every named file, so the filter cannot over-reach."""
        _write(tmp_path, "a.py", "__init__.py")
        options = _options(tmp_path, ["a.py", "__init__.py"], disallowed_files=())

        instance = ConcatenatedFileContent()

        assert _names(instance, options) == {"a.py", "__init__.py"}


class TestFilteringEveryFileOutIsALoudError:
    """An emptied ``file_paths`` list must raise where the option is read, not produce an empty join.

    RED: the filter currently leaves ``file_paths`` empty, still creates the join child, and stores an
    empty ``in_features`` frozenset. The uncomputable join then fails far away from the cause
    (``AttributeError: 'NoneType' object has no attribute 'columns'`` in ``calculate_feature``, plus a
    ``FeatureResolutionError`` for the join feature name). On ``origin/main`` this same input returned
    the file, so this is a regression the filtering fix introduced. The ``target_folder`` branch already
    behaves correctly here: ``find_file_paths`` raises ``No files found in the root directory``.
    """

    def test_the_declared_default_filtering_out_the_only_file_raises(self, tmp_path: Path) -> None:
        """The reviewer's reproduction: one explicit path, removed by the declared default."""
        _write(tmp_path, "__init__.py")
        options = _options(tmp_path, ["__init__.py"])

        instance = ConcatenatedFileContent()
        instance._create_join_class(instance.join_feature_name)

        with pytest.raises(ValueError) as exc_info:
            instance.input_features(options, PROBE_FEATURE)

        message = str(exc_info.value)
        assert "file_paths" in message
        assert "disallowed_files" in message
        assert ConcatenatedFileContent.get_class_name() in message

    def test_an_explicit_disallowed_files_emptying_the_list_raises(self, tmp_path: Path) -> None:
        """The same error for an explicitly named exclusion: the emptied result is what matters."""
        _write(tmp_path, "skip.py")
        options = _options(tmp_path, ["skip.py"], disallowed_files=("skip.py",))

        instance = ConcatenatedFileContent()
        instance._create_join_class(instance.join_feature_name)

        with pytest.raises(ValueError, match="file_paths"):
            instance.input_features(options, PROBE_FEATURE)

    def test_both_source_branches_raise_when_nothing_survives_the_filter(self, tmp_path: Path) -> None:
        """The two branches must agree on the empty case as they now do on the non-empty one.

        Scanning the folder already raises through ``find_file_paths``; naming the same file
        explicitly must not instead hand back a join child with an empty ``in_features`` frozenset.
        """
        _write(tmp_path, "__init__.py")
        explicit = _options(tmp_path, ["__init__.py"])
        scanned = Options(
            {
                "target_folder": [str(tmp_path)],
                "document_reader_class": PyFileReader.get_class_name(),
            }
        )

        instance = ConcatenatedFileContent()
        instance._create_join_class(instance.join_feature_name)

        with pytest.raises(ValueError):
            instance.input_features(scanned, PROBE_FEATURE)
        with pytest.raises(ValueError):
            instance.input_features(explicit, PROBE_FEATURE)

    def test_one_survivor_is_enough_to_keep_going(self, tmp_path: Path) -> None:
        """Control: partial filtering is normal and must not raise."""
        _write(tmp_path, "a.py", "__init__.py")
        options = _options(tmp_path, ["a.py", "__init__.py"])

        instance = ConcatenatedFileContent()

        assert _names(instance, options) == {"a.py"}

    def test_an_empty_file_paths_option_still_falls_through_to_target_folder(self) -> None:
        """An empty ``file_paths`` is not an emptied one: it selects the other source branch."""
        options = Options(
            {
                "file_paths": [],
                "document_reader_class": PyFileReader.get_class_name(),
            }
        )

        instance = ConcatenatedFileContent()

        with pytest.raises(ValueError, match="target_folder"):
            instance.input_features(options, PROBE_FEATURE)


class TestEmbeddedNewlinesAreStrippedFromExplicitFilePaths:
    """A ``file_paths`` entry read from a line-oriented list must not carry its newline into the reader."""

    @pytest.mark.parametrize("newline_position", ["trailing", "interior"])
    def test_newline_is_stripped_from_the_source_value(self, tmp_path: Path, newline_position: str) -> None:
        """The source value is the clean path, so the reader receives an openable file name."""
        _write(tmp_path, "a.py")
        clean = str(tmp_path / "a.py")
        split = len(str(tmp_path))
        raw = f"{clean}\n" if newline_position == "trailing" else f"{clean[:split]}\n{clean[split:]}"
        options = Options(
            {
                "file_paths": [raw],
                "document_reader_class": PyFileReader.get_class_name(),
            }
        )

        instance = ConcatenatedFileContent()

        assert _values(instance, options) == {clean}

    def test_newline_does_not_leak_into_the_feature_name(self, tmp_path: Path) -> None:
        """The short name drives merge indexes and links, so it must not end in a newline either."""
        _write(tmp_path, "a.py")
        options = Options(
            {
                "file_paths": [f"{tmp_path / 'a.py'}\n"],
                "document_reader_class": PyFileReader.get_class_name(),
            }
        )

        instance = ConcatenatedFileContent()

        assert _names(instance, options) == {"a.py"}
