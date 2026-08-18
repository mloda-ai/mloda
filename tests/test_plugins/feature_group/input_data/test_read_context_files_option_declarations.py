"""Pins ``ConcatenatedFileContent``'s five option keys on a real, ENFORCED ``PROPERTY_MAPPING``.

``input_features`` reads ``disallowed_files``, ``file_paths``, ``target_folder``, ``file_type`` and
``document_reader_class``, and hand-rolls both their defaults (``["__init__.py"]``, ``"py"``) and their
requiredness (two ``ValueError`` raises). Those are promises to users that no declaration carries.

What is pinned here:

* **Inventory**: the five keys are declared, and every declared value is a ``PropertySpec``.
* **Truthful defaults**: ``disallowed_files`` declares ``("__init__.py",)`` (a TUPLE, because with
  ``context=False`` the materialized default lands in GROUP options, which are hashed) and
  ``file_type`` declares ``"py"``. ``file_paths`` / ``target_folder`` declare ``default=None``
  (optional, applies no value), not ``NO_DEFAULT``. All five are group parameters: they change what
  gets read, so they must split feature groups.
* **Truthful requiredness, enforced**: ``target_folder`` is required exactly when ``file_paths`` is
  absent, ``document_reader_class`` always. ``FeatureGroup.__init_subclass__`` installs the
  ``required_when`` guard on the class's resolved matcher, so the declaration is enforced at
  match time even for this pattern-less feature group; the four matcher cases pin that.
* **Load-bearing defaults**: ``input_features`` must read through ``self.options_with_defaults(options)``.
  The engine calls ``input_features`` with the feature's DECLARED (pre-default) options
  (``mloda/core/core/engine.py``: ``declared_options = feature.options``), so a declared default only
  reaches the read site when the group opts in itself. ``TestDeclaredDefaultsAreLoadBearing`` proves
  the opt-in behaviorally: a subclass that declares ``file_type`` default ``"md"`` must read ``.md``
  files with no ``file_type`` option set.

The hand-rolled ``ValueError`` backstops in ``input_features`` are KEPT, so
``test_read_context_files.py::TestConcatenatedFileContentFormatAgnostic::test_missing_document_reader_class_raises_error``
stays green.

Subclass-leak policy: this module tolerates NO leak, unlike its reader-side siblings. Its throwaway
class is a ``FeatureGroup``, and a leaked feature group is reachable by resolution: it inherits a
class-name matcher and would compete for features in any test that enumerates feature groups. It is
therefore built inside a helper (never at module import time, so a missing ``PROPERTY_MAPPING`` fails
one test instead of the whole module), carries a distinctive ``Rcfd`` name, and the ONE test that
builds it requests the ``no_feature_group_registry_pollution`` fixture that forces the reclaiming GC
pass and asserts the class is gone. The fixture is deliberately not autouse: a failure anywhere else
in the module would otherwise pin the frame, keep the class alive, and fire a second, misleading
failure in teardown.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DefaultOptionKeys, PropertySpec, is_no_default
from mloda.user import FeatureName, Options
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader


# Every option key ``ConcatenatedFileContent.input_features`` reads.
DECLARED_KEYS = frozenset({"disallowed_files", "file_paths", "target_folder", "file_type", "document_reader_class"})

PROBE_FEATURE = FeatureName("rcfd_declaration_probe")


@pytest.fixture(autouse=True)
def _cleanup_dynamic_feature_groups() -> Iterator[None]:
    yield
    DynamicFeatureGroupCreator._created_classes.pop(ConcatenatedFileContent.join_feature_name, None)


@pytest.fixture
def no_feature_group_registry_pollution() -> Iterator[None]:
    """Guarantee the one test that builds a throwaway ConcatenatedFileContent subclass leaks nothing.

    A test-local FeatureGroup subclass sits in reference cycles, so it lingers in
    ``FeatureGroup.__subclasses__()`` until a GC cycle runs, where other tests enumerating feature
    groups trip over it. Forcing the collection here reclaims it and pins the no-pollution contract.

    Requested explicitly rather than autouse: only one test in this module builds a class, and an
    autouse teardown assertion turns any unrelated failure in the module into two failures, because
    pytest holds the failing frame alive and with it every class that frame references.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _declared_mapping() -> dict[str, PropertySpec]:
    """The declared ``PROPERTY_MAPPING``, asserted present so callers read a non-optional mapping."""
    mapping = ConcatenatedFileContent.PROPERTY_MAPPING
    assert mapping is not None, "ConcatenatedFileContent declares no PROPERTY_MAPPING"
    return mapping


def _spec(key: str) -> PropertySpec:
    """The ``PropertySpec`` ``ConcatenatedFileContent`` declares for ``key``."""
    mapping = _declared_mapping()
    assert key in mapping, f"ConcatenatedFileContent does not declare '{key}'; declared: {sorted(mapping)}"
    return mapping[key]


def _source_tuple_names(instance: ConcatenatedFileContent, options: Options) -> set[str]:
    """The SourceTuple feature names ``input_features`` produced, following the sibling test's pattern."""
    instance._create_join_class(instance.join_feature_name)
    features = instance.input_features(options, PROBE_FEATURE)
    assert features is not None
    feature = next(iter(features))
    source_tuples = feature.options.get(DefaultOptionKeys.in_features)
    return {source_tuple.feature_name for source_tuple in source_tuples}


def _source_tuple_values(instance: ConcatenatedFileContent, options: Options) -> set[str]:
    """The SourceTuple source values (file paths) ``input_features`` produced."""
    instance._create_join_class(instance.join_feature_name)
    features = instance.input_features(options, PROBE_FEATURE)
    assert features is not None
    feature = next(iter(features))
    source_tuples = feature.options.get(DefaultOptionKeys.in_features)
    return {source_tuple.source_value for source_tuple in source_tuples}


def _make_markdown_default_subclass() -> type[ConcatenatedFileContent]:
    """A throwaway subclass whose only change is a declared ``file_type`` default of ``"md"``.

    Built here rather than in the class body of this module, so a missing base ``PROPERTY_MAPPING``
    fails only the test that needs it instead of breaking module import.
    """
    inherited = _declared_mapping()

    class RcfdMarkdownDefaultProbeFeatureGroup(ConcatenatedFileContent):
        """Declares markdown as its file type; everything else is inherited."""

        PROPERTY_MAPPING = {
            **inherited,
            "file_type": PropertySpec("Markdown files by declaration.", default="md", context=False),
        }

    return RcfdMarkdownDefaultProbeFeatureGroup


class TestDeclarationInventory:
    """The five keys ``input_features`` reads are declared, and nothing else is."""

    def test_declares_exactly_the_five_keys_it_reads(self) -> None:
        """``declared_option_keys`` is the inventory of what ``input_features`` reads."""
        assert ConcatenatedFileContent.declared_option_keys() == DECLARED_KEYS

    def test_every_declared_value_is_a_property_spec(self) -> None:
        """A declaration is only enforceable when each value IS a ``PropertySpec``."""
        mapping = ConcatenatedFileContent.PROPERTY_MAPPING
        assert mapping is not None, "ConcatenatedFileContent declares no PROPERTY_MAPPING"
        non_specs = {key: type(value).__name__ for key, value in mapping.items() if not isinstance(value, PropertySpec)}
        assert non_specs == {}


class TestDeclaredDefaults:
    """The declared defaults are the real runtime fallbacks, not documentation."""

    def test_disallowed_files_declares_a_tuple_default(self) -> None:
        """``context=False`` puts the materialized default in GROUP options, which decide feature identity."""
        default = _spec("disallowed_files").default

        assert default == ("__init__.py",)
        assert isinstance(default, tuple), (
            "a list default would compare unequal to the tuple form while hashing the same, so a caller "
            f"passing the default explicitly would never merge with the materialized one; got {type(default).__name__}"
        )
        assert hash(default) is not None

    def test_a_list_default_would_hash_equal_yet_compare_unequal(self) -> None:
        """The real reason the declared default is a tuple, pinned on ``Options`` itself.

        Group hashing is NOT the problem: ``Options.__hash__`` runs ``_deep_hashable``, which normalizes
        a list to a tuple, so a list default hashes fine. The problem is that hashing and equality then
        disagree: the two forms land in the same hash bucket but compare unequal, so a caller who passes
        ``["__init__.py"]`` gets a feature that never merges with the materialized ``("__init__.py",)``.
        """
        as_declared = Options({"disallowed_files": ("__init__.py",)})
        as_list = Options({"disallowed_files": ["__init__.py"]})

        assert hash(as_declared) == hash(as_list), "group hashing normalizes lists, so hashing is not the issue"
        assert as_declared != as_list
        assert as_declared.group["disallowed_files"] != as_list.group["disallowed_files"]

    def test_file_type_declares_the_py_default(self) -> None:
        """``file_type`` falls back to ``"py"`` at the read site, so the spec declares it."""
        assert _spec("file_type").default == "py"

    @pytest.mark.parametrize("key", ["file_paths", "target_folder"])
    def test_optional_source_keys_declare_none_rather_than_no_default(self, key: str) -> None:
        """Either source key may be omitted, so each declares ``default=None`` (applies no value)."""
        default = _spec(key).default

        assert not is_no_default(default), f"'{key}' declares NO_DEFAULT, which makes it unconditionally required"
        assert default is None

    @pytest.mark.parametrize("key", sorted(DECLARED_KEYS))
    def test_every_key_is_a_group_parameter(self, key: str) -> None:
        """All five keys change WHAT gets read, so each must split feature groups (``context=False``)."""
        assert _spec(key).context is False


class TestDeclaredRequiredness:
    """Requiredness is declared as ``required_when`` predicates, and the installed guard enforces them."""

    def test_target_folder_is_required_exactly_when_file_paths_is_absent(self) -> None:
        """The predicate mirrors the read site's ``if options.get("file_paths")`` branch."""
        predicate = _spec("target_folder").required_when
        assert predicate is not None, "target_folder declares no required_when predicate"

        assert bool(predicate(Options({"file_paths": ["/rcfd/a.py"]}))) is False
        assert bool(predicate(Options({}))) is True

    def test_document_reader_class_is_always_required(self) -> None:
        """The read site raises whenever it is absent, so its predicate is always satisfied."""
        predicate = _spec("document_reader_class").required_when
        assert predicate is not None, "document_reader_class declares no required_when predicate"

        assert bool(predicate(Options({}))) is True
        assert bool(predicate(Options({"file_paths": ["/rcfd/a.py"], "target_folder": ["/rcfd"]}))) is True

    def test_matcher_accepts_file_paths_with_a_reader(self) -> None:
        """``file_paths`` satisfies the source requirement, so the group matches."""
        options = Options({"file_paths": ["/rcfd/a.py"], "document_reader_class": PyFileReader.get_class_name()})

        matched = ConcatenatedFileContent.match_feature_group_criteria(
            FeatureName(ConcatenatedFileContent.get_class_name()), options
        )

        assert matched is True

    def test_matcher_accepts_target_folder_with_a_reader(self) -> None:
        """``target_folder`` is the other way to satisfy the source requirement."""
        options = Options({"target_folder": ["/rcfd"], "document_reader_class": PyFileReader.get_class_name()})

        matched = ConcatenatedFileContent.match_feature_group_criteria(
            FeatureName(ConcatenatedFileContent.get_class_name()), options
        )

        assert matched is True

    def test_matcher_rejects_a_missing_document_reader_class(self) -> None:
        """Without the always-required reader the group is a non-match, not a late ValueError."""
        options = Options({"file_paths": ["/rcfd/a.py"]})

        matched = ConcatenatedFileContent.match_feature_group_criteria(
            FeatureName(ConcatenatedFileContent.get_class_name()), options
        )

        assert matched is False

    def test_matcher_rejects_neither_file_paths_nor_target_folder(self) -> None:
        """With no source at all the group is a non-match, not a late ValueError."""
        options = Options({"document_reader_class": PyFileReader.get_class_name()})

        matched = ConcatenatedFileContent.match_feature_group_criteria(
            FeatureName(ConcatenatedFileContent.get_class_name()), options
        )

        assert matched is False


class TestDeclaredDefaultsAreLoadBearing:
    """The declaration IS the mechanism at the read site, not a second copy of the constants."""

    def test_absent_defaults_materialize_into_the_group_namespace(self) -> None:
        """``options_with_defaults`` fills the two concrete defaults, and ``context=False`` puts them in group."""
        options = Options({"target_folder": ["/rcfd"], "document_reader_class": PyFileReader.get_class_name()})

        materialized = ConcatenatedFileContent.options_with_defaults(options)

        assert materialized.group.get("file_type") == "py"
        assert materialized.group.get("disallowed_files") == ("__init__.py",)
        assert "file_type" not in materialized.context
        assert "disallowed_files" not in materialized.context

    def test_present_values_are_never_overridden(self) -> None:
        """A user-set value wins over the declared default."""
        options = Options(
            {
                "target_folder": ["/rcfd"],
                "document_reader_class": PyFileReader.get_class_name(),
                "file_type": "md",
                "disallowed_files": ("a.py",),
            }
        )

        materialized = ConcatenatedFileContent.options_with_defaults(options)

        assert materialized.get("file_type") == "md"
        assert materialized.get("disallowed_files") == ("a.py",)

    def test_declared_file_type_default_drives_the_read(
        self, tmp_path: Path, no_feature_group_registry_pollution: None
    ) -> None:
        """A subclass declaring ``file_type="md"`` reads the ``.md`` files with no ``file_type`` option set.

        This is the whole point of the refactor: ``input_features`` must resolve ``file_type`` through
        ``self.options_with_defaults(options)``, so a subclass's declaration changes what is read. A
        hard-coded ``or "py"`` fallback ignores the declaration and picks up ``a.py`` instead.
        """
        (tmp_path / "a.py").write_text("# python\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("# markdown\n", encoding="utf-8")
        options = Options(
            {
                "target_folder": [str(tmp_path)],
                "document_reader_class": MarkdownDocumentReader.get_class_name(),
            }
        )

        instance = _make_markdown_default_subclass()()

        assert _source_tuple_names(instance, options) == {"b.md"}

    def test_an_explicit_empty_file_type_is_not_replaced_by_the_declared_default(self) -> None:
        """Presence, not truthiness: ``options_with_defaults`` keeps an explicit ``""``.

        The declared ``"py"`` default fills only an ABSENT key, so a caller who explicitly asks for an
        empty suffix gets exactly that. Pinned as the contrast case to the reader surface, where the
        ``or``-based fallback cannot tell an absent key from an explicit empty one.
        """
        options = Options(
            {
                "target_folder": ["/rcfd"],
                "document_reader_class": PyFileReader.get_class_name(),
                "file_type": "",
            }
        )

        materialized = ConcatenatedFileContent.options_with_defaults(options)

        assert materialized.get("file_type") == ""

    def test_an_explicit_empty_file_type_raises_the_no_files_found_error(self, tmp_path: Path) -> None:
        """An empty suffix matches nothing, and a zero-match scan is the existing loud ValueError.

        Chosen behavior: no new validation layer for ``file_type=""``. The honoured empty value reaches
        ``find_file_paths``, whose ``rglob("*.")`` matches nothing, and the existing
        "No files found in the root directory" raise names the folder that was scanned. That is already
        the clear, actionable failure this branch gives for any suffix with no matches, so an extra
        empty-string check would add surface without adding information.
        """
        (tmp_path / "a.py").write_text("# python\n", encoding="utf-8")
        options = Options(
            {
                "target_folder": [str(tmp_path)],
                "document_reader_class": PyFileReader.get_class_name(),
                "file_type": "",
            }
        )

        instance = ConcatenatedFileContent()
        instance._create_join_class(instance.join_feature_name)

        with pytest.raises(ValueError) as exc_info:
            instance.input_features(options, PROBE_FEATURE)

        message = str(exc_info.value)
        assert "No files found" in message
        assert str(tmp_path) in message

    def test_declared_disallowed_files_default_excludes_dunder_init(self, tmp_path: Path) -> None:
        """With no ``disallowed_files`` option the declared ``("__init__.py",)`` default excludes it.

        Behavior-preservation guard: the declaration must reproduce today's hand-rolled fallback exactly,
        so this stays green across the refactor.
        """
        (tmp_path / "a.py").write_text("# kept\n", encoding="utf-8")
        (tmp_path / "__init__.py").write_text("# skipped\n", encoding="utf-8")
        options = Options(
            {
                "target_folder": [str(tmp_path)],
                "document_reader_class": PyFileReader.get_class_name(),
            }
        )

        instance = ConcatenatedFileContent()

        assert _source_tuple_names(instance, options) == {"a.py"}
        assert _source_tuple_values(instance, options) == {str((tmp_path / "a.py").resolve())}
