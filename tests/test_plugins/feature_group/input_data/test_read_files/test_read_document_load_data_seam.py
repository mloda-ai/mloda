"""Tests for the ReadDocument.load_data template-method lifecycle seam.

These pin the planned contract (mirroring the ReadDB seam from issue #535):

    ReadDocument gains
      - classmethod produce_document(cls, file_path: str) -> Any: the per-format
        parse hook; the base default raises NotImplementedError.
      - classmethod document_file_type(cls, file_path: str) -> str: default is
        cls.suffix()[0].lstrip(".").

    load_data becomes a template method that
      1. probes the parse hook FIRST: if produce_document is still the
         ReadDocument base default, raise NotImplementedError IMMEDIATELY,
         before touching features (so load_data(None, None) raises
         NotImplementedError, never AttributeError),
      2. file_path = features.get_options_key(cls.__name__),
      3. content = cls.produce_document(file_path),
      4. return [{cls.get_class_name(): content, "source": file_path,
                  "file_type": cls.document_file_type(file_path)}].

    is_final_reader() is decided STRUCTURALLY (no execution):
    True iff load_data is overridden wholesale relative to ReadDocument, or
    produce_document is overridden. ReadDocument itself and hook-less
    intermediate bases stay False.

Test isolation note:
All ReadDocument subclasses used by these tests are defined at MODULE scope,
never inside test methods. Function-local subclasses linger in
``ReadDocument.__subclasses__()`` (the global plugin registry) until GC runs,
leaking into sibling tests' plugin discovery, and they are unpicklable, which
breaks multiprocessing runners. Module-scope classes are picklable and stable.
Additionally, every seam reader uses a sentinel suffix in the ``.zzzdocseam*``
family that no real file in this repo or in any sibling test has, so a leaked
registry entry can never match a file in ``test_read_document.py`` or
``test_read_context_files.py``.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_files.json_document_reader import JsonDocumentReader
from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader, TextFileReader
from mloda_plugins.feature_group.input_data.read_files.yaml_document_reader import YamlDocumentReader


def _underlying(member: Any) -> Any:
    """Underlying function of a classmethod/staticmethod/plain override, for identity comparison."""
    return getattr(member, "__func__", member)


class MockFeatureSet:
    """Mock FeatureSet for driving document readers, mirroring test_markdown_document_reader.py."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def get_options_key(self, key: str) -> Any:
        return self._options.get(key)


# --------------------------------------------------------------------------------------
# Module-scope readers for the load_data lifecycle seam tests.
#
# Defined at MODULE scope (never inside a test) so they are picklable and stable in the
# plugin registry. Every suffix is a unique ``.zzzdocseam*`` sentinel that matches no
# real file anywhere, so a leaked class can never be selected by sibling tests' plugin
# discovery. The tests below drive these readers directly and record interactions via
# ClassVar lists.
# --------------------------------------------------------------------------------------


class _SeamIntermediateDoc(ReadDocument):
    """Intermediate base: overrides suffix() but NOT produce_document; not a final reader."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseamint",)


class _SeamHappyPathDoc(ReadDocument):
    """Final seam reader: overrides suffix + produce_document, records received file paths."""

    received_paths: ClassVar[list[str]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseam",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        cls.received_paths.append(file_path)
        with open(file_path, encoding="utf-8") as file:
            return "produced::" + file.read()


class _CustomFileTypeDoc(ReadDocument):
    """Final seam reader overriding document_file_type to use the ACTUAL path suffix."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseamcfta", ".zzzdocseamcftb")

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        with open(file_path, encoding="utf-8") as file:
            return file.read()

    @classmethod
    def document_file_type(cls, file_path: str) -> str:
        return Path(file_path).suffix.lstrip(".")


class _ProbeRecorderDoc(ReadDocument):
    """Final seam reader whose produce_document records calls; the probe must never invoke it."""

    produce_calls: ClassVar[list[str]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseamprobe",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        cls.produce_calls.append(file_path)
        return "recorded"


class _BoomProbeDoc(ReadDocument):
    """Final seam reader whose parse hook explodes; the probe must never invoke it."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseamboom",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        raise RuntimeError("boom")


class _WholesaleDoc(ReadDocument):
    """Legacy-style reader: overrides load_data wholesale, no produce_document; still final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseamwhole",)

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return [{"_WholesaleDoc": "wholesale", "source": "n/a", "file_type": "zzzdocseamwhole"}]


class TestReadDocumentLoadDataSeam:
    def test_abstract_read_document_probe_stays_false(self) -> None:
        """Abstract ReadDocument has no parse hook: load_data raises and probe is False."""
        with pytest.raises(NotImplementedError):
            ReadDocument.load_data(None, None)  # type: ignore[arg-type]

        assert ReadDocument.is_final_reader() is False

    def test_intermediate_base_probe_false_and_raises_before_features(self) -> None:
        """A base that overrides suffix() but NOT produce_document is not a final reader.

        The parse hook must be probed *before* features access, so
        load_data(None, None) raises NotImplementedError, not AttributeError.
        """
        assert _SeamIntermediateDoc.is_final_reader() is False

        with pytest.raises(NotImplementedError):
            _SeamIntermediateDoc.load_data(None, None)  # type: ignore[arg-type]

    def test_seam_happy_path_returns_envelope(self) -> None:
        """A reader overriding produce_document is final: load_data builds the one-row envelope.

        Fails today: ReadDocument.load_data unconditionally raises NotImplementedError,
        so the seam never runs and produce_document is never consulted.
        """
        _SeamHappyPathDoc.received_paths.clear()

        assert _SeamHappyPathDoc.is_final_reader() is True

    def test_seam_happy_path_load_data_envelope(self, tmp_path: Path) -> None:
        """load_data resolves the path from options, parses via the hook, and wraps the envelope."""
        _SeamHappyPathDoc.received_paths.clear()

        file_path = tmp_path / "doc.zzzdocseam"
        file_path.write_text("hello seam", encoding="utf-8")

        features = MockFeatureSet({"_SeamHappyPathDoc": str(file_path)})
        result = _SeamHappyPathDoc.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {
                "_SeamHappyPathDoc": "produced::hello seam",
                "source": str(file_path),
                "file_type": "zzzdocseam",
            }
        ]
        assert _SeamHappyPathDoc.received_paths == [str(file_path)]

    def test_document_file_type_default_is_first_suffix_without_dot(self) -> None:
        """The default document_file_type is suffix()[0].lstrip(".").

        Fails today: ReadDocument has no document_file_type classmethod at all.
        """
        assert _SeamHappyPathDoc.document_file_type("anything.zzzdocseam") == "zzzdocseam"

    def test_document_file_type_override_is_honored_in_envelope(self, tmp_path: Path) -> None:
        """An override using the actual path suffix (like YamlDocumentReader needs) reaches the envelope."""
        file_path = tmp_path / "doc.zzzdocseamcftb"
        file_path.write_text("second suffix", encoding="utf-8")

        features = MockFeatureSet({"_CustomFileTypeDoc": str(file_path)})
        result = _CustomFileTypeDoc.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {
                "_CustomFileTypeDoc": "second suffix",
                "source": str(file_path),
                "file_type": "zzzdocseamcftb",
            }
        ]


class TestIsFinalReaderIsSideEffectFree:
    """is_final_reader() must classify readers structurally, not by execution."""

    def test_probe_does_not_run_produce_document(self) -> None:
        """A seam reader is final, but probing it must NOT execute the parse hook.

        Fails today: there is no produce_document seam, so the base probe runs
        load_data(None, None), hits ReadDocument's NotImplementedError, and
        wrongly classifies the reader as False.
        """
        _ProbeRecorderDoc.produce_calls.clear()

        is_final = _ProbeRecorderDoc.is_final_reader()

        assert is_final is True
        assert _ProbeRecorderDoc.produce_calls == [], "probe must not call produce_document()"

    def test_probe_does_not_raise_when_produce_document_would_raise(self) -> None:
        """Probing a reader whose parse hook raises must still return True, not propagate."""
        is_final = _BoomProbeDoc.is_final_reader()

        assert is_final is True

    def test_wholesale_load_data_override_is_final(self) -> None:
        """Regression guard: overriding load_data wholesale still counts as a final reader."""
        assert _WholesaleDoc.is_final_reader() is True

    def test_abstract_and_intermediate_bases_stay_false(self) -> None:
        """Regression guard: classes without a parse hook are not final readers."""
        assert ReadDocument.is_final_reader() is False
        assert _SeamIntermediateDoc.is_final_reader() is False


class TestInRepoReadersMigrateToTheSeam:
    """The four concrete readers must adopt the seam instead of rewriting load_data.

    All of these fail today: each reader still overrides load_data wholesale and
    none defines produce_document.
    """

    @pytest.mark.parametrize(
        "reader",
        [MarkdownDocumentReader, TextFileReader, JsonDocumentReader, YamlDocumentReader],
    )
    def test_readers_use_inherited_load_data_template(self, reader: type[ReadDocument]) -> None:
        """Each in-repo reader inherits ReadDocument.load_data and overrides produce_document."""
        assert _underlying(reader.load_data) is _underlying(ReadDocument.load_data), (
            f"{reader.__name__} must not override load_data wholesale anymore"
        )
        assert _underlying(reader.produce_document) is not _underlying(ReadDocument.produce_document), (
            f"{reader.__name__} must override the produce_document parse hook"
        )

    def test_markdown_produce_document_returns_raw_text(self, tmp_path: Path) -> None:
        """MarkdownDocumentReader.produce_document returns the raw file text."""
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Title\n\nbody\n", encoding="utf-8")

        assert MarkdownDocumentReader.produce_document(str(file_path)) == "# Title\n\nbody\n"

    def test_text_produce_document_returns_raw_text(self, tmp_path: Path) -> None:
        """TextFileReader.produce_document returns the raw file text."""
        file_path = tmp_path / "doc.text"
        file_path.write_text("plain text body\n", encoding="utf-8")

        assert TextFileReader.produce_document(str(file_path)) == "plain text body\n"

    def test_json_produce_document_returns_json_dumps_of_loaded_content(self, tmp_path: Path) -> None:
        """JsonDocumentReader.produce_document round-trips via json.load + json.dumps."""
        file_path = tmp_path / "doc.json"
        file_path.write_text('{"a": 1, "b": [true, null]}', encoding="utf-8")

        expected = json.dumps(json.loads('{"a": 1, "b": [true, null]}'))
        assert JsonDocumentReader.produce_document(str(file_path)) == expected

    def test_yaml_produce_document_single_document_is_unwrapped(self, tmp_path: Path) -> None:
        """YamlDocumentReader.produce_document dumps a single document unwrapped."""
        file_path = tmp_path / "doc.yaml"
        file_path.write_text("a: 1\nb: two\n", encoding="utf-8")

        expected = yaml.dump({"a": 1, "b": "two"})
        assert YamlDocumentReader.produce_document(str(file_path)) == expected

    def test_yaml_produce_document_multi_document_stays_a_list(self, tmp_path: Path) -> None:
        """YamlDocumentReader.produce_document keeps multi-document streams as a list."""
        file_path = tmp_path / "doc.yaml"
        file_path.write_text("a: 1\n---\nb: 2\n", encoding="utf-8")

        expected = yaml.dump([{"a": 1}, {"b": 2}])
        assert YamlDocumentReader.produce_document(str(file_path)) == expected

    def test_yaml_document_file_type_uses_actual_path_suffix(self) -> None:
        """YamlDocumentReader.document_file_type reports the real suffix, e.g. 'yml' for .yml files.

        suffix()[0] is '.yaml', so the default would wrongly report 'yaml' for a .yml file.
        """
        assert YamlDocumentReader.document_file_type("/some/where/config.yml") == "yml"


class TestPyFileReaderStaysOnTheSeam:
    def test_py_file_reader_inherits_text_reader_and_is_final(self) -> None:
        """PyFileReader keeps inheriting from TextFileReader and still classifies as final."""
        assert issubclass(PyFileReader, TextFileReader)
        assert PyFileReader.is_final_reader() is True


# --------------------------------------------------------------------------------------
# Module-scope readers pinning the review-gap fix: a parse hook alone is NOT final.
#
# is_final_reader() classifies a reader as final from the produce_document
# override alone, mirroring the ReadFile hole the reviewers found: a class without a
# suffix override still classifies final, yet the default document_file_type calls
# cls.suffix(), so load_data would crash on the abstract suffix(). Document *matching*
# is guarded (_document_file_matches checks _has_suffix, and match_document_data_access
# swallows the NotImplementedError), so a leaked suffix-less class is already harmless
# in sibling discovery, but the classification screen must still mirror the family
# pattern (ReadDB requires produce_rows AND connect; ReadFile requires produce_table AND
# suffix AND _pyarrow_module): a ReadDocument subclass is "final via the seam" only if
# produce_document AND suffix are BOTH overridden relative to ReadDocument.
# --------------------------------------------------------------------------------------


class _DocHookNoSuffixDoc(ReadDocument):
    """Overrides produce_document but NOT suffix: cannot match or label files, so not final.

    The default document_file_type calls cls.suffix(), so load_data on this shape would
    raise NotImplementedError; the classification screen must reject it up front (like
    _RowHookNoConnectDB in the ReadDB seam tests).
    """

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return "no-suffix"


class _DocHookNoSuffixBaseDoc(ReadDocument):
    """Intermediate base: overrides produce_document, leaves suffix abstract; not final.

    The reviewer-reported shape: a shared parse base for a family of document formats.
    Only concrete children that add suffix complete the hook set.
    """

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return "base"


class _SuffixOnlyChildDoc(_DocHookNoSuffixBaseDoc):
    """Concrete child completing the base with only a sentinel suffix: the full hook set, final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdocseamsfxchild",)


class TestReadDocumentSeamClassificationScreensMissingSuffix:
    """Final via the seam iff produce_document AND suffix are both overridden."""

    def test_parse_hook_without_suffix_is_not_final(self) -> None:
        """A parse hook without a suffix override cannot label documents, so it must be screened out.

        Fails today: the structural check returns True from the produce_document override
        alone, ignoring that document_file_type would crash on the abstract suffix().
        """
        assert _DocHookNoSuffixDoc.is_final_reader() is False

    def test_intermediate_parse_base_without_suffix_is_not_final(self) -> None:
        """An intermediate parse base (produce_document, no suffix) is not final.

        Fails today: the produce_document override alone classifies the base as final.
        """
        assert _DocHookNoSuffixBaseDoc.is_final_reader() is False

    def test_concrete_child_completing_the_hook_set_is_final(self) -> None:
        """Regression guard: a child adding only suffix on top of a parse base stays final.

        Passes today; pins that tightening the screen must not reject inherited overrides.
        """
        assert _SuffixOnlyChildDoc.is_final_reader() is True
