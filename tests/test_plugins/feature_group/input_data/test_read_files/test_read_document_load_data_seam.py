"""Tests for the ReadDocument.load_data template-method lifecycle seam.

Contract pinned here (mirrors the ReadDB seam):

    ReadDocument gains
      - classmethod produce_document(file_path): the per-format parse hook; the base default
        raises NotImplementedError.
      - classmethod document_file_type(file_path): overridable; default is
        cls.suffix()[0].lstrip(".").
      - classmethod _read_text(file_path): shared helper reading the whole file as UTF-8 text.

    load_data becomes a template method that
      1. probes the DYNAMIC anchor first: it resolves cls.final_reader_anchor() and checks the
         anchor's _final_reader_requires() hooks via _is_overridden against THAT anchor (never
         hardcoded against ReadDocument); when the hook set is incomplete it raises
         NotImplementedError BEFORE any file read,
      2. file_path = features.get_options_key(cls.__name__),
      3. content = cls.produce_document(file_path),
      4. returns [{cls.get_class_name(): content, "source": file_path,
                   "file_type": cls.document_file_type(file_path)}].

    ReadDocument._final_reader_requires() returns ("produce_document", "suffix"): a reader is
    final iff it overrides BOTH hooks relative to its anchor, or overrides load_data wholesale.

    The four in-repo readers (TextFileReader, MarkdownDocumentReader, JsonDocumentReader,
    YamlDocumentReader) migrate onto the seam: they implement only produce_document (+ suffix;
    Yaml also overrides document_file_type to report the actual file suffix) and their
    end-to-end envelopes stay exactly as today.

Test isolation note:
All ReadDocument subclasses used by these tests are defined at MODULE scope, never inside
test methods. Function-local subclasses linger in ``ReadDocument.__subclasses__()`` (the
global plugin registry) until GC runs, leaking into sibling tests' plugin discovery, and
they are unpicklable, which breaks multiprocessing runners. Module-scope classes are
picklable and stable. Additionally, every seam reader uses a sentinel suffix in the
``.zzzdoc2seam*`` family that no real file in this repo or in any sibling test has, so a
leaked registry entry can never match a file in sibling discovery.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from mloda.provider import FeatureSet
from mloda.user import Feature
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
# plugin registry. Every suffix is a unique ``.zzzdoc2seam*`` sentinel that matches no
# real file anywhere, so a leaked class can never be selected by sibling tests' plugin
# discovery. The tests below drive these readers directly and record interactions via
# ClassVar lists.
# --------------------------------------------------------------------------------------


class _SeamIntermediateDoc(ReadDocument):
    """Intermediate base: overrides suffix() but NOT produce_document; not a final reader."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seamint",)


class _SeamHappyPathDoc(ReadDocument):
    """Final seam reader: overrides suffix + produce_document, records received file paths."""

    received_paths: ClassVar[list[str]] = []

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seam",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        cls.received_paths.append(file_path)
        return "produced::" + cls._read_text(file_path)


class _CustomFileTypeDoc(ReadDocument):
    """Final seam reader overriding document_file_type to use the ACTUAL path suffix."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seamcfta", ".zzzdoc2seamcftb")

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return cls._read_text(file_path)

    @classmethod
    def document_file_type(cls, file_path: str) -> str:
        return Path(file_path).suffix.lstrip(".")


class _WholesaleDoc(ReadDocument):
    """Legacy-style reader: overrides load_data wholesale, no produce_document; still final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seamwhole",)

    @classmethod
    def load_data(cls, data_access: Any, features: Any) -> Any:
        return [{"_WholesaleDoc": "wholesale", "source": "n/a", "file_type": "zzzdoc2seamwhole"}]


class _DocHookNoSuffixDoc(ReadDocument):
    """Overrides produce_document but NOT suffix: cannot match or label files, so not final.

    The default document_file_type calls cls.suffix(), so the classification screen must
    reject this shape up front (mirroring ReadDB's connect requirement).
    """

    produce_calls: ClassVar[list[str]] = []

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        cls.produce_calls.append(file_path)
        return cls._read_text(file_path)


class _DocParseBaseDoc(ReadDocument):
    """Intermediate base: overrides produce_document, leaves suffix abstract; not final."""

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return "base"


class _SuffixOnlyChildDoc(_DocParseBaseDoc):
    """Concrete child completing the base with only a sentinel suffix: full hook set, final."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seamsfxchild",)


# --------------------------------------------------------------------------------------
# Third-party sub-family re-anchoring _final_reader_requires below ReadDocument.
#
# _SubFamilyDoc shares produce_document for the whole sub-family and redeclares the
# requires tuple, becoming the anchor for its children. Relative to ReadDocument its
# produce_document IS overridden, so a probe hardcoded against ReadDocument would wrongly
# let _SubFamilyDoc.load_data proceed into the file read. The dynamic probe resolves the
# anchor (_SubFamilyDoc itself) and must raise NotImplementedError first.
# --------------------------------------------------------------------------------------


class _SubFamilyDoc(ReadDocument):
    """Sub-family base: redeclares the requires tuple and provides a shared produce_document."""

    produce_calls: ClassVar[list[str]] = []

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        return ("produce_document", "suffix")

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seamsubfam",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        cls.produce_calls.append(file_path)
        return cls._read_text(file_path)


class _SubFamilyChildDoc(_SubFamilyDoc):
    """Concrete child overriding both hooks relative to _SubFamilyDoc: final, runs the template."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".zzzdoc2seamsubchild",)

    @classmethod
    def produce_document(cls, file_path: str) -> Any:
        return "child::" + cls._read_text(file_path)


class TestReadDocumentSeamHooks:
    def test_produce_document_base_default_raises_not_implemented(self) -> None:
        """The base parse hook exists on ReadDocument and raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ReadDocument.produce_document("/some/where/doc.txt")

    def test_document_file_type_default_is_first_suffix_without_dot(self) -> None:
        """The default document_file_type is suffix()[0].lstrip(".")."""
        assert _SeamHappyPathDoc.document_file_type("anything.zzzdoc2seam") == "zzzdoc2seam"

    def test_read_text_reads_utf8(self, tmp_path: Path) -> None:
        """_read_text is a shared UTF-8 text-reading helper on ReadDocument."""
        file_path = tmp_path / "doc.txt"
        file_path.write_text("héllo séam\n", encoding="utf-8")

        assert ReadDocument._read_text(str(file_path)) == "héllo séam\n"

    def test_read_document_requires_produce_document_and_suffix(self) -> None:
        """ReadDocument._final_reader_requires() names the phase-2 hook set."""
        assert ReadDocument._final_reader_requires() == ("produce_document", "suffix")


class TestReadDocumentTemplate:
    def test_seam_happy_path_load_data_envelope(self, tmp_path: Path) -> None:
        """load_data resolves the path from options, parses via the hook, and wraps the envelope."""
        _SeamHappyPathDoc.received_paths.clear()

        file_path = tmp_path / "doc.zzzdoc2seam"
        file_path.write_text("hello seam", encoding="utf-8")

        features = MockFeatureSet({"_SeamHappyPathDoc": str(file_path)})
        result = _SeamHappyPathDoc.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {
                "_SeamHappyPathDoc": "produced::hello seam",
                "source": str(file_path),
                "file_type": "zzzdoc2seam",
            }
        ]
        assert _SeamHappyPathDoc.received_paths == [str(file_path)]

    def test_document_file_type_override_is_honored_in_envelope(self, tmp_path: Path) -> None:
        """An override using the actual path suffix (like YamlDocumentReader) reaches the envelope."""
        file_path = tmp_path / "doc.zzzdoc2seamcftb"
        file_path.write_text("second suffix", encoding="utf-8")

        features = MockFeatureSet({"_CustomFileTypeDoc": str(file_path)})
        result = _CustomFileTypeDoc.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {
                "_CustomFileTypeDoc": "second suffix",
                "source": str(file_path),
                "file_type": "zzzdoc2seamcftb",
            }
        ]


class TestReadDocumentProbeIsDynamic:
    """The template probe resolves the anchor dynamically; it is never hardcoded against ReadDocument."""

    def test_sub_family_re_anchors_classification(self) -> None:
        """Redeclaring _final_reader_requires moves the anchor to the sub-family base."""
        assert _SubFamilyDoc.final_reader_anchor() is _SubFamilyDoc
        assert _SubFamilyChildDoc.final_reader_anchor() is _SubFamilyDoc

    def test_sub_family_base_is_not_final_and_raises_before_reading(self, tmp_path: Path) -> None:
        """The sub-family base shares produce_document but is not final; load_data must probe
        against the re-anchored hook set and raise BEFORE any file read.

        The pointed-to path does not exist: a file read would raise FileNotFoundError, so
        NotImplementedError proves the probe fired first. A probe hardcoded against
        ReadDocument would see produce_document overridden and read the file.
        """
        _SubFamilyDoc.produce_calls.clear()

        assert _SubFamilyDoc.is_final_reader() is False

        missing = tmp_path / "missing.zzzdoc2seamsubfam"
        features = MockFeatureSet({"_SubFamilyDoc": str(missing)})
        with pytest.raises(NotImplementedError):
            _SubFamilyDoc.load_data(None, features)  # type: ignore[arg-type]

        assert _SubFamilyDoc.produce_calls == [], "produce_document must not run before the dynamic probe"

    def test_sub_family_child_is_final_and_runs_the_template(self, tmp_path: Path) -> None:
        """A child overriding both hooks relative to the sub-family anchor loads via the template."""
        assert _SubFamilyChildDoc.is_final_reader() is True

        file_path = tmp_path / "doc.zzzdoc2seamsubchild"
        file_path.write_text("family body", encoding="utf-8")

        features = MockFeatureSet({"_SubFamilyChildDoc": str(file_path)})
        result = _SubFamilyChildDoc.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {
                "_SubFamilyChildDoc": "child::family body",
                "source": str(file_path),
                "file_type": "zzzdoc2seamsubchild",
            }
        ]

    @pytest.mark.parametrize(
        "non_final",
        [ReadDocument, _SeamIntermediateDoc, _DocHookNoSuffixDoc, _DocParseBaseDoc, _SubFamilyDoc],
    )
    def test_non_final_implies_template_raises_before_any_file_read(
        self, non_final: type[ReadDocument], tmp_path: Path
    ) -> None:
        """Consistency contract: is_final_reader() False implies load_data raises NotImplementedError.

        Every path points at a nonexistent file, so a premature file read would surface as
        FileNotFoundError instead of NotImplementedError.
        """
        assert non_final.is_final_reader() is False

        missing = tmp_path / f"missing_{non_final.__name__}.zzzdoc2seamnone"
        features = MockFeatureSet({non_final.__name__: str(missing)})
        with pytest.raises(NotImplementedError):
            non_final.load_data(None, features)  # type: ignore[arg-type]

    def test_hook_without_suffix_probe_raises_before_reading(self, tmp_path: Path) -> None:
        """A produce_document-only class fails the probe before its parse hook runs."""
        _DocHookNoSuffixDoc.produce_calls.clear()

        missing = tmp_path / "missing.zzzdoc2seamnosfx"
        features = MockFeatureSet({"_DocHookNoSuffixDoc": str(missing)})
        with pytest.raises(NotImplementedError):
            _DocHookNoSuffixDoc.load_data(None, features)  # type: ignore[arg-type]

        assert _DocHookNoSuffixDoc.produce_calls == [], "produce_document must not run when the probe fails"


class TestReadDocumentClassification:
    def test_read_document_itself_is_not_final(self) -> None:
        assert ReadDocument.is_final_reader() is False

    def test_hook_complete_reader_is_final_without_load_data_override(self) -> None:
        """Overriding produce_document + suffix is enough; no wholesale load_data needed."""
        assert "load_data" not in _SeamHappyPathDoc.__dict__
        assert _SeamHappyPathDoc.is_final_reader() is True

    def test_suffix_only_intermediate_base_is_not_final(self) -> None:
        assert _SeamIntermediateDoc.is_final_reader() is False

    def test_parse_hook_without_suffix_is_not_final(self) -> None:
        """suffix is in the requires tuple precisely to screen out this shape."""
        assert _DocHookNoSuffixDoc.is_final_reader() is False

    def test_intermediate_parse_base_without_suffix_is_not_final(self) -> None:
        assert _DocParseBaseDoc.is_final_reader() is False

    def test_concrete_child_completing_the_hook_set_is_final(self) -> None:
        """A child adding only suffix on top of a shared parse base is final."""
        assert _SuffixOnlyChildDoc.is_final_reader() is True

    def test_wholesale_load_data_override_is_final(self) -> None:
        assert _WholesaleDoc.is_final_reader() is True
        assert _WholesaleDoc.load_data(None, None) == [
            {"_WholesaleDoc": "wholesale", "source": "n/a", "file_type": "zzzdoc2seamwhole"}
        ]


class TestInRepoReadersMigrateToTheSeam:
    """The four concrete readers adopt the seam instead of rewriting load_data,
    and their end-to-end envelopes stay exactly as today.
    """

    @pytest.mark.parametrize(
        "reader",
        [TextFileReader, MarkdownDocumentReader, JsonDocumentReader, YamlDocumentReader],
    )
    def test_readers_use_inherited_load_data_template(self, reader: type[ReadDocument]) -> None:
        """Each in-repo reader inherits the template and overrides produce_document."""
        assert "load_data" not in reader.__dict__, f"{reader.__name__} must not override load_data wholesale anymore"
        assert _underlying(reader.produce_document) is not _underlying(ReadDocument.produce_document), (
            f"{reader.__name__} must override the produce_document parse hook"
        )

    def test_text_reader_envelope_stays_exact(self, tmp_path: Path) -> None:
        file_path = tmp_path / "doc.text"
        file_path.write_text("plain text body\n", encoding="utf-8")

        features = MockFeatureSet({"TextFileReader": str(file_path)})
        result = TextFileReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [{"TextFileReader": "plain text body\n", "source": str(file_path), "file_type": "text"}]

    def test_markdown_reader_envelope_stays_exact(self, tmp_path: Path) -> None:
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Title\n\nbody\n", encoding="utf-8")

        features = MockFeatureSet({"MarkdownDocumentReader": str(file_path)})
        result = MarkdownDocumentReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [{"MarkdownDocumentReader": "# Title\n\nbody\n", "source": str(file_path), "file_type": "md"}]

    def test_json_reader_envelope_stays_exact(self, tmp_path: Path) -> None:
        """JSON content is json.dumps of the parsed file."""
        raw = '{"a": 1, "b": [true, null]}'
        file_path = tmp_path / "doc.json"
        file_path.write_text(raw, encoding="utf-8")

        features = MockFeatureSet({"JsonDocumentReader": str(file_path)})
        result = JsonDocumentReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {"JsonDocumentReader": json.dumps(json.loads(raw)), "source": str(file_path), "file_type": "json"}
        ]

    def test_yaml_reader_envelope_single_document_stays_exact(self, tmp_path: Path) -> None:
        """YAML content is yaml.dump of the single parsed document."""
        file_path = tmp_path / "doc.yaml"
        file_path.write_text("a: 1\nb: two\n", encoding="utf-8")

        features = MockFeatureSet({"YamlDocumentReader": str(file_path)})
        result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {"YamlDocumentReader": yaml.dump({"a": 1, "b": "two"}), "source": str(file_path), "file_type": "yaml"}
        ]

    def test_yaml_reader_envelope_multi_document_stays_a_list(self, tmp_path: Path) -> None:
        """Multi-document YAML streams yield yaml.dump of the document list."""
        file_path = tmp_path / "doc.yaml"
        file_path.write_text("a: 1\n---\nb: 2\n", encoding="utf-8")

        features = MockFeatureSet({"YamlDocumentReader": str(file_path)})
        result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {"YamlDocumentReader": yaml.dump([{"a": 1}, {"b": 2}]), "source": str(file_path), "file_type": "yaml"}
        ]

    def test_yml_file_reports_yml_file_type(self, tmp_path: Path) -> None:
        """Yaml overrides document_file_type so .yml files report 'yml', not suffix()[0]."""
        file_path = tmp_path / "doc.yml"
        file_path.write_text("key: value\n", encoding="utf-8")

        features = MockFeatureSet({"YamlDocumentReader": str(file_path)})
        result = YamlDocumentReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [
            {"YamlDocumentReader": yaml.dump({"key": "value"}), "source": str(file_path), "file_type": "yml"}
        ]

    def test_yaml_document_file_type_uses_actual_path_suffix(self) -> None:
        """YamlDocumentReader.document_file_type reports the real suffix without touching the file."""
        assert YamlDocumentReader.document_file_type("/some/where/config.yml") == "yml"
        assert YamlDocumentReader.document_file_type("/some/where/config.yaml") == "yaml"


class TestPyFileReaderStaysOnTheSeam:
    def test_py_file_reader_inherits_text_reader_and_is_final(self) -> None:
        """PyFileReader keeps inheriting from TextFileReader and still classifies as final."""
        assert issubclass(PyFileReader, TextFileReader)
        assert PyFileReader.is_final_reader() is True

    def test_py_file_reader_uses_inherited_produce_document(self) -> None:
        """PyFileReader adds only suffix; the parse hook comes from TextFileReader."""
        assert "load_data" not in PyFileReader.__dict__
        assert "produce_document" not in PyFileReader.__dict__
        assert _underlying(PyFileReader.produce_document) is _underlying(TextFileReader.produce_document)

    def test_py_file_reader_envelope_for_py_documents(self, tmp_path: Path) -> None:
        file_path = tmp_path / "module.py"
        file_path.write_text("print('hi')\n", encoding="utf-8")

        features = MockFeatureSet({"PyFileReader": str(file_path)})
        result = PyFileReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [{"PyFileReader": "print('hi')\n", "source": str(file_path), "file_type": "py"}]


class TestReadDocumentUsesResolvedDataAccessPath:
    """The template uses the RESOLVED data_access path.

    When ``data_access`` is a ``str`` or ``Path`` it IS the file path. Otherwise the template
    falls back to ``features.get_options_key(cls.__name__)``. On the GLOBAL
    DataAccessCollection matching path the matched file is stored solely under the reserved
    ``"BaseInputData"`` options key, so ``get_options_key`` would return ``None``; using the
    resolved ``data_access`` path instead is what lets ``BaseInputData.load`` hand the matched
    path through correctly.
    """

    def test_direct_load_data_uses_data_access_string_path(self, tmp_path: Path) -> None:
        """(2a) A str data_access is the file path; no reader-named options key needed."""
        file_path = tmp_path / "doc.text"
        file_path.write_text("body via data_access\n", encoding="utf-8")

        features = MockFeatureSet({})
        result = TextFileReader.load_data(str(file_path), features)  # type: ignore[arg-type]

        assert result == [{"TextFileReader": "body via data_access\n", "source": str(file_path), "file_type": "text"}]

    def test_direct_load_data_uses_data_access_path_object(self, tmp_path: Path) -> None:
        """(2a) A pathlib.Path data_access is the file path as well.

        The source assertion is stringified on purpose: whether the envelope carries the
        Path or its str form is an implementation detail; the CONTENT and the pointed-to
        file are the contract.
        """
        file_path = tmp_path / "doc.text"
        file_path.write_text("path object body\n", encoding="utf-8")

        result = TextFileReader.load_data(Path(file_path), MockFeatureSet({}))  # type: ignore[arg-type]

        assert len(result) == 1
        envelope = result[0]
        assert envelope["TextFileReader"] == "path object body\n"
        assert str(envelope["source"]) == str(file_path)
        assert envelope["file_type"] == "text"

    def test_global_scope_matched_path_reaches_the_template_end_to_end(self, tmp_path: Path) -> None:
        """(2b) End-to-end global scope: options carry ONLY the reserved BaseInputData tuple
        (as global DataAccessCollection matching stores it), and instance.load(features)
        returns the correct envelope.
        """
        file_path = tmp_path / "doc.text"
        file_path.write_text("global scope body\n", encoding="utf-8")

        feature = Feature(name="doc_feature", options={"BaseInputData": (TextFileReader, str(file_path))})
        feature_set = FeatureSet()
        feature_set.add(feature)

        instance = TextFileReader()
        result = instance.load(feature_set)

        assert result == [{"TextFileReader": "global scope body\n", "source": str(file_path), "file_type": "text"}]

    def test_feature_scoped_options_key_still_wins_when_data_access_is_none(self, tmp_path: Path) -> None:
        """(2c) Guard, passes today and must keep passing: with data_access None the path
        still resolves from the reader-named options key (feature-scoped access)."""
        file_path = tmp_path / "doc.text"
        file_path.write_text("feature scoped body\n", encoding="utf-8")

        features = MockFeatureSet({"TextFileReader": str(file_path)})
        result = TextFileReader.load_data(None, features)  # type: ignore[arg-type]

        assert result == [{"TextFileReader": "feature scoped body\n", "source": str(file_path), "file_type": "text"}]


class TestMissingOptionsKeyGuard:
    """An absent reader-named options key raises an actionable ValueError.

    On the fallback path (data_access is not a str/Path) the template resolves the file path
    via ``features.get_options_key(cls.__name__)``. When the options are initialized but do
    NOT carry the reader-name key, the template raises ValueError naming the reader class and
    the expected options key (the reader class name) before any parse attempt, instead of a
    bare TypeError from ``open(None)``.
    """

    @pytest.mark.parametrize("reader", [TextFileReader, MarkdownDocumentReader])
    def test_missing_reader_named_key_raises_actionable_value_error(self, reader: type[ReadDocument]) -> None:
        """Initialized options without the reader-name key: ValueError naming the reader,
        not a bare TypeError from open(None).
        """
        feature = Feature(name="doc_feature", options={"unrelated_key": "unrelated_value"})
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(ValueError, match=reader.__name__):
            reader.load_data(None, feature_set)
