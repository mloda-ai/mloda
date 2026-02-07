"""Tests that ConcatenatedFileContent uses ReadDocumentFeature instead of ReadFileFeature.

After migrating TextFileReader/PyFileReader from ReadFile to ReadDocument,
consumer code must reference ReadDocumentFeature for link matching.
PyFileReader is now a ReadDocument subclass, so ReadFileFeature links
will not find it. These tests verify the consumer has been updated.
"""

import inspect
import os
import tempfile

import pytest

from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda.user import FeatureName, Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader
from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader


class TestConcatenatedFileContentUsesReadDocumentFeature:
    """ConcatenatedFileContent must use ReadDocumentFeature after PyFileReader migration."""

    def test_source_code_references_read_document_feature(self) -> None:
        """The ConcatenatedFileContent source should reference ReadDocumentFeature, not ReadFileFeature."""
        source = inspect.getsource(ConcatenatedFileContent)
        assert "ReadDocumentFeature" in source, (
            "ConcatenatedFileContent source does not reference ReadDocumentFeature. "
            "After PyFileReader migration to ReadDocument, links must use ReadDocumentFeature."
        )

    def test_source_code_does_not_reference_read_file_feature(self) -> None:
        """The ConcatenatedFileContent source should not reference ReadFileFeature."""
        source = inspect.getsource(ConcatenatedFileContent)
        assert "ReadFileFeature" not in source, (
            "ConcatenatedFileContent source still references ReadFileFeature. "
            "PyFileReader is now a ReadDocument subclass; links must use ReadDocumentFeature."
        )

    def test_module_imports_read_document_feature(self) -> None:
        """The read_context_files module should import ReadDocumentFeature."""
        import mloda_plugins.feature_group.input_data.read_context_files as module

        module_source = inspect.getsource(module)
        assert (
            "from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature"
            in module_source
        ), (
            "Module does not import ReadDocumentFeature. "
            "It should replace the ReadFileFeature import with ReadDocumentFeature."
        )

    def test_module_does_not_import_read_file_feature(self) -> None:
        """The read_context_files module should no longer import ReadFileFeature."""
        import mloda_plugins.feature_group.input_data.read_context_files as module

        module_source = inspect.getsource(module)
        assert (
            "from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature" not in module_source
        ), "Module still imports ReadFileFeature. This import should be replaced with ReadDocumentFeature."

    def test_source_tuples_link_to_read_document_feature(self) -> None:
        """SourceTuples created by _create_source_tuples should use ReadDocumentFeature in links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_names = ["a.py", "b.py", "c.py"]
            for name in file_names:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(f"# {name}")

            instance = ConcatenatedFileContent()
            file_paths = [os.path.join(tmpdir, n) for n in file_names]

            instance._create_join_class(ConcatenatedFileContent.join_feature_name)
            features = instance._create_source_tuples(
                file_paths, FeatureName("test_feature"), PyFileReader.get_class_name()
            )

            feature = next(iter(features))
            source_tuples = feature.options.get(DefaultOptionKeys.in_features)

            tuples_with_links = [st for st in source_tuples if st.left_link is not None]
            assert len(tuples_with_links) > 0, "Expected at least one SourceTuple with link references."

            for st in tuples_with_links:
                left_link_class = st.left_link[0]
                assert left_link_class is ReadDocumentFeature, (
                    f"SourceTuple left_link references {left_link_class.__name__}, expected ReadDocumentFeature."
                )

                if st.right_link is not None:
                    right_link_class = st.right_link[0]
                    assert right_link_class is ReadDocumentFeature, (
                        f"SourceTuple right_link references {right_link_class.__name__}, expected ReadDocumentFeature."
                    )

    def test_no_source_tuple_links_reference_read_file_feature(self) -> None:
        """No SourceTuple should reference ReadFileFeature in its links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_names = ["x.py", "y.py"]
            for name in file_names:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(f"# {name}")

            instance = ConcatenatedFileContent()
            file_paths = [os.path.join(tmpdir, n) for n in file_names]

            instance._create_join_class(ConcatenatedFileContent.join_feature_name)
            features = instance._create_source_tuples(
                file_paths, FeatureName("test_feature"), PyFileReader.get_class_name()
            )

            feature = next(iter(features))
            source_tuples = feature.options.get(DefaultOptionKeys.in_features)

            for st in source_tuples:
                if st.left_link is not None:
                    assert st.left_link[0] is not ReadFileFeature, (
                        f"SourceTuple left_link still references ReadFileFeature. "
                        f"Must use ReadDocumentFeature after PyFileReader migration."
                    )
                if st.right_link is not None:
                    assert st.right_link[0] is not ReadFileFeature, (
                        f"SourceTuple right_link still references ReadFileFeature. "
                        f"Must use ReadDocumentFeature after PyFileReader migration."
                    )


class TestConcatenatedFileContentFormatAgnostic:
    """Tests for making ConcatenatedFileContent format-agnostic via document_reader_class option."""

    def test_missing_document_reader_class_raises_error(self) -> None:
        """ConcatenatedFileContent should raise ValueError if document_reader_class option is not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_names = ["test1.py", "test2.py"]
            for name in file_names:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(f"# {name}")

            options = Options({
                "file_paths": [os.path.join(tmpdir, n) for n in file_names]
                # NOTE: document_reader_class is NOT provided
            })

            instance = ConcatenatedFileContent()
            instance._create_join_class(ConcatenatedFileContent.join_feature_name)

            with pytest.raises(ValueError, match="document_reader_class.*required"):
                instance.input_features(options, FeatureName("test"))

    def test_explicit_document_reader_class_option(self) -> None:
        """With document_reader_class option, should use specified reader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .md files
            file_names = ["test1.md", "test2.md"]
            for name in file_names:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(f"# {name}")

            # Create options WITH document_reader_class
            options = Options(
                {
                    "file_paths": [os.path.join(tmpdir, n) for n in file_names],
                    "document_reader_class": MarkdownDocumentReader.get_class_name(),
                }
            )

            instance = ConcatenatedFileContent()
            instance._create_join_class(ConcatenatedFileContent.join_feature_name)
            features = instance.input_features(options, FeatureName("test"))

            # Extract SourceTuples and verify source_class
            assert features is not None
            feature = next(iter(features))
            source_tuples = feature.options.get(DefaultOptionKeys.in_features)

            for st in source_tuples:
                assert st.source_class == MarkdownDocumentReader.get_class_name(), (
                    f"Should use MarkdownDocumentReader, but got {st.source_class}"
                )

    def test_markdown_reader_with_md_files(self) -> None:
        """Integration test: ConcatenatedFileContent with MarkdownDocumentReader should process .md files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .md files with markdown content
            file_names = ["doc1.md", "doc2.md"]
            file_contents = {
                "doc1.md": "# Document 1\nThis is markdown content.",
                "doc2.md": "# Document 2\nAnother markdown file.",
            }

            for name, content in file_contents.items():
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(content)

            # Use ConcatenatedFileContent with document_reader_class option
            options = Options(
                {
                    "file_paths": [os.path.join(tmpdir, n) for n in file_names],
                    "document_reader_class": MarkdownDocumentReader.get_class_name(),
                }
            )

            instance = ConcatenatedFileContent()
            instance._create_join_class(ConcatenatedFileContent.join_feature_name)
            features = instance.input_features(options, FeatureName("test_md"))

            # Verify it routes to MarkdownDocumentReader
            assert features is not None
            feature = next(iter(features))
            source_tuples = feature.options.get(DefaultOptionKeys.in_features)

            # All source tuples should reference MarkdownDocumentReader
            reader_classes = {st.source_class for st in source_tuples}
            assert reader_classes == {MarkdownDocumentReader.get_class_name()}, (
                f"Expected only MarkdownDocumentReader, but got {reader_classes}"
            )
