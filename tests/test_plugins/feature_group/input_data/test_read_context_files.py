"""Tests that ConcatenatedFileContent uses ReadDocumentFeature instead of ReadFileFeature.

After migrating TextFileReader/PyFileReader from ReadFile to ReadDocument,
consumer code must reference ReadDocumentFeature for link matching.
PyFileReader is now a ReadDocument subclass, so ReadFileFeature links
will not find it. These tests verify the consumer has been updated.
"""

import inspect
import os
import tempfile

from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda.user import FeatureName
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


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
            features = instance._create_source_tuples(file_paths, FeatureName("test_feature"))

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
            features = instance._create_source_tuples(file_paths, FeatureName("test_feature"))

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
