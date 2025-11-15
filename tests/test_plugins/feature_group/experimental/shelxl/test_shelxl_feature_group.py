"""
Tests for SHELXL Feature Group

This module contains comprehensive tests for the SHELXL crystallographic
structure refinement feature group.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.shelxl.base import ShelxlFeatureGroup, ShelxlRefinementArtifact
from mloda_plugins.feature_group.experimental.shelxl.pandas import ShelxlFeatureGroupPandas


class TestShelxlRefinementArtifact:
    """Test the SHELXL refinement artifact."""

    def test_custom_saver(self):
        """Test artifact saving functionality."""
        features = Mock()
        artifact_data = {"r1_factor": 0.05, "wr2_factor": 0.12}

        result = ShelxlRefinementArtifact.custom_saver(features, artifact_data)
        assert result == artifact_data

    def test_custom_loader(self):
        """Test artifact loading functionality."""
        features = Mock()
        features.options = Options({"test_feature": "test_data"})
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "test_feature"

        result = ShelxlRefinementArtifact.custom_loader(features)
        assert result == "test_data"

    def test_custom_loader_no_options(self):
        """Test artifact loading with no options."""
        features = Mock()
        features.options = None
        features.name_of_one_feature = None

        result = ShelxlRefinementArtifact.custom_loader(features)
        assert result is None


class TestShelxlFeatureGroupBase:
    """Test the base SHELXL feature group functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.feature_group = ShelxlFeatureGroup()

    def test_artifact_class(self):
        """Test that the correct artifact class is returned."""
        artifact_class = ShelxlFeatureGroup.artifact()
        assert artifact_class == ShelxlRefinementArtifact

    def test_extract_source_feature_refine(self):
        """Test extracting source feature from refine feature name."""
        result = self.feature_group._extract_source_feature("shelxl_refine__test_structure")
        assert result == "test_structure"

    def test_extract_source_feature_validate(self):
        """Test extracting source feature from validate feature name."""
        result = self.feature_group._extract_source_feature("shelxl_validate__my_crystal")
        assert result == "my_crystal"

    def test_extract_source_feature_stats(self):
        """Test extracting source feature from stats feature name."""
        result = self.feature_group._extract_source_feature("shelxl_stats__compound_x")
        assert result == "compound_x"

    def test_extract_source_feature_invalid(self):
        """Test extracting source feature from invalid feature name."""
        result = self.feature_group._extract_source_feature("invalid_feature_name")
        assert result is None

    def test_feature_names_supported(self):
        """Test supported feature names."""
        supported = ShelxlFeatureGroup.feature_names_supported()
        expected = {"shelxl_refine", "shelxl_validate", "shelxl_stats"}
        assert supported == expected

    def test_input_features(self):
        """Test input features definition."""
        options = Options()
        feature_name = FeatureName("shelxl_refine__test_structure")

        input_features = self.feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 2

        feature_names = {f.name for f in input_features}
        expected_names = {"test_structure_ins_file", "test_structure_hkl_file"}
        assert feature_names == expected_names

    def test_input_features_no_source(self):
        """Test input features with invalid feature name."""
        options = Options()
        feature_name = FeatureName("invalid_feature")

        input_features = self.feature_group.input_features(options, feature_name)
        assert input_features is None

    def test_parse_res_file_missing(self):
        """Test parsing non-existent .res file."""
        non_existent_file = Path("/tmp/non_existent.res")
        result = self.feature_group._parse_res_file(non_existent_file)
        assert result == {}

    def test_parse_checkcif_output(self):
        """Test parsing checkCIF output."""
        checkcif_output = """
ALERT level A - Serious problem
ALERT level B - Potentially serious problem  
ALERT level C - Check and explain
ALERT level D - Improvement, suggestion, query
"""

        result = self.feature_group._parse_checkcif_output(checkcif_output)

        assert "alerts_level_a" in result
        assert "alerts_level_b" in result
        assert "alerts_level_c" in result
        assert "alerts_level_d" in result

        assert len(result["alerts_level_a"]) == 1
        assert len(result["alerts_level_b"]) == 1
        assert len(result["alerts_level_c"]) == 1
        assert len(result["alerts_level_d"]) == 1

    def test_evaluate_validation_level_a(self):
        """Test validation evaluation for level A."""
        validation_results = {
            "alerts_level_a": [],
            "alerts_level_b": ["some alert"],
            "alerts_level_c": [],
            "alerts_level_d": [],
        }

        result = self.feature_group._evaluate_validation(validation_results, "A")
        assert result is True  # No level A alerts

    def test_evaluate_validation_level_b(self):
        """Test validation evaluation for level B."""
        validation_results = {
            "alerts_level_a": [],
            "alerts_level_b": [],
            "alerts_level_c": ["some alert"],
            "alerts_level_d": [],
        }

        result = self.feature_group._evaluate_validation(validation_results, "B")
        assert result is True  # No level A or B alerts

    def test_evaluate_validation_level_c(self):
        """Test validation evaluation for level C."""
        validation_results = {
            "alerts_level_a": [],
            "alerts_level_b": [],
            "alerts_level_c": [],
            "alerts_level_d": ["some alert"],
        }

        result = self.feature_group._evaluate_validation(validation_results, "C")
        assert result is True  # No level A, B, or C alerts

    def test_evaluate_validation_level_d(self):
        """Test validation evaluation for level D."""
        validation_results = {
            "alerts_level_a": ["serious alert"],
            "alerts_level_b": [],
            "alerts_level_c": [],
            "alerts_level_d": [],
        }

        result = self.feature_group._evaluate_validation(validation_results, "D")
        assert result is True  # Level D accepts all alerts

    def test_evaluate_validation_failed(self):
        """Test validation evaluation failure."""
        validation_results = {
            "alerts_level_a": ["serious alert"],
            "alerts_level_b": [],
            "alerts_level_c": [],
            "alerts_level_d": [],
        }

        result = self.feature_group._evaluate_validation(validation_results, "A")
        assert result is False  # Level A alerts present

    @patch("tempfile.mkdtemp")
    @patch("shutil.copy2")
    def test_prepare_working_directory(self, mock_copy, mock_mkdtemp):
        """Test preparing working directory."""
        mock_mkdtemp.return_value = "/tmp/shelxl_test"

        # Create temporary test files
        with tempfile.NamedTemporaryFile(suffix=".ins", delete=False) as ins_file:
            ins_path = ins_file.name
        with tempfile.NamedTemporaryFile(suffix=".hkl", delete=False) as hkl_file:
            hkl_path = hkl_file.name

        try:
            result = self.feature_group._prepare_working_directory(ins_path, hkl_path)

            assert result == Path("/tmp/shelxl_test")
            assert mock_copy.call_count == 2

        finally:
            Path(ins_path).unlink()
            Path(hkl_path).unlink()

    def test_prepare_working_directory_missing_files(self):
        """Test preparing working directory with missing files."""
        with pytest.raises(FileNotFoundError):
            self.feature_group._prepare_working_directory("/non/existent/file.ins", "/non/existent/file.hkl")


class TestShelxlFeatureGroupPandas:
    """Test the Pandas implementation of SHELXL feature group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.feature_group = ShelxlFeatureGroupPandas()

    def test_calculate_feature_invalid_input(self):
        """Test calculate_feature with invalid input type."""
        features = Mock()

        with pytest.raises(ValueError, match="ShelxlFeatureGroupPandas requires pandas DataFrame input"):
            ShelxlFeatureGroupPandas.calculate_feature("not a dataframe", features)

    def test_calculate_feature_missing_feature_name(self):
        """Test calculate_feature with missing feature name."""
        data = pd.DataFrame({"test": [1, 2, 3]})
        features = Mock()
        features.name_of_one_feature = None

        with pytest.raises(ValueError, match="Feature name is required"):
            ShelxlFeatureGroupPandas.calculate_feature(data, features)

    def test_calculate_feature_invalid_feature_name(self):
        """Test calculate_feature with invalid feature name."""
        data = pd.DataFrame({"test": [1, 2, 3]})
        features = Mock()
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "invalid_feature"
        features.options = Options()

        with pytest.raises(ValueError, match="Could not extract source feature"):
            ShelxlFeatureGroupPandas.calculate_feature(data, features)

    def test_calculate_feature_missing_columns(self):
        """Test calculate_feature with missing required columns."""
        data = pd.DataFrame({"test": [1, 2, 3]})
        features = Mock()
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "shelxl_refine__test_structure"
        features.options = Options()

        with pytest.raises(ValueError, match="Required column.*not found"):
            ShelxlFeatureGroupPandas.calculate_feature(data, features)

    @patch.object(ShelxlFeatureGroupPandas, "_perform_complete_refinement")
    def test_calculate_feature_refine_success(self, mock_refine):
        """Test successful refinement calculation."""
        # Mock the refinement method
        mock_refine.return_value = {
            "success": True,
            "r1_factor": 0.045,
            "wr2_factor": 0.123,
            "gof": 1.05,
            "validation_passed": True,
        }

        # Create test data
        data = pd.DataFrame(
            {"test_structure_ins_file": ["/path/to/test.ins"], "test_structure_hkl_file": ["/path/to/test.hkl"]}
        )

        # Create feature set
        features = Mock()
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "shelxl_refine__test_structure"
        features.options = Options()

        # Execute
        result = ShelxlFeatureGroupPandas.calculate_feature(data, features)

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "shelxl_refine__test_structure" in result.columns
        assert result["shelxl_refine__test_structure"].iloc[0] == "success"
        assert result["r1_factor"].iloc[0] == 0.045
        assert mock_refine.called

    @patch.object(ShelxlFeatureGroupPandas, "_perform_validation_only")
    def test_calculate_feature_validate_success(self, mock_validate):
        """Test successful validation calculation."""
        # Mock the validation method
        mock_validate.return_value = {
            "success": True,
            "validation_passed": True,
            "alerts_level_a": [],
            "alerts_level_b": [],
        }

        # Create test data
        data = pd.DataFrame(
            {"test_structure_ins_file": ["/path/to/test.ins"], "test_structure_hkl_file": ["/path/to/test.hkl"]}
        )

        # Create feature set
        features = Mock()
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "shelxl_validate__test_structure"
        features.options = Options()

        # Execute
        result = ShelxlFeatureGroupPandas.calculate_feature(data, features)

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "shelxl_validate__test_structure" in result.columns
        assert result["shelxl_validate__test_structure"].iloc[0] == True
        assert mock_validate.called

    @patch.object(ShelxlFeatureGroupPandas, "_extract_statistics_only")
    def test_calculate_feature_stats_success(self, mock_stats):
        """Test successful statistics extraction."""
        # Mock the statistics method
        mock_stats.return_value = {"success": True, "r1_factor": 0.067, "wr2_factor": 0.145, "gof": 1.12}

        # Create test data
        data = pd.DataFrame(
            {"test_structure_ins_file": ["/path/to/test.ins"], "test_structure_hkl_file": ["/path/to/test.hkl"]}
        )

        # Create feature set
        features = Mock()
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "shelxl_stats__test_structure"
        features.options = Options()

        # Execute
        result = ShelxlFeatureGroupPandas.calculate_feature(data, features)

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "shelxl_stats__test_structure" in result.columns
        assert result["shelxl_stats__test_structure"].iloc[0] == 0.067
        assert mock_stats.called

    def test_calculate_feature_error_handling(self):
        """Test error handling in calculate_feature."""
        # Create test data
        data = pd.DataFrame(
            {"test_structure_ins_file": ["/path/to/test.ins"], "test_structure_hkl_file": ["/path/to/test.hkl"]}
        )

        # Create feature set
        features = Mock()
        features.name_of_one_feature = Mock()
        features.name_of_one_feature.name = "shelxl_refine__test_structure"
        features.options = Options()

        # Mock the refinement method to raise an exception
        with patch.object(ShelxlFeatureGroupPandas, "_perform_complete_refinement") as mock_refine:
            mock_refine.side_effect = Exception("Test error")

            result = ShelxlFeatureGroupPandas.calculate_feature(data, features)

            # Verify error handling
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert result["success"].iloc[0] == False
            assert "Test error" in result["error"].iloc[0]
            assert result["shelxl_refine__test_structure"].iloc[0] == "failed"

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    def test_apply_global_restraints(self, mock_open, mock_exists):
        """Test applying global restraints to .ins file."""
        mock_exists.return_value = True

        # Mock file content
        mock_file_content = """TITL test structure
CELL 1.54178 10.0 10.0 10.0 90 90 90
ZERR 4 0.001 0.001 0.001 0.01 0.01 0.01
LATT 1
SYMM -X, -Y, -Z
SFAC C H N O
C1 1 0.5 0.5 0.5 11.0 0.05
"""

        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = mock_file_content
        mock_open.return_value.__enter__.return_value = mock_file

        # Test
        ins_file = Path("/tmp/test.ins")
        self.feature_group._apply_global_restraints(ins_file)

        # Verify file was opened for reading and writing
        assert mock_open.call_count == 2  # Once for read, once for write

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    def test_apply_anisotropic_refinement(self, mock_open, mock_exists):
        """Test applying anisotropic refinement."""
        mock_exists.return_value = True

        # Mock file content
        mock_file_content = """TITL test structure
CELL 1.54178 10.0 10.0 10.0 90 90 90
FVAR 1.0
C1 1 0.5 0.5 0.5 11.0 0.05
"""

        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = mock_file_content
        mock_open.return_value.__enter__.return_value = mock_file

        # Test
        ins_file = Path("/tmp/test.ins")
        self.feature_group._apply_anisotropic_refinement(ins_file)

        # Verify file was opened for reading and writing
        assert mock_open.call_count == 2

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    def test_set_refinement_cycles(self, mock_open, mock_exists):
        """Test setting refinement cycles."""
        mock_exists.return_value = True

        # Mock file content
        mock_file_content = """TITL test structure
L.S. 10
FVAR 1.0
"""

        # Mock file operations
        mock_file = MagicMock()
        mock_file.read.return_value = mock_file_content
        mock_open.return_value.__enter__.return_value = mock_file

        # Test
        ins_file = Path("/tmp/test.ins")
        self.feature_group._set_refinement_cycles(ins_file, 20)

        # Verify file was opened for reading and writing
        assert mock_open.call_count == 2

    @patch("pathlib.Path.exists")
    def test_generate_cif_file_existing(self, mock_exists):
        """Test CIF file generation when file already exists."""
        mock_exists.return_value = True

        working_dir = Path("/tmp/test_dir")
        basename = "test_structure"

        result = self.feature_group._generate_cif_file(working_dir, basename)

        expected_path = working_dir / f"{basename}.cif"
        assert result == expected_path


class TestShelxlIntegration:
    """Integration tests for SHELXL feature group."""

    def test_feature_group_description(self):
        """Test feature group description."""
        description = ShelxlFeatureGroupPandas.description()
        assert isinstance(description, str)
        assert len(description) > 0

    def test_feature_group_version(self):
        """Test feature group version."""
        version = ShelxlFeatureGroupPandas.version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_domain(self):
        """Test domain definition."""
        domain = ShelxlFeatureGroupPandas.get_domain()
        assert domain is not None
