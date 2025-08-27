"""
Base SHELXL Feature Group for Crystallography Structure Refinement

This module provides the abstract base class for SHELXL crystallographic
structure refinement workflows.
"""

from abc import ABC
from typing import Any, Optional, Set, Type
from pathlib import Path
import subprocess
import tempfile
import os

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class ShelxlRefinementArtifact(BaseArtifact):
    """Artifact for storing SHELXL refinement results."""

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """Save refinement results to artifact storage."""
        # Implementation for saving refinement data
        # For now, return the artifact to let the framework handle it
        return artifact

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """Load refinement results from artifact storage."""
        # Implementation for loading refinement data
        # For now, use the default implementation
        return (
            features.options.get(features.name_of_one_feature.name)
            if features.options and features.name_of_one_feature
            else None
        )


class ShelxlFeatureGroup(AbstractFeatureGroup, ABC):
    """
    Abstract base class for SHELXL crystallographic structure refinement.

    This feature group automates the complete SHELXL refinement workflow:
    1. Load .ins and .hkl files
    2. Run initial refinement cycle (L.S. 10)
    3. Apply restraints based on R1 factor
    4. Perform anisotropic refinement
    5. Place hydrogen atoms
    6. Handle disordered regions
    7. Validate geometry
    8. Refine occupancies
    9. Final refinement cycle (L.S. 20)
    10. Run checkCIF validation
    11. Generate output files (.res, .cif)
    """

    # Configuration option keys
    REFINEMENT_CYCLES = "refinement_cycles"
    R1_THRESHOLD = "r1_threshold"
    RESOLUTION_THRESHOLD = "resolution_threshold"
    COMPLETENESS_THRESHOLD = "completeness_threshold"
    VALIDATION_LEVEL = "validation_level"
    SHELXL_EXECUTABLE = "shelxl_executable"
    CHECKCIF_EXECUTABLE = "checkcif_executable"

    # Default values
    DEFAULT_REFINEMENT_CYCLES = 10
    DEFAULT_R1_THRESHOLD = 0.10
    DEFAULT_RESOLUTION_THRESHOLD = 0.8
    DEFAULT_COMPLETENESS_THRESHOLD = 0.90
    DEFAULT_VALIDATION_LEVEL = "C"  # Accept Level C and D alerts
    DEFAULT_SHELXL_EXECUTABLE = "shelxl"
    DEFAULT_CHECKCIF_EXECUTABLE = "checkcif"

    @staticmethod
    def artifact() -> Type[BaseArtifact]:
        """Returns the artifact class for SHELXL refinement results."""
        return ShelxlRefinementArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """
        Define input features for SHELXL refinement.

        Requires:
        - .ins file path
        - .hkl file path
        """
        source_feature = self._extract_source_feature(feature_name.name)
        if source_feature:
            return {Feature(f"{source_feature}_ins_file"), Feature(f"{source_feature}_hkl_file")}
        return None

    def _extract_source_feature(self, feature_name: str) -> Optional[str]:
        """Extract the source feature name from the SHELXL feature name."""
        if feature_name.startswith("shelxl_refine__"):
            return feature_name.replace("shelxl_refine__", "")
        elif feature_name.startswith("shelxl_validate__"):
            return feature_name.replace("shelxl_validate__", "")
        elif feature_name.startswith("shelxl_stats__"):
            return feature_name.replace("shelxl_stats__", "")
        return None

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        """Return supported feature name patterns."""
        return {"shelxl_refine", "shelxl_validate", "shelxl_stats"}

    def _run_shelxl_command(self, ins_file: Path, working_dir: Path, options: Options) -> dict:
        """
        Execute SHELXL refinement command.

        Args:
            ins_file: Path to the .ins instruction file
            working_dir: Working directory for refinement
            options: Configuration options

        Returns:
            Dictionary containing refinement results and statistics
        """
        shelxl_exe = options.get(self.SHELXL_EXECUTABLE) or self.DEFAULT_SHELXL_EXECUTABLE

        try:
            # Run SHELXL command
            result = subprocess.run(
                [shelxl_exe, str(ins_file.stem)],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"SHELXL execution failed: {result.stderr}")

            # Parse output for refinement statistics
            return self._parse_shelxl_output(working_dir, ins_file.stem)

        except subprocess.TimeoutExpired:
            raise RuntimeError("SHELXL execution timed out")
        except FileNotFoundError:
            raise RuntimeError(f"SHELXL executable '{shelxl_exe}' not found")

    def _parse_shelxl_output(self, working_dir: Path, basename: str) -> dict:
        """
        Parse SHELXL output files to extract refinement statistics.

        Args:
            working_dir: Directory containing output files
            basename: Base name of the structure files

        Returns:
            Dictionary containing parsed refinement statistics
        """
        results = {
            "r1_factor": None,
            "wr2_factor": None,
            "gof": None,
            "shift_error_max": None,
            "completeness": None,
            "resolution": None,
            "reflections_total": None,
            "reflections_unique": None,
            "parameters": None,
            "restraints": None,
        }

        # Parse .res file for final statistics
        res_file = working_dir / f"{basename}.res"
        if res_file.exists():
            results.update(self._parse_res_file(res_file))

        # Parse .lst file for detailed output
        lst_file = working_dir / f"{basename}.lst"
        if lst_file.exists():
            results.update(self._parse_lst_file(lst_file))

        return results

    def _parse_res_file(self, res_file: Path) -> dict:
        """Parse .res file for structure and refinement data."""
        results: dict[str, Any] = {}

        try:
            with open(res_file, "r") as f:
                content = f.read()

            # Extract R factors from REM lines
            lines = content.split("\n")
            for line in lines:
                if line.startswith("REM") and "R1" in line:
                    # Parse R1 and wR2 values
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "R1" and i + 2 < len(parts):
                            try:
                                results["r1_factor"] = float(parts[i + 2])
                            except ValueError:
                                pass
                        elif part == "wR2" and i + 2 < len(parts):
                            try:
                                results["wr2_factor"] = float(parts[i + 2])
                            except ValueError:
                                pass
                        elif part == "GooF" and i + 1 < len(parts):
                            try:
                                results["gof"] = float(parts[i + 1])
                            except ValueError:
                                pass

        except Exception as e:
            print(f"Warning: Could not parse .res file: {e}")

        return results

    def _parse_lst_file(self, lst_file: Path) -> dict:
        """Parse .lst file for detailed refinement information."""
        results: dict[str, Any] = {}

        try:
            with open(lst_file, "r") as f:
                content = f.read()

            lines = content.split("\n")
            for line in lines:
                # Look for final refinement statistics
                if "Final R indices" in line:
                    # Extract R1 and wR2 from final statistics
                    pass
                elif "Goodness-of-fit on F^2" in line:
                    # Extract GOF
                    pass
                elif "Largest diff. peak and hole" in line:
                    # Extract electron density peaks
                    pass

        except Exception as e:
            print(f"Warning: Could not parse .lst file: {e}")

        return results

    def _run_checkcif_validation(self, cif_file: Path, options: Options) -> dict:
        """
        Run checkCIF validation on the refined structure.

        Args:
            cif_file: Path to the .cif file
            options: Configuration options

        Returns:
            Dictionary containing validation results
        """
        checkcif_exe = options.get(self.CHECKCIF_EXECUTABLE) or self.DEFAULT_CHECKCIF_EXECUTABLE
        validation_results = {
            "alerts_level_a": [],
            "alerts_level_b": [],
            "alerts_level_c": [],
            "alerts_level_d": [],
            "validation_passed": False,
        }

        try:
            result = subprocess.run([checkcif_exe, str(cif_file)], capture_output=True, text=True, timeout=60)

            # Parse checkCIF output for alerts
            validation_results.update(self._parse_checkcif_output(result.stdout))

            # Determine if validation passed based on alert levels
            validation_level = options.get(self.VALIDATION_LEVEL) or self.DEFAULT_VALIDATION_LEVEL
            validation_results["validation_passed"] = self._evaluate_validation(validation_results, validation_level)

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Warning: checkCIF validation failed: {e}")

        return validation_results

    def _parse_checkcif_output(self, output: str) -> dict:
        """Parse checkCIF output for validation alerts."""
        alerts: dict[str, list[str]] = {
            "alerts_level_a": [],
            "alerts_level_b": [],
            "alerts_level_c": [],
            "alerts_level_d": [],
        }

        lines = output.split("\n")
        for line in lines:
            if "ALERT level A" in line:
                alerts["alerts_level_a"].append(line.strip())
            elif "ALERT level B" in line:
                alerts["alerts_level_b"].append(line.strip())
            elif "ALERT level C" in line:
                alerts["alerts_level_c"].append(line.strip())
            elif "ALERT level D" in line:
                alerts["alerts_level_d"].append(line.strip())

        return alerts

    def _evaluate_validation(self, validation_results: dict, validation_level: str) -> bool:
        """
        Evaluate if validation passed based on alert levels.

        Args:
            validation_results: Dictionary containing validation alerts
            validation_level: Maximum acceptable alert level (A, B, C, or D)

        Returns:
            True if validation passed, False otherwise
        """
        if validation_level == "A":
            return len(validation_results["alerts_level_a"]) == 0
        elif validation_level == "B":
            return len(validation_results["alerts_level_a"]) == 0 and len(validation_results["alerts_level_b"]) == 0
        elif validation_level == "C":
            return (
                len(validation_results["alerts_level_a"]) == 0
                and len(validation_results["alerts_level_b"]) == 0
                and len(validation_results["alerts_level_c"]) == 0
            )
        else:  # Level D or any other value
            return True  # Accept all alerts

    def _prepare_working_directory(self, ins_file_path: str, hkl_file_path: str) -> Path:
        """
        Prepare a temporary working directory with input files.

        Args:
            ins_file_path: Path to the .ins file
            hkl_file_path: Path to the .hkl file

        Returns:
            Path to the prepared working directory
        """
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="shelxl_"))

        # Copy input files to working directory
        ins_source = Path(ins_file_path)
        hkl_source = Path(hkl_file_path)

        if not ins_source.exists():
            raise FileNotFoundError(f"INS file not found: {ins_file_path}")
        if not hkl_source.exists():
            raise FileNotFoundError(f"HKL file not found: {hkl_file_path}")

        # Copy files with consistent basename
        basename = ins_source.stem
        ins_dest = temp_dir / f"{basename}.ins"
        hkl_dest = temp_dir / f"{basename}.hkl"

        import shutil

        shutil.copy2(ins_source, ins_dest)
        shutil.copy2(hkl_source, hkl_dest)

        return temp_dir

    def _cleanup_working_directory(self, working_dir: Path, preserve_outputs: bool = True) -> dict:
        """
        Clean up working directory and preserve output files.

        Args:
            working_dir: Path to the working directory
            preserve_outputs: Whether to preserve output files

        Returns:
            Dictionary containing paths to preserved output files
        """
        output_files = {}

        if preserve_outputs:
            # Find and preserve important output files
            for ext in [".res", ".cif", ".lst", ".fcf"]:
                output_file = list(working_dir.glob(f"*{ext}"))
                if output_file:
                    # Move to a permanent location or read content
                    output_files[ext] = output_file[0].read_text()

        # Clean up temporary directory
        import shutil

        try:
            shutil.rmtree(working_dir)
        except Exception as e:
            print(f"Warning: Could not clean up working directory: {e}")

        return output_files
