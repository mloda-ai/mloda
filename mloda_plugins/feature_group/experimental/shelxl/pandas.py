"""
Pandas implementation of SHELXL Feature Group for Crystallography Structure Refinement

This module provides the Pandas-specific implementation of the SHELXL feature group
for automated crystallographic structure refinement workflows.
"""

from typing import Any, Optional
from pathlib import Path
import pandas as pd

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_plugins.feature_group.experimental.shelxl.base import ShelxlFeatureGroup


class ShelxlFeatureGroupPandas(ShelxlFeatureGroup):
    """
    Pandas implementation of SHELXL crystallographic structure refinement.

    This feature group automates the complete SHELXL refinement workflow using
    Pandas DataFrames for data handling and result storage.

    Supported feature names:
    - shelxl_refine__{structure_name}: Complete refinement workflow
    - shelxl_validate__{structure_name}: Validation only
    - shelxl_stats__{structure_name}: Extract refinement statistics

    Input requirements:
    - {structure_name}_ins_file: Path to .ins instruction file
    - {structure_name}_hkl_file: Path to .hkl reflection data file

    Configuration options:
    - refinement_cycles: Number of refinement cycles (default: 10)
    - r1_threshold: R1 factor threshold for restraints (default: 0.10)
    - resolution_threshold: Resolution cutoff for anisotropic refinement (default: 0.8)
    - completeness_threshold: Data completeness threshold (default: 0.90)
    - validation_level: checkCIF alert level filtering (default: "C")
    - shelxl_executable: SHELXL executable name (default: "shelxl")
    - checkcif_executable: checkCIF executable name (default: "checkcif")
    """

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> pd.DataFrame:
        """
        Calculate SHELXL refinement features.

        Args:
            data: Input data containing file paths
            features: Feature set configuration

        Returns:
            DataFrame containing refinement results and statistics
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("ShelxlFeatureGroupPandas requires pandas DataFrame input")

        # Get feature name and options
        feature_name = features.name_of_one_feature
        options = features.options or Options()

        if not feature_name:
            raise ValueError("Feature name is required")

        # Extract source feature name
        instance = cls()
        source_feature = instance._extract_source_feature(feature_name.name)

        if not source_feature:
            raise ValueError(f"Could not extract source feature from: {feature_name.name}")

        # Get input file paths
        ins_file_col = f"{source_feature}_ins_file"
        hkl_file_col = f"{source_feature}_hkl_file"

        if ins_file_col not in data.columns:
            raise ValueError(f"Required column '{ins_file_col}' not found in input data")
        if hkl_file_col not in data.columns:
            raise ValueError(f"Required column '{hkl_file_col}' not found in input data")

        # Process each row
        results = []

        for idx, row in data.iterrows():
            ins_file_path = str(row[ins_file_col])
            hkl_file_path = str(row[hkl_file_col])

            try:
                # Determine operation type based on feature name
                if feature_name.name.startswith("shelxl_refine__"):
                    result = instance._perform_complete_refinement(ins_file_path, hkl_file_path, options)
                elif feature_name.name.startswith("shelxl_validate__"):
                    result = instance._perform_validation_only(ins_file_path, hkl_file_path, options)
                elif feature_name.name.startswith("shelxl_stats__"):
                    result = instance._extract_statistics_only(ins_file_path, hkl_file_path, options)
                else:
                    raise ValueError(f"Unsupported feature name pattern: {feature_name.name}")

                # Add row index for joining back to original data
                result["row_index"] = idx
                results.append(result)

            except Exception as e:
                # Create error result
                error_result = {
                    "row_index": idx,
                    "error": str(e),
                    "success": False,
                    "r1_factor": None,
                    "wr2_factor": None,
                    "gof": None,
                    "validation_passed": False,
                }
                results.append(error_result)

        # Convert results to DataFrame
        result_df = pd.DataFrame(results)

        # Set the feature name as column name for the main result
        if feature_name.name.startswith("shelxl_refine__"):
            result_df[feature_name.name] = result_df.apply(
                lambda row: "success" if row.get("success", False) else "failed", axis=1
            )
        elif feature_name.name.startswith("shelxl_validate__"):
            result_df[feature_name.name] = result_df["validation_passed"]
        elif feature_name.name.startswith("shelxl_stats__"):
            result_df[feature_name.name] = result_df["r1_factor"]

        return result_df

    def _perform_complete_refinement(self, ins_file_path: str, hkl_file_path: str, options: Options) -> dict:
        """
        Perform complete SHELXL refinement workflow.

        Args:
            ins_file_path: Path to .ins file
            hkl_file_path: Path to .hkl file
            options: Configuration options

        Returns:
            Dictionary containing refinement results
        """
        try:
            # Prepare working directory
            working_dir = self._prepare_working_directory(ins_file_path, hkl_file_path)
            ins_file = working_dir / f"{Path(ins_file_path).stem}.ins"

            # Step 1: Initial refinement cycle
            initial_results = self._run_shelxl_command(ins_file, working_dir, options)

            # Step 2: Check R1 factor and apply restraints if needed
            r1_threshold = options.get(self.R1_THRESHOLD) or self.DEFAULT_R1_THRESHOLD
            if initial_results.get("r1_factor", 1.0) > r1_threshold:
                self._apply_global_restraints(ins_file)
                # Re-run refinement after applying restraints
                initial_results = self._run_shelxl_command(ins_file, working_dir, options)

            # Step 3: Anisotropic refinement (if conditions are met)
            resolution = initial_results.get("resolution", 1.0)
            completeness = initial_results.get("completeness", 0.0)

            resolution_threshold = options.get(self.RESOLUTION_THRESHOLD) or self.DEFAULT_RESOLUTION_THRESHOLD
            completeness_threshold = options.get(self.COMPLETENESS_THRESHOLD) or self.DEFAULT_COMPLETENESS_THRESHOLD

            if resolution < resolution_threshold and completeness > completeness_threshold:
                self._apply_anisotropic_refinement(ins_file)
                initial_results = self._run_shelxl_command(ins_file, working_dir, options)

            # Step 4: Place hydrogen atoms (if heavy atom completeness is sufficient)
            if completeness > 0.95:
                self._place_hydrogen_atoms(ins_file)
                initial_results = self._run_shelxl_command(ins_file, working_dir, options)

            # Step 5: Final refinement cycle with more iterations
            final_cycles = options.get(self.REFINEMENT_CYCLES) or self.DEFAULT_REFINEMENT_CYCLES
            self._set_refinement_cycles(ins_file, final_cycles * 2)  # Double cycles for final refinement
            final_results = self._run_shelxl_command(ins_file, working_dir, options)

            # Step 6: Generate CIF and run validation
            cif_file = self._generate_cif_file(working_dir, ins_file.stem)
            validation_results = {}
            if cif_file and cif_file.exists():
                validation_results = self._run_checkcif_validation(cif_file, options)

            # Step 7: Preserve output files
            output_files = self._cleanup_working_directory(working_dir, preserve_outputs=True)

            # Combine all results
            complete_results = {
                **final_results,
                **validation_results,
                "output_files": output_files,
                "success": True,
                "workflow_completed": True,
            }

            return complete_results

        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "workflow_completed": False,
                "r1_factor": None,
                "wr2_factor": None,
                "gof": None,
                "validation_passed": False,
            }

    def _perform_validation_only(self, ins_file_path: str, hkl_file_path: str, options: Options) -> dict:
        """
        Perform validation only (assumes structure is already refined).

        Args:
            ins_file_path: Path to .ins file
            hkl_file_path: Path to .hkl file
            options: Configuration options

        Returns:
            Dictionary containing validation results
        """
        try:
            # Prepare working directory
            working_dir = self._prepare_working_directory(ins_file_path, hkl_file_path)

            # Generate CIF file
            cif_file = self._generate_cif_file(working_dir, Path(ins_file_path).stem)

            if not cif_file or not cif_file.exists():
                raise RuntimeError("Could not generate CIF file for validation")

            # Run checkCIF validation
            validation_results = self._run_checkcif_validation(cif_file, options)

            # Clean up
            self._cleanup_working_directory(working_dir, preserve_outputs=False)

            return {**validation_results, "success": True, "validation_only": True}

        except Exception as e:
            return {"error": str(e), "success": False, "validation_passed": False, "validation_only": True}

    def _extract_statistics_only(self, ins_file_path: str, hkl_file_path: str, options: Options) -> dict:
        """
        Extract refinement statistics only (assumes structure is already refined).

        Args:
            ins_file_path: Path to .ins file
            hkl_file_path: Path to .hkl file
            options: Configuration options

        Returns:
            Dictionary containing refinement statistics
        """
        try:
            # Prepare working directory
            working_dir = self._prepare_working_directory(ins_file_path, hkl_file_path)

            # Parse existing output files for statistics
            basename = Path(ins_file_path).stem
            stats = self._parse_shelxl_output(working_dir, basename)

            # Clean up
            self._cleanup_working_directory(working_dir, preserve_outputs=False)

            return {**stats, "success": True, "statistics_only": True}

        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "statistics_only": True,
                "r1_factor": None,
                "wr2_factor": None,
                "gof": None,
            }

    def _apply_global_restraints(self, ins_file: Path) -> None:
        """Apply global DFIX restraints to the .ins file."""
        try:
            with open(ins_file, "r") as f:
                content = f.read()

            # Add global restraints before the first atom definition
            lines = content.split("\n")
            new_lines = []
            restraints_added = False

            for line in lines:
                if not restraints_added and (
                    line.strip().startswith("C")
                    or line.strip().startswith("N")
                    or line.strip().startswith("O")
                    or line.strip().startswith("S")
                ):
                    # Add restraints before first atom
                    new_lines.append("REM Global restraints applied due to high R1 factor")
                    new_lines.append("DFIX 1.54 C C")
                    new_lines.append("DFIX 1.39 C C_$")
                    new_lines.append("DANG 2.51 C C")
                    restraints_added = True

                new_lines.append(line)

            # Write back to file
            with open(ins_file, "w") as f:
                f.write("\n".join(new_lines))

        except Exception as e:
            print(f"Warning: Could not apply global restraints: {e}")

    def _apply_anisotropic_refinement(self, ins_file: Path) -> None:
        """Apply anisotropic refinement to non-hydrogen atoms."""
        try:
            with open(ins_file, "r") as f:
                content = f.read()

            # Add ANIS instruction
            lines = content.split("\n")
            new_lines = []
            anis_added = False

            for line in lines:
                if not anis_added and line.strip().startswith("FVAR"):
                    new_lines.append(line)
                    new_lines.append("ANIS")  # Apply anisotropic refinement to all non-H atoms
                    anis_added = True
                else:
                    new_lines.append(line)

            # Write back to file
            with open(ins_file, "w") as f:
                f.write("\n".join(new_lines))

        except Exception as e:
            print(f"Warning: Could not apply anisotropic refinement: {e}")

    def _place_hydrogen_atoms(self, ins_file: Path) -> None:
        """Place hydrogen atoms using AFIX instructions."""
        try:
            with open(ins_file, "r") as f:
                content = f.read()

            # Add hydrogen placement instructions
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                new_lines.append(line)
                # Add AFIX instructions for common carbon environments
                if line.strip().startswith("C") and not line.strip().startswith("CELL"):
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_name = parts[0]
                        new_lines.append(f"AFIX 23 {atom_name}")  # Methyl group

            # Write back to file
            with open(ins_file, "w") as f:
                f.write("\n".join(new_lines))

        except Exception as e:
            print(f"Warning: Could not place hydrogen atoms: {e}")

    def _set_refinement_cycles(self, ins_file: Path, cycles: int) -> None:
        """Set the number of refinement cycles in the .ins file."""
        try:
            with open(ins_file, "r") as f:
                content = f.read()

            # Update L.S. instruction
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                if line.strip().startswith("L.S."):
                    new_lines.append(f"L.S. {cycles}")
                else:
                    new_lines.append(line)

            # Write back to file
            with open(ins_file, "w") as f:
                f.write("\n".join(new_lines))

        except Exception as e:
            print(f"Warning: Could not set refinement cycles: {e}")

    def _generate_cif_file(self, working_dir: Path, basename: str) -> Optional[Path]:
        """Generate CIF file from refined structure."""
        try:
            # Look for existing .cif file first
            cif_file = working_dir / f"{basename}.cif"
            if cif_file.exists():
                return cif_file

            # If no CIF file exists, try to generate one from .res file
            res_file = working_dir / f"{basename}.res"
            if res_file.exists():
                # Simple CIF generation (in practice, you might use SHELXL's CIF generation)
                with open(res_file, "r") as f:
                    res_content = f.read()

                # Create a basic CIF file (this is a simplified version)
                cif_content = f"""
data_{basename}

_audit_creation_method           'Generated from SHELXL .res file'

# Add basic structure information here
# This is a placeholder - in practice, you would use proper CIF generation tools

{res_content}
"""

                with open(cif_file, "w") as f:
                    f.write(cif_content)

                return cif_file

            return None

        except Exception as e:
            print(f"Warning: Could not generate CIF file: {e}")
            return None
