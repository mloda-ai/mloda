import unittest
import importlib.metadata

from mloda_core.abstract_plugins.components.feature_group_version import FeatureGroupVersion
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1


class TestAbstractFeatureGroupVersion(unittest.TestCase):
    def test_version_composite(self) -> None:
        # Patch importlib.metadata.version to a known value.
        original_version = importlib.metadata.version
        importlib.metadata.version = lambda pkg: "1.2.3"  # type: ignore
        try:
            composite = BaseTestFeatureGroup1.version()
            # Expected format: "1.2.3-{module_name}-{hash}"
            expected_prefix = f"1.2.3-{BaseTestFeatureGroup1.__module__}-"
            self.assertTrue(
                composite.startswith(expected_prefix),
                f"Composite version should start with '{expected_prefix}', got '{composite}'",
            )

            # Split the composite string into parts.
            parts = composite.split("-")
            self.assertEqual(len(parts), 3, "Composite version should have three parts separated by '-'")

            # Check that the hash part is 64 hex characters (SHA-256 produces 64 hex digits).
            hash_val = parts[2]
            self.assertEqual(len(hash_val), 64, "Hash length should be 64 characters")
            # Verify that the hash is valid hexadecimal.
            int(hash_val, 16)
        finally:
            # Restore the original importlib.metadata.version function.
            importlib.metadata.version = original_version

    def test_invalid_target_class_for_hash(self) -> None:
        # Calling FeatureGroupVersion.class_source_hash with a class not inheriting from AbstractFeatureGroup should raise a ValueError.
        with self.assertRaises(ValueError):
            FeatureGroupVersion.class_source_hash(str)
