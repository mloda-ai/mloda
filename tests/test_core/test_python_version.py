import sys
import unittest


class TestPythonVersion(unittest.TestCase):
    def test_python_version(self) -> None:
        """
        This test is to make it easy to see which python version runs, when doing tox.
        """
        self.assertEqual(sys.version_info.major, 3)
        self.assertEqual(sys.version_info.minor, 10)
