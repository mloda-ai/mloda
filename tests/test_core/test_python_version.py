import sys


class TestPythonVersion:
    def test_python_version(self) -> None:
        """
        This test is to make it easy to see which python version runs, when doing tox.
        """
        assert sys.version_info.major == 3
        assert sys.version_info.minor == 10
