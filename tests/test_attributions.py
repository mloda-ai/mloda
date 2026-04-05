import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from attribution.attributions import add_file_to_git, download_files, get_version, remove_tox, run_tox


class TestGetVersion:
    def test_reads_version_from_valid_pyproject(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "1.2.3"\n')
        assert get_version(str(pyproject)) == "1.2.3"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_version(str(tmp_path / "nonexistent.toml"))

    def test_raises_on_missing_key(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\n")
        with pytest.raises(KeyError):
            get_version(str(pyproject))


class TestDownloadFiles:
    @patch("attribution.attributions.urlopen")
    def test_downloads_file_to_output_dir(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read = MagicMock(side_effect=[b"file content", b""])

        mock_urlopen.return_value = mock_response

        download_files("https://example.com/", ["test.txt"], str(tmp_path))

        mock_urlopen.assert_called_once_with("https://example.com/test.txt")
        assert (tmp_path / "test.txt").read_bytes() == b"file content"

    @patch("attribution.attributions.urlopen")
    def test_downloads_multiple_files(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read = MagicMock(side_effect=[b"content1", b"", b"content2", b""])

        mock_urlopen.return_value = mock_response

        download_files("https://example.com/", ["a.txt", "b.txt"], str(tmp_path))

        assert mock_urlopen.call_count == 2


class TestRemoveTox:
    def test_removes_existing_tox_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        tox_dir = tmp_path / ".tox"
        tox_dir.mkdir()
        (tox_dir / "somefile").touch()

        assert remove_tox() is True
        assert not tox_dir.exists()

    def test_returns_true_when_tox_does_not_exist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert remove_tox() is True


class TestRunTox:
    @patch("attribution.attributions.subprocess.run")
    def test_calls_tox(self, mock_run: MagicMock) -> None:
        assert run_tox() is True
        mock_run.assert_called_once_with(["tox"], check=True)

    @patch("attribution.attributions.subprocess.run", side_effect=subprocess.CalledProcessError(1, "tox"))
    def test_raises_on_tox_failure(self, mock_run: MagicMock) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            run_tox()


class TestAddFileToGit:
    @patch("attribution.attributions.subprocess.run")
    def test_stages_files(self, mock_run: MagicMock) -> None:
        add_file_to_git(["a.md", "b.md"], "output/")
        assert mock_run.call_args_list == [
            call(["git", "add", os.path.join("output/", "a.md")], check=True),
            call(["git", "add", os.path.join("output/", "b.md")], check=True),
        ]

    @patch("attribution.attributions.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git"))
    def test_raises_on_git_failure(self, mock_run: MagicMock) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            add_file_to_git(["file.md"], "output/")
