import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from attribution.attributions import (
    add_file_to_git,
    download_files,
    get_version,
    remove_tox,
    run_sync_version_command,
    run_tox,
    update_mloda_version,
)


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


class TestUpdateMlodaVersion:
    def test_updates_only_the_mloda_version_cell(self) -> None:
        content = (
            "| Name  | Version | License    |\n"
            "|-------|---------|------------|\n"
            "| alpha | 1.0.0   | MIT        |\n"
            "| mloda | 0.9.0   | Apache-2.0 |\n"
            "| zeta  | 2.3.4   | BSD        |\n"
        )
        expected = (
            "| Name  | Version | License    |\n"
            "|-------|---------|------------|\n"
            "| alpha | 1.0.0   | MIT        |\n"
            "| mloda | 1.0.0   | Apache-2.0 |\n"
            "| zeta  | 2.3.4   | BSD        |\n"
        )
        assert update_mloda_version(content, "1.0.0") == expected

    def test_widens_version_column_when_new_version_is_longer(self) -> None:
        content = (
            "| Name  | Version | License    |\n"
            "|-------|---------|------------|\n"
            "| alpha | 1.0.0   | MIT        |\n"
            "| mloda | 0.9.0   | Apache-2.0 |\n"
            "| zeta  | 2.3.4   | BSD        |\n"
        )
        expected = (
            "| Name  | Version    | License    |\n"
            "|-------|------------|------------|\n"
            "| alpha | 1.0.0      | MIT        |\n"
            "| mloda | 10.100.100 | Apache-2.0 |\n"
            "| zeta  | 2.3.4      | BSD        |\n"
        )
        assert update_mloda_version(content, "10.100.100") == expected

    def test_shrinks_version_column_when_old_version_was_the_widest_cell(self) -> None:
        content = (
            "| Name  | Version    | License    |\n"
            "|-------|------------|------------|\n"
            "| alpha | 1.0.0      | MIT        |\n"
            "| mloda | 10.100.100 | Apache-2.0 |\n"
            "| zeta  | 2.3.4      | BSD        |\n"
        )
        expected = (
            "| Name  | Version | License    |\n"
            "|-------|---------|------------|\n"
            "| alpha | 1.0.0   | MIT        |\n"
            "| mloda | 1.0.0   | Apache-2.0 |\n"
            "| zeta  | 2.3.4   | BSD        |\n"
        )
        assert update_mloda_version(content, "1.0.0") == expected

    def test_idempotent_when_mloda_already_has_target_version(self) -> None:
        content = (
            "| Name  | Version | License    |\n"
            "|-------|---------|------------|\n"
            "| alpha | 1.0.0   | MIT        |\n"
            "| mloda | 0.11.0  | Apache-2.0 |\n"
        )
        assert update_mloda_version(content, "0.11.0") == content

    def test_raises_value_error_when_mloda_row_is_missing(self) -> None:
        content = (
            "| Name  | Version | License |\n"
            "|-------|---------|---------|\n"
            "| alpha | 1.0.0   | MIT     |\n"
            "| zeta  | 2.3.4   | BSD     |\n"
        )
        with pytest.raises(ValueError, match="mloda"):
            update_mloda_version(content, "1.0.0")

    def test_does_not_match_package_name_that_merely_contains_mloda(self) -> None:
        content = (
            "| Name         | Version | License    |\n"
            "|--------------|---------|------------|\n"
            "| mloda-plugin | 3.3.3   | MIT        |\n"
            "| mloda        | 0.9.0   | Apache-2.0 |\n"
        )
        expected = (
            "| Name         | Version | License    |\n"
            "|--------------|---------|------------|\n"
            "| mloda-plugin | 3.3.3   | MIT        |\n"
            "| mloda        | 1.5.0   | Apache-2.0 |\n"
        )
        assert update_mloda_version(content, "1.5.0") == expected


class TestRunSyncVersionCommand:
    def test_writes_updated_attribution_file_at_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        attribution_dir = tmp_path / "attribution"
        attribution_dir.mkdir()
        original = (
            "| Name  | Version | License    |\n"
            "|-------|---------|------------|\n"
            "| alpha | 1.0.0   | MIT        |\n"
            "| mloda | 0.9.0   | Apache-2.0 |\n"
        )
        attribution_file = attribution_dir / "ATTRIBUTION.md"
        attribution_file.write_text(original)

        run_sync_version_command("2.0.0")

        assert attribution_file.read_text() == update_mloda_version(original, "2.0.0")
