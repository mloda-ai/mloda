import os
import sys

from mloda.core.core.cfw_manager import leave_parent_working_directory


def test_leave_parent_working_directory_moves_out_of_cwd_and_absolutizes_sys_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(sys, "path", [".", ""])
    expected_path = [os.path.abspath("."), os.path.abspath("")]

    leave_parent_working_directory()

    assert os.getcwd() != str(tmp_path)
    assert sys.path == expected_path


def test_leave_parent_working_directory_is_noop_off_windows(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(sys, "path", [".", ""])

    leave_parent_working_directory()

    assert os.getcwd() == str(tmp_path)
    assert sys.path == [".", ""]
