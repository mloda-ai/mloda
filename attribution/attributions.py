import os
import shutil
import subprocess  # nosec
import sys
from typing import Any
from urllib.request import urlopen

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def download_files(base_url: str, files: list[str], output_dir: str = ".") -> None:
    """Downloads files from a given base URL."""
    for f in files:
        url = f"{base_url}{f}"
        with urlopen(url) as response:  # nosec
            with open(os.path.join(output_dir, f), "wb") as out:
                while chunk := response.read(8192):
                    out.write(chunk)
        print(f"Downloaded {f} successfully.")


def get_version(path: str = "pyproject.toml") -> Any:
    """Extracts the version from a pyproject.toml file."""
    with open(path, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def remove_tox() -> bool:
    """Removes the .tox directory, if it exists"""
    tox_dir = ".tox"
    if os.path.exists(tox_dir):
        print(f"Removing existing {tox_dir} directory...")
        shutil.rmtree(tox_dir)
        print(f"{tox_dir} directory removed successfully.")
    else:
        print(f"{tox_dir} does not exist. Continuing.")
    return True


def run_tox() -> bool:
    """Executes the tox command"""
    print("Running tox command...")
    subprocess.run(["tox"], check=True)
    print("tox command executed successfully.")
    return True


def add_file_to_git(files: list[str], out: str) -> None:
    """Stages the given file for commit using git add"""
    for file in files:
        file_path = os.path.join(out, file)
        print(f"Adding {file_path} to git staging area.")
        subprocess.run(["git", "add", file_path], check=True)
        print(f"{file_path} was successfully added to the git staging area.")


if __name__ == "__main__":
    """
        This script downloads attribution files from the latest release of mloda and compares them using tox.
        The version is extracted from the pyproject.toml file, thus you need the latest pull from main.

        You can view the results with git diff.
    """

    files = ["THIRD_PARTY_LICENSES.md"]
    version = get_version()
    print(f"Version: {version}")

    base = f"https://github.com/mloda-ai/mloda/releases/download/{version}/"
    out = "attribution/"

    download_files(base, files, out)
    add_file_to_git(files, out)
    remove_tox()

    os.environ["TOX_WRITE_THIRD_PARTY_LICENSES"] = "true"
    try:
        run_tox()
    finally:
        del os.environ["TOX_WRITE_THIRD_PARTY_LICENSES"]
