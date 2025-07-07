import shutil
import subprocess  # nosec
from typing import Any, List
import requests
import toml
import os


def download_files(base_url: str, files: List[str], output_dir: str = ".") -> None:
    """Downloads files from a given base URL."""
    for f in files:
        url = f"{base_url}{f}"
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(os.path.join(output_dir, f), "wb") as out:
                    for chunk in r.iter_content(8192):
                        out.write(chunk)
            print(f"Downloaded {f} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {f}: {e}")


def get_version(path: str = "pyproject.toml") -> Any:
    """Extracts the version from a pyproject.toml file."""
    try:
        return toml.load(path)["project"]["version"]
    except (FileNotFoundError, toml.TomlDecodeError, KeyError) as e:
        print(f"Error reading version: {e}")
        return None


def remove_tox() -> bool:
    """Removes the .tox directory, if it exists"""
    tox_dir = ".tox"
    if os.path.exists(tox_dir):
        print(f"Removing existing {tox_dir} directory...")
        try:
            shutil.rmtree(tox_dir)
            print(f"{tox_dir} directory removed successfully.")
        except OSError as e:
            print(f"Error removing {tox_dir}: {e}")
            return False
    else:
        print(f"{tox_dir} does not exist. Continuing.")
    return True


def run_tox() -> bool:
    """Executes the tox command"""
    print("Running tox command...")
    try:
        subprocess.run(["tox"], check=True)
        print("tox command executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running tox: {e}")
        return False


def add_file_to_git(files: List[str], out: str) -> None:
    """Stages the given file for commit using git add"""
    for file in files:
        file_path = os.path.join(out, file)
        print(f"Adding {file_path} to git staging area.")
        try:
            subprocess.run(["git", "add", file_path], check=True)
            print(f"{file_path} was successfully added to the git staging area.")
        except subprocess.CalledProcessError as e:
            print(f"Error adding {file_path} to git index {e}")


if __name__ == "__main__":
    """
        This script downloads attribution files from the latest release of mloda and compares them using tox.
        The version is extracted from the pyproject.toml file, thus you need the latest pull from main.

        You can view the results with git diff.
    """

    files = ["THIRD_PARTY_LICENSES.md"]
    base = f"https://github.com/mloda-ai/mloda/releases/download/{get_version()}/"
    out = "attribution/"

    try:
        if version := get_version():
            print(f"Version: {version}")
            download_files(base, files, out)
            add_file_to_git(files, out)
            remove_tox()
            os.environ["TOX_WRITE_THIRD_PARTY_LICENSES"] = "true"
            run_tox()
        else:
            print("Exiting")

    finally:
        if "TOX_WRITE_THIRD_PARTY_LICENSES" in os.environ:
            del os.environ["TOX_WRITE_THIRD_PARTY_LICENSES"]
