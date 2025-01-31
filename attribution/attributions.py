import shutil
import subprocess
from typing import Any, List
import requests  # type: ignore
import toml  # type: ignore
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


if __name__ == "__main__":
    """
        This script downloads attribution files from the latest release of mloda and compares them using tox.
        The version is extracted from the pyproject.toml file, thus you need the latest pull from main.

        You can view the results with git diff.
    """

    files = ["ATTRIBUTION.md", "THIRD_PARTY_LICENSES.md"]
    base = f"https://github.com/TomKaltofen/mloda/releases/download/{get_version()}/"
    out = "attribution/"

    if version := get_version():
        print(f"Version: {version}")
        download_files(base, files, out)
        remove_tox()
        run_tox()
    else:
        print("Exiting")
