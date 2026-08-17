import os
import shutil
import subprocess  # nosec
import sys
from typing import Any
from urllib.request import urlopen


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
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

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


def update_mloda_version(content: str, new_version: str) -> str:
    """Replaces the mloda row's version cell in a pip-licenses markdown table and re-renders the table."""
    lines = [line for line in content.split("\n") if line.strip() != ""]
    if len(lines) < 2:
        raise ValueError("Attribution table content must contain a header row and a separator row.")
    header_line = lines[0]
    data_lines = lines[2:]

    def parse_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header_cells = parse_row(header_line)
    rows = [parse_row(line) for line in data_lines]

    for line, row in zip(data_lines, rows):
        if len(row) != len(header_cells):
            raise ValueError(f"Expected {len(header_cells)} cell(s) per row but found {len(row)} in row: {line!r}")

    mloda_index = None
    for index, row in enumerate(rows):
        if row[0] == "mloda":
            mloda_index = index
            break

    if mloda_index is None:
        raise ValueError("No row with package name 'mloda' found in attribution table.")

    rows[mloda_index][1] = new_version

    all_rows = [header_cells] + rows
    widths = [max(len(row[col]) for row in all_rows) for col in range(len(header_cells))]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    rendered = [render_row(header_cells), separator] + [render_row(row) for row in rows]
    return "\n".join(rendered) + "\n"


def update_third_party_licenses_version(content: str, new_version: str) -> str:
    """Replaces the mloda block's version line in a pip-licenses plain-vertical format document."""
    lines = content.split("\n")

    mloda_index = None
    for index, line in enumerate(lines):
        if line.strip() == "mloda" and (index == 0 or lines[index - 1].strip() == ""):
            mloda_index = index
            break

    if mloda_index is None:
        raise ValueError("No block for package 'mloda' found in third-party licenses content.")

    lines[mloda_index + 1] = new_version
    return "\n".join(lines)


def run_sync_version_command(version: str, path: str = "attribution/ATTRIBUTION.md") -> None:
    """Atomically rewrites the mloda version in the attribution file and, if present, the sibling third-party licenses file."""
    with open(path, "r") as f:
        content = f.read()
    updated = update_mloda_version(content, version)

    third_party_path = os.path.join(os.path.dirname(path), "THIRD_PARTY_LICENSES.md")
    updated_third_party = None
    if os.path.exists(third_party_path):
        with open(third_party_path, "r") as f:
            third_party_content = f.read()
        updated_third_party = update_third_party_licenses_version(third_party_content, version)

    with open(path, "w") as f:
        f.write(updated)

    if updated_third_party is not None:
        with open(third_party_path, "w") as f:
            f.write(updated_third_party)


def run_default_sync_command() -> None:
    """Downloads attribution files from the latest mloda release and compares them using tox."""
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


def main(argv: list[str]) -> None:
    """Dispatches to the default sync workflow or the sync-version subcommand."""
    if len(argv) <= 1:
        run_default_sync_command()
        return

    if argv[1] != "sync-version":
        raise ValueError(f"Unknown subcommand: {argv[1]!r}")

    if len(argv) < 3 or not argv[2].strip():
        raise ValueError("sync-version requires a non-empty version argument.")

    run_sync_version_command(argv[2])


if __name__ == "__main__":
    main(sys.argv)
