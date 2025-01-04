from typing import Any
from setuptools import setup, find_packages
import os


def get_metadata() -> Any:
    """Simple parser for pyproject.toml"""
    with open("pyproject.toml", "r") as f:
        lines = f.readlines()

    # Simple parsing for the specific keys you need
    metadata = {}
    for line in lines:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes if present
            metadata[key] = value

    return metadata["name"], metadata["version"]


name, version = get_metadata()

# We enable with this examples in dev mode
setup(
    name=name,
    version=version,
    packages=find_packages(include=["mloda_core*", "mloda_plugins*"] + (["docs"] if os.getenv("ML_DEV_MODE") else [])),
)
