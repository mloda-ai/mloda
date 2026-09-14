"""Public provider exports for ``FileSource`` / ``InputDataDescriptor`` and ``FileSource``
path normalization.

Contract:

  * Requirement F1: ``from mloda.provider import FileSource`` works.
  * Requirement F2: ``from mloda.provider import InputDataDescriptor`` works.
  * Requirement F3: both names appear in ``mloda.provider.__all__``.
  * Requirement G1: ``FileSource.__post_init__`` coerces ``path`` to ``str``, honoring its
    ``path: str`` annotation (it already coerces ``columns`` to a tuple).
  * Requirement G2 (guard): the descriptor stays frozen and hashable after coercion.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.input_data_descriptor import InputDataDescriptor


class TestProviderExports:
    def test_file_source_is_importable_from_provider(self) -> None:
        """(F1) ``from mloda.provider import FileSource`` resolves to the real class."""
        exported = importlib.import_module("mloda.provider").FileSource
        assert exported is FileSource

    def test_input_data_descriptor_is_importable_from_provider(self) -> None:
        """(F2) ``from mloda.provider import InputDataDescriptor`` resolves to the real class."""
        exported = importlib.import_module("mloda.provider").InputDataDescriptor
        assert exported is InputDataDescriptor

    def test_both_names_in_provider_all(self) -> None:
        """(F3) Both names are advertised in ``mloda.provider.__all__``."""
        assert "FileSource" in provider.__all__
        assert "InputDataDescriptor" in provider.__all__


class TestFileSourcePathCoercion:
    def test_path_is_coerced_to_str(self, tmp_path: Path) -> None:
        """(G1) A ``Path`` passed for ``path`` is normalized to ``str`` in ``__post_init__``."""
        p = tmp_path / "x.csv"
        source = FileSource(path=p, format="csv", columns=["a"])  # type: ignore[arg-type]
        assert isinstance(source.path, str)
        assert source.path == str(p)

    def test_descriptor_stays_frozen_and_hashable_after_coercion(self, tmp_path: Path) -> None:
        """(G2) The coerced descriptor remains frozen and hashable."""
        source = FileSource(path=tmp_path / "x.csv", format="csv", columns=["a"])  # type: ignore[arg-type]

        hash(source)  # must not raise
        with pytest.raises(Exception):
            source.path = str(tmp_path / "y.csv")  # type: ignore[misc]
