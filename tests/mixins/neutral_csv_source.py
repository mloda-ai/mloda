"""Shared builder of per-call neutral csv ToySourceFG subclasses, kept under tests/mixins/ so leaks are reported."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.provider import SourceMatch
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG


def neutral_csv_group(path: Path) -> type[ToySourceFG]:
    """A fresh ToySourceFG subclass whose neutral form is a csv FileSource written to path (no loaders)."""

    class _NeutralCsvFG(ToySourceFG):
        @classmethod
        def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
            names = sorted(features.get_all_names())
            rows = zip(*(match.access[n] for n in names))
            path.write_text(",".join(names) + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
            return FileSource(path=str(path), format="csv", columns=tuple(names))

    return _NeutralCsvFG
