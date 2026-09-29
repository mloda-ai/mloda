"""A framework overriding only the private _extract_column_names must not silently reuse an inherited
public extract_column_names; the public classmethod raises and names the missing override.
Frameworks are defined per test and unavailable, so plugin discovery never selects them.
"""

from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


def _frame() -> Any:
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"a": [1], "b": [2]})


def test_builtin_subclass_overriding_only_private_raises_on_public() -> None:
    class LegacyPrivateOnlyPandasFramework(PandasDataFrame):
        @staticmethod
        def is_available() -> bool:
            return False

        def _extract_column_names(self, data: Any) -> set[str]:
            return {"custom"}

    with pytest.raises(NotImplementedError) as exc_info:
        LegacyPrivateOnlyPandasFramework.extract_column_names(_frame())
    message = str(exc_info.value)
    assert "LegacyPrivateOnlyPandasFramework" in message
    assert "_extract_column_names" in message
    assert "extract_column_names classmethod" in message


def test_direct_subclass_overriding_only_private_names_the_private_override() -> None:
    class LegacyPrivateOnlyDirectFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

        def _extract_column_names(self, data: Any) -> set[str]:
            return {"custom"}

    with pytest.raises(NotImplementedError) as exc_info:
        LegacyPrivateOnlyDirectFramework.extract_column_names({"a": [1]})
    message = str(exc_info.value)
    assert "LegacyPrivateOnlyDirectFramework" in message
    assert "_extract_column_names" in message


def test_subclass_without_overrides_inherits_public_extract_column_names() -> None:
    class PlainPandasSubclassFramework(PandasDataFrame):
        @staticmethod
        def is_available() -> bool:
            return False

    assert PlainPandasSubclassFramework.extract_column_names(_frame()) == {"a", "b"}


def test_descendant_defining_public_classmethod_uses_it() -> None:
    class LegacyParentFramework(PandasDataFrame):
        @staticmethod
        def is_available() -> bool:
            return False

        def _extract_column_names(self, data: Any) -> set[str]:
            return {"custom"}

    class FixedDescendantFramework(LegacyParentFramework):
        @classmethod
        def extract_column_names(cls, data: Any) -> set[str]:
            return {"fixed"}

    assert FixedDescendantFramework.extract_column_names(_frame()) == {"fixed"}
