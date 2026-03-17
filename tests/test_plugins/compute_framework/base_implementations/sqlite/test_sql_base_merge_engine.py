import inspect
from typing import get_type_hints

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_merge_engine import SqlBaseMergeEngine


def test_join_logic_has_no_jointype_param() -> None:
    sig = inspect.signature(SqlBaseMergeEngine.join_logic)
    assert "jointype" not in sig.parameters, "jointype is an unused parameter and should be removed"


def test_is_empty_data_returns_bool() -> None:
    hints = get_type_hints(SqlBaseMergeEngine.is_empty_data)
    assert hints.get("return") is bool, f"Expected bool return type but got {hints.get('return')}"
