"""Chain COMPOSITION does not require pyarrow.

``get_transformation_chain`` composes registered edges via breadth-first shortest path
over ``transformer_map``, so with pyarrow completely absent a 2-hop chain over synthetic
non-pyarrow edges (A -> B -> C) still resolves, and ``apply_chain`` executes it end to
end: a cross-framework move avoiding pyarrow entirely.

The body runs in a subprocess with pyarrow blocked via ``sys.meta_path`` (see
``_pyarrow_blocker.run_blocked``), so the test is meaningful both in the dev venv
(pyarrow installed but blocked) and in the nopyarrow tox env (pyarrow truly absent).
"""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY_COMPOSE_AND_APPLY: str = """
import sys

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)


class _FwA:
    pass


class _FwB:
    pass


class _FwC:
    pass


class _TransAB(BaseTransformer):
    @classmethod
    def framework(cls):
        return _FwA

    @classmethod
    def other_framework(cls):
        return _FwB

    @classmethod
    def import_fw(cls):
        raise ImportError("synthetic transformer, never auto-registered")

    @classmethod
    def import_other_fw(cls):
        raise ImportError("synthetic transformer, never auto-registered")

    @classmethod
    def transform_fw_to_other_fw(cls, data):
        return [*data, "A->B"]

    @classmethod
    def transform_other_fw_to_fw(cls, data, framework_connection_object=None):
        return [*data, "B->A"]


class _TransBC(BaseTransformer):
    @classmethod
    def framework(cls):
        return _FwB

    @classmethod
    def other_framework(cls):
        return _FwC

    @classmethod
    def import_fw(cls):
        raise ImportError("synthetic transformer, never auto-registered")

    @classmethod
    def import_other_fw(cls):
        raise ImportError("synthetic transformer, never auto-registered")

    @classmethod
    def transform_fw_to_other_fw(cls, data):
        return [*data, "B->C"]

    @classmethod
    def transform_other_fw_to_fw(cls, data, framework_connection_object=None):
        return [*data, "C->B"]


registry = ComputeFrameworkTransformer()
# Only the synthetic non-pyarrow edges, both directions per pair, like add() registers.
registry.transformer_map = {
    (_FwA, _FwB): _TransAB,
    (_FwB, _FwA): _TransAB,
    (_FwB, _FwC): _TransBC,
    (_FwC, _FwB): _TransBC,
}

chain = registry.get_transformation_chain(_FwA, _FwC)
if chain is None:
    print("CHAIN_IS_NONE")
    sys.exit(1)
if chain != [_TransAB, _TransBC]:
    print("WRONG_CHAIN:" + repr([t.__name__ for t in chain]))
    sys.exit(1)

result = registry.apply_chain(_FwA, _FwC, chain, ["start"], None)
if result != ["start", "A->B", "B->C"]:
    print("WRONG_RESULT:" + repr(result))
    sys.exit(1)

print("OK:COMPOSED")
sys.exit(0)
"""


@pytest.mark.timeout(30)
def test_chain_composition_and_apply_work_without_pyarrow() -> None:
    """With pyarrow blocked, (A, C) composes to [_TransAB, _TransBC] over synthetic
    edges and ``apply_chain`` executes the move end to end.
    """
    result = run_blocked(_BODY_COMPOSE_AND_APPLY)

    assert "CHAIN_IS_NONE" not in result.stdout, (
        "get_transformation_chain returned None with pyarrow blocked: composition over "
        "registered non-pyarrow edges is not implemented.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0, (
        "Chain composition/application crashed under blocked pyarrow.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "OK:COMPOSED" in result.stdout, (
        f"Expected OK:COMPOSED sentinel. Got stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
