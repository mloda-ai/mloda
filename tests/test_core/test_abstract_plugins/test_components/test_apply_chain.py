"""ONE shared chain-application helper on ComputeFrameworkTransformer.

Contract:

  * ``ComputeFrameworkTransformer.apply_chain(from_framework, to_framework, chain, data,
    connection)`` walks a transformation chain: every intermediate hop resolves its target
    framework from the transformer map, the LAST hop targets ``to_framework`` directly, and
    each transformer is applied in sequence.
  * A chain that names a transformer with NO matching edge from the current framework in
    the transformer map raises a CLEAR ``KeyError``/``ValueError`` naming the transformer
    and the framework, never an ``UnboundLocalError``.
  * ``TransformFrameworkStep.transform`` DELEGATES chain application to ``apply_chain``
    instead of re-implementing the walk inline.

All frameworks and transformers here are synthetic and defined inside a factory function,
so nothing leaks into the global BaseTransformer subclass discovery or the compute
framework enumeration of other tests. No FileSource / file IO involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.user import ParallelizationMode

_ChainEnv = tuple[
    type[Any],  # _FwA
    type[Any],  # _FwB
    type[Any],  # _FwC
    type[BaseTransformer],  # A <-> B
    type[BaseTransformer],  # B <-> C
    ComputeFrameworkTransformer,
]


def _chain_env() -> _ChainEnv:
    """Build a synthetic 2-hop environment A -> B -> C with recording transformers.

    Each transformer appends a tag to the (list) payload, so application ORDER is
    observable in the final result. The transformer map contains only the synthetic
    edges, mirroring how ``add`` registers both directions of a pair.
    """

    class _FwA:
        pass

    class _FwB:
        pass

    class _FwC:
        pass

    class _TransAToB(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return _FwA

        @classmethod
        def other_framework(cls) -> Any:
            return _FwB

        @classmethod
        def import_fw(cls) -> None:
            raise ImportError("synthetic transformer, never auto-registered")

        @classmethod
        def import_other_fw(cls) -> None:
            raise ImportError("synthetic transformer, never auto-registered")

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return [*data, "A->B"]

        @classmethod
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            return [*data, "B->A"]

    class _TransBToC(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return _FwB

        @classmethod
        def other_framework(cls) -> Any:
            return _FwC

        @classmethod
        def import_fw(cls) -> None:
            raise ImportError("synthetic transformer, never auto-registered")

        @classmethod
        def import_other_fw(cls) -> None:
            raise ImportError("synthetic transformer, never auto-registered")

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return [*data, "B->C"]

        @classmethod
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            return [*data, "C->B"]

    transformer = ComputeFrameworkTransformer()
    transformer.transformer_map = {
        (_FwA, _FwB): _TransAToB,
        (_FwB, _FwA): _TransAToB,
        (_FwB, _FwC): _TransBToC,
        (_FwC, _FwB): _TransBToC,
    }
    return _FwA, _FwB, _FwC, _TransAToB, _TransBToC, transformer


class TestApplyChain:
    def test_multi_hop_chain_applies_transformers_in_order(self) -> None:
        """A 2-hop chain A -> B -> C resolves the intermediate target (B) from the map and
        applies both transformers in sequence.
        """
        fw_a, _fw_b, fw_c, trans_ab, trans_bc, transformer = _chain_env()

        result = transformer.apply_chain(fw_a, fw_c, [trans_ab, trans_bc], ["start"], None)

        assert result == ["start", "A->B", "B->C"]

    def test_single_hop_chain_targets_to_framework_directly(self) -> None:
        """A single-element chain transforms straight to ``to_framework`` (no intermediate
        resolution involved).
        """
        fw_a, fw_b, _fw_c, trans_ab, _trans_bc, transformer = _chain_env()

        result = transformer.apply_chain(fw_a, fw_b, [trans_ab], ["start"], None)

        assert result == ["start", "A->B"]

    def test_inconsistent_chain_raises_clear_error_naming_transformer_and_framework(self) -> None:
        """A chain whose first transformer has NO edge from the current framework must raise
        a clear KeyError/ValueError naming the transformer and the framework, and must never
        surface an ``UnboundLocalError`` from an unbound intermediate target.

        The chain starts at ``_FwA`` but names the B <-> C transformer, which has no edge
        from ``_FwA`` in the map: the registry and the chain are inconsistent.
        """
        fw_a, _fw_b, fw_c, _trans_ab, trans_bc, transformer = _chain_env()

        with pytest.raises((KeyError, ValueError)) as excinfo:
            transformer.apply_chain(fw_a, fw_c, [trans_bc, trans_bc], ["start"], None)

        message = str(excinfo.value)
        assert "_TransBToC" in message, message
        assert "_FwA" in message, message


class TestTransformFrameworkStepDelegatesToApplyChain:
    def test_transform_delegates_chain_application_to_apply_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``TransformFrameworkStep.transform`` hands the resolved chain to
        ``ComputeFrameworkTransformer.apply_chain`` instead of walking it inline.
        """
        fw_a, fw_b, _fw_c, trans_ab, _trans_bc, _transformer = _chain_env()

        # Local ComputeFramework subclasses so nothing is discovered by other tests'
        # framework enumeration.
        class _CfwFrom(ComputeFramework):
            @classmethod
            def expected_data_framework(cls) -> Any:
                return fw_a

        class _CfwTo(ComputeFramework):
            @classmethod
            def expected_data_framework(cls) -> Any:
                return fw_b

        step = TransformFrameworkStep(
            from_framework=_CfwFrom,
            to_framework=_CfwTo,
            required_uuids=set(),
            from_feature_group=FeatureGroup,
            to_feature_group=FeatureGroup,
        )
        step.transformer.transformer_map = {(fw_a, fw_b): trans_ab, (fw_b, fw_a): trans_ab}

        recorded: dict[str, Any] = {}

        def _spy(
            self: ComputeFrameworkTransformer,
            from_framework: type[Any],
            to_framework: type[Any],
            chain: list[type[BaseTransformer]],
            data: Any,
            connection: Any,
        ) -> Any:
            recorded["args"] = (from_framework, to_framework, list(chain), data, connection)
            return "CHAIN_APPLIED"

        monkeypatch.setattr(ComputeFrameworkTransformer, "apply_chain", _spy)

        cfw = _CfwTo(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        result = step.transform(cfw, ["payload"], set())

        assert result == "CHAIN_APPLIED"
        assert recorded["args"] == (fw_a, fw_b, [trans_ab], ["payload"], cfw.framework_connection_object)
