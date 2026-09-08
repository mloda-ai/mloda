"""TransformFrameworkStep and JoinStep resolve their connection through cfw.ensure_connection(),
not cfw.framework_connection_object / cfw.get_framework_connection_object().
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.provider import ComputeFramework, FeatureGroup


class _FwFrom:
    pass


class _FwTo:
    pass


class _HopTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return _FwFrom

    @classmethod
    def other_framework(cls) -> Any:
        return _FwTo

    @classmethod
    def import_fw(cls) -> None:
        raise ImportError("synthetic transformer, never auto-registered")

    @classmethod
    def import_other_fw(cls) -> None:
        raise ImportError("synthetic transformer, never auto-registered")

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        return data


class _CfwFrom(ComputeFramework):
    @classmethod
    def expected_data_framework(cls) -> Any:
        return _FwFrom


class _CfwTo(ComputeFramework):
    @classmethod
    def expected_data_framework(cls) -> Any:
        return _FwTo


class TestTransformFrameworkStepUsesEnsureConnection:
    def test_transform_passes_ensure_connection_result_to_apply_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        step = TransformFrameworkStep(
            from_framework=_CfwFrom,
            to_framework=_CfwTo,
            required_uuids=set(),
            from_feature_group=FeatureGroup,
            to_feature_group=FeatureGroup,
        )
        step.transformer.transformer_map = {(_FwFrom, _FwTo): _HopTransformer}

        cfw = Mock(spec=ComputeFramework)
        sentinel_connection = object()
        cfw.ensure_connection.return_value = sentinel_connection

        recorded: dict[str, Any] = {}

        def _spy(
            self: ComputeFrameworkTransformer,
            from_framework: type[Any],
            to_framework: type[Any],
            chain: list[type[BaseTransformer]],
            data: Any,
            connection: Any,
        ) -> Any:
            recorded["connection"] = connection
            return "CHAIN_APPLIED"

        monkeypatch.setattr(ComputeFrameworkTransformer, "apply_chain", _spy)

        result = step.transform(cfw, ["payload"], set())

        assert result == "CHAIN_APPLIED"
        assert recorded["connection"] is sentinel_connection
        cfw.ensure_connection.assert_called_once_with()


class TestJoinStepUsesEnsureConnectionForMergeEngine:
    def test_do_merge_data_builds_merge_engine_from_ensure_connection(self) -> None:
        step = JoinStep(
            link=Mock(),
            destination_framework=ComputeFramework,
            source_framework=ComputeFramework,
            required_uuids=set(),
            destination_framework_uuids=set(),
            source_framework_uuids=set(),
        )

        cfw = Mock(spec=ComputeFramework)
        sentinel_connection = object()
        cfw.ensure_connection.return_value = sentinel_connection
        cfw.data = "left-data"

        merge_engine_instance = Mock()
        merge_engine_instance.merge.return_value = "merged"
        merge_engine_class = Mock(return_value=merge_engine_instance)
        cfw.merge_engine.return_value = merge_engine_class

        step._do_merge_data(cfw, "right-data")

        merge_engine_class.assert_called_once_with(sentinel_connection)
        assert cfw.data == "merged"
