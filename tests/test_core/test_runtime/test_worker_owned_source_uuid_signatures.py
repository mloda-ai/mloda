"""Signature tests: prepare_tfs_and_joinstep and thread_worker must both accept a worker-owned
source cfw handed on as a UUID, not only a live ComputeFramework instance (or None)."""

import inspect
import typing
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor
from mloda.core.runtime.worker.multiprocessing_worker import worker
from mloda.core.runtime.worker.thread_worker import thread_worker


class TestPrepareTfsAndJoinStepReturnAnnotation:
    """prepare_tfs_and_joinstep's return annotation must name ComputeFramework, UUID and None."""

    def test_return_annotation_names_compute_framework_uuid_and_none(self) -> None:
        # The module uses `from __future__ import annotations`, so the raw annotation is a string.
        raw_annotation = inspect.signature(ComputeFrameworkExecutor.prepare_tfs_and_joinstep).return_annotation

        assert raw_annotation == "ComputeFramework | UUID | None"


class TestThreadWorkerFromCfwAnnotation:
    """thread_worker's from_cfw parameter must accept ComputeFramework, UUID and None."""

    def test_from_cfw_annotation_includes_compute_framework_uuid_and_none(self) -> None:
        annotation = inspect.signature(thread_worker).parameters["from_cfw"].annotation

        assert set(typing.get_args(annotation)) == {ComputeFramework, UUID, type(None)}


class TestWorkerFromCfwAnnotation:
    """worker's from_cfw parameter must accept only UUID and None, never a live ComputeFramework."""

    def test_from_cfw_annotation_is_uuid_or_none(self) -> None:
        # get_type_hints(worker) would also resolve command_queue's multiprocessing.Queue[Any]
        # annotation, which is not subscriptable at runtime; probe just the from_cfw string instead.
        raw_annotation = worker.__annotations__["from_cfw"]

        def _probe(from_cfw: Any) -> None: ...

        _probe.__annotations__["from_cfw"] = raw_annotation
        annotation = typing.get_type_hints(_probe)["from_cfw"]

        assert set(typing.get_args(annotation)) == {UUID, type(None)}
