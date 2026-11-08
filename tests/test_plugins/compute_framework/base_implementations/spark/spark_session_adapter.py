"""Mixin that makes the Spark hosts create the session before the adapter borrows it."""

from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from mloda._testing.framework_adapter import FrameworkAdapter, SparkAdapter


class SparkSessionAdapter:
    """Overrides ``adapter`` so the active session exists before the adapter borrows it."""

    adapter_class: ClassVar[type[FrameworkAdapter]] = SparkAdapter

    @pytest.fixture
    def adapter(self, spark_session: Any) -> Iterator[FrameworkAdapter]:
        spark_adapter = self.adapter_class()
        with spark_adapter:
            yield spark_adapter
