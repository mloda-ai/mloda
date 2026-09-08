"""Connection-seam tests for SparkFramework.open_connection() and ensure_connection().

SparkSession.builder (a class-level descriptor on the real pyspark class) is patched so no
JVM starts and tests stay well under the pytest timeout.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from mloda.user import ConnectionSpec, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework

try:
    from pyspark.sql import SparkSession

    PYSPARK_IMPORTABLE = True
except ImportError:
    SparkSession = None
    PYSPARK_IMPORTABLE = False

# Only the import needs to be available; unlike the shared conftest's PYSPARK_AVAILABLE,
# no JAVA_HOME / JVM is required here since SparkSession.builder is always mocked.
pytestmark = pytest.mark.skipif(not PYSPARK_IMPORTABLE, reason="PySpark is not installed.")


def _mock_builder(session: Any) -> Any:
    """A builder mock whose appName/master/config chain to itself; getOrCreate() returns `session`."""
    builder = Mock()
    builder.appName.return_value = builder
    builder.master.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = session
    return builder


def test_set_framework_connection_object_none_raises() -> None:
    fw = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    with pytest.raises(ValueError, match="SparkSession object is required"):
        fw.set_framework_connection_object(None)


def test_open_connection_none_builds_default_session_via_builder() -> None:
    builder = _mock_builder(Mock())
    with patch.object(SparkSession, "builder", new=builder):
        result = SparkFramework.open_connection(None)

    builder.appName.assert_called_once_with("MLoda-Spark-Framework")
    builder.master.assert_called_once_with("local[*]")
    builder.config.assert_any_call("spark.sql.adaptive.enabled", "true")
    builder.config.assert_any_call("spark.sql.adaptive.coalescePartitions.enabled", "true")
    builder.getOrCreate.assert_called_once_with()
    assert result is builder.getOrCreate.return_value


def test_open_connection_with_empty_spec_builds_default_session_via_builder() -> None:
    builder = _mock_builder(Mock())
    with patch.object(SparkSession, "builder", new=builder):
        result = SparkFramework.open_connection(ConnectionSpec(SparkFramework))

    builder.appName.assert_called_once_with("MLoda-Spark-Framework")
    builder.master.assert_called_once_with("local[*]")
    builder.config.assert_any_call("spark.sql.adaptive.enabled", "true")
    builder.config.assert_any_call("spark.sql.adaptive.coalescePartitions.enabled", "true")
    assert result is builder.getOrCreate.return_value


def test_open_connection_with_spec_uses_provided_values() -> None:
    """config merges OVER the two adaptive defaults; it does not replace them."""
    builder = _mock_builder(Mock())
    spec = ConnectionSpec(SparkFramework, app_name="x", master="local[2]", config={"a": "b"})
    with patch.object(SparkSession, "builder", new=builder):
        SparkFramework.open_connection(spec)

    builder.appName.assert_called_once_with("x")
    builder.master.assert_called_once_with("local[2]")
    builder.config.assert_any_call("spark.sql.adaptive.enabled", "true")
    builder.config.assert_any_call("spark.sql.adaptive.coalescePartitions.enabled", "true")
    builder.config.assert_any_call("a", "b")


def test_open_connection_with_unknown_keys_raises_value_error_naming_allowed_keys() -> None:
    builder = _mock_builder(Mock())
    spec = ConnectionSpec(SparkFramework, appName="x")
    with patch.object(SparkSession, "builder", new=builder):
        with pytest.raises(ValueError) as excinfo:
            SparkFramework.open_connection(spec)

    message = str(excinfo.value)
    assert "app_name" in message
    assert "master" in message
    assert "config" in message


def test_ensure_connection_binds_default_session_and_transform_reuses_it() -> None:
    """ensure_connection() must bind the default session via open_connection(None); no bare
    set_framework_connection_object() call is allowed during that resolution."""
    fw = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    session = Mock(spec=SparkSession)
    session.createDataFrame = Mock(return_value=Mock())
    builder = _mock_builder(session)
    wrapped = Mock(wraps=fw.set_framework_connection_object)
    fw.set_framework_connection_object = wrapped  # type: ignore[method-assign]

    with patch.object(SparkSession, "builder", new=builder):
        bound = fw.ensure_connection()

        for call_args in wrapped.call_args_list:
            assert call_args.args or call_args.kwargs, "must not bind via a bare no-argument call"

        assert bound is session

        fw.transform({"a": [1, 2, 3]}, [])

    assert fw.framework_connection_object is session
