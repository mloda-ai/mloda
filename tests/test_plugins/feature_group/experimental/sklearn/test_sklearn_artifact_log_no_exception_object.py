"""An unreadable artifact file is logged as text, never as the exception object.

The object would land in LogRecord.args, and a retained record then pins ``exc.__traceback__``,
whose frames hold the loader locals of a dynamically loaded feature group.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pytest

from mloda.user import Feature, Options
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact

ARTIFACT_LOGGER_NAME = SklearnArtifact.__module__


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _artifact_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == ARTIFACT_LOGGER_NAME]


@pytest.fixture
def corrupt_artifact(tmp_path: Path) -> FeatureSet:
    """A storage dir holding one file that matches the artifact pattern but is not a joblib dump."""
    pytest.importorskip("joblib")
    (tmp_path / "sklearn_artifact_broken.joblib").write_bytes(b"not a joblib payload")
    features = FeatureSet()
    features.add(Feature("sklearn_artifact_log_probe", Options({"artifact_storage_path": str(tmp_path)})))
    return features


class TestUnreadableArtifactLogsNoExceptionObject:
    """The contained load failure must reach the log as text only."""

    def test_record_retains_no_exception_object(
        self, caplog: pytest.LogCaptureFixture, corrupt_artifact: FeatureSet
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=ARTIFACT_LOGGER_NAME):
            assert SklearnArtifact.custom_loader(corrupt_artifact) is None

        assert _artifact_records(caplog), "the unreadable file must report at WARNING"
        for record in _artifact_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_message_still_names_the_file_and_the_exception(
        self, caplog: pytest.LogCaptureFixture, corrupt_artifact: FeatureSet
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=ARTIFACT_LOGGER_NAME):
            SklearnArtifact.custom_loader(corrupt_artifact)

        messages = [record.getMessage() for record in _artifact_records(caplog)]
        assert len(messages) == 1, f"exactly one WARNING record reports the failure, got: {messages}"
        assert "sklearn_artifact_broken.joblib" in messages[0], f"the file must stay in the message: {messages[0]}"
        assert "raised" in messages[0], f"the message must name what the load raised: {messages[0]}"
