"""Runtime credential reads isolate selected values from metadata and other hooks."""

from __future__ import annotations

import logging
import pickle  # nosec B403  # Only dumps synthetic metadata.
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import ExtenderHook
from mloda.core.api.plan_lock import plan_content_hash, plan_structure_hash
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Credential, DataAccessCollection, Feature, FeatureName, Options, PluginCollector, mloda

from tests.test_core.test_runtime.test_exception_preservation import _RecordingExtender
from tests.test_core.test_runtime.test_runtime_credential_access import (
    _HANDLE,
    _collection,
    _digest,
    _request,
    RuntimeCredentialDuringPlanning,
    RuntimeCredentialLookup,
    RuntimeCredentialSeed,
)


_SHORT = "Q"
_SHORT_BYTES = b"JZ"
_NESTED_TEXT = "nested-material-2351"
_NESTED_BYTES = b"\x00\xffb5"
_SET_TEXT = "c8!"
_CAUSE_TEXT = "h9$"
_UNREAD = "untouched-material-2351"


def _nested_collection() -> DataAccessCollection:
    return DataAccessCollection(
        credentials={
            _HANDLE: Credential(
                material=[
                    _SHORT,
                    {
                        "detail": (_NESTED_TEXT, _NESTED_BYTES),
                        "pieces": {_SET_TEXT, _NESTED_BYTES},
                        "frozen_pieces": frozenset({_SET_TEXT, _NESTED_BYTES}),
                    },
                ],
                mode=_SHORT_BYTES,
                region=_CAUSE_TEXT,
            ),
            "unread-service": Credential(material=_UNREAD),
        }
    )


def _raise_selected_values(group: type[FeatureGroup]) -> None:
    credential = group.read_credential(_HANDLE)
    assert credential["material"][0] == _SHORT
    # One slot read protects its other leaves without requiring individual field reads.
    message = f"observed {_SHORT} / {_NESTED_TEXT} / {_NESTED_BYTES!r}; untouched {_UNREAD}"
    nested = credential["material"][1]
    raise ValueError(
        message,
        _SHORT_BYTES,
        {
            "detail": (_NESTED_BYTES, _SET_TEXT),
            "pieces": nested["pieces"],
            "frozen_pieces": nested["frozen_pieces"],
        },
    ) from KeyError(_CAUSE_TEXT)


class RuntimeCredentialFailureHooks(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(RuntimeCredentialSeed.get_class_name())}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        if features.get_options_key("failure_hook") == "input":
            _raise_selected_values(cls)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.get_options_key("failure_hook") == "calculate":
            _raise_selected_values(cls)
        return {cls.get_class_name(): ["ready"]}

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        if features.get_options_key("failure_hook") == "output":
            _raise_selected_values(cls)


@pytest.mark.parametrize(
    ("stage", "hook"),
    [
        ("input", ExtenderHook.VALIDATE_INPUT_FEATURE),
        ("calculate", ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE),
        ("output", ExtenderHook.VALIDATE_OUTPUT_FEATURE),
    ],
)
def test_selected_nested_values_are_redacted_before_extenders_observe_errors(
    stage: str, hook: ExtenderHook, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = _RecordingExtender(hook)
    collection = _nested_collection()
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as exc_info:
        mloda.run_all(
            **_request(RuntimeCredentialFailureHooks, collection, failure_hook=stage),
            function_extender={recorder},
        )

    assert recorder.seen
    assert type(exc_info.value.__cause__) is KeyError
    rendered = [
        *recorder.seen,
        str(exc_info.value),
        exc_info.value.args[0],
        repr(exc_info.value.args),
        repr(exc_info.value.__cause__),
    ]
    secrets = (
        _SHORT,
        _NESTED_TEXT,
        _SET_TEXT,
        _CAUSE_TEXT,
        _SHORT_BYTES.decode(),
        repr(_NESTED_BYTES),
        repr(_NESTED_BYTES).replace("\\", "\\\\"),
    )
    for text in rendered:
        for secret in secrets:
            assert secret not in text
    assert _UNREAD in recorder.seen[0]
    assert _UNREAD in str(exc_info.value)
    assert _NESTED_TEXT not in caplog.text
    assert repr(_NESTED_BYTES) not in caplog.text
    assert exc_info.value.args[1] != _SHORT_BYTES
    assert exc_info.value.args[2]["detail"][0] != _NESTED_BYTES
    assert type(exc_info.value.args[2]["pieces"]) is set
    assert type(exc_info.value.args[2]["frozen_pieces"]) is frozenset
    original_nested = collection.credentials[_HANDLE]["material"][1]
    assert original_nested["pieces"] == {_SET_TEXT, _NESTED_BYTES}
    assert original_nested["frozen_pieces"] == frozenset({_SET_TEXT, _NESTED_BYTES})


class RuntimeCredentialFailedLookup(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        try:
            cls.read_credential("missing-service")
        except ValueError:
            raise ValueError(_SHORT, _UNREAD) from None
        raise AssertionError("an unknown handle unexpectedly resolved")


def test_failed_lookup_does_not_enroll_any_unread_values() -> None:
    recorder = _RecordingExtender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
    with pytest.raises(ValueError) as exc_info:
        mloda.run_all(
            **_request(RuntimeCredentialFailedLookup, _nested_collection()),
            function_extender={recorder},
        )

    assert exc_info.value.args == (_SHORT, _UNREAD)
    assert recorder.seen
    assert _SHORT in recorder.seen[0]
    assert _UNREAD in recorder.seen[0]


class RuntimeCredentialEcho(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        credential = cls.read_credential(_HANDLE)
        return {cls.get_class_name(): [credential["material"]]}


class RuntimeCredentialLaterHook(RuntimeCredentialEcho):
    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        # This validator has not read credentials; short literals from calculate are out of scope.
        raise ValueError(data[cls.get_class_name()][0])


@pytest.mark.parametrize("value", [_SHORT, _NESTED_BYTES])
def test_successful_calculation_data_remains_raw(value: str | bytes) -> None:
    collection = DataAccessCollection(credentials={_HANDLE: Credential(material=value)})
    result = mloda.run_all(**_request(RuntimeCredentialEcho, collection))

    assert result == [{RuntimeCredentialEcho.get_class_name(): [value]}]
    assert collection.credentials[_HANDLE]["material"] == value


def test_short_value_masking_ends_before_the_next_hook() -> None:
    collection = DataAccessCollection(credentials={_HANDLE: Credential(material=_SHORT)})
    recorder = _RecordingExtender(ExtenderHook.VALIDATE_OUTPUT_FEATURE)
    with pytest.raises(ValueError) as exc_info:
        mloda.run_all(**_request(RuntimeCredentialLaterHook, collection), function_extender={recorder})

    assert exc_info.value.args == (_SHORT,)
    assert recorder.seen
    assert _SHORT in recorder.seen[0]


_CONCURRENT_VALUES = ("a7!", "b6?")
_OUTER = "l7?"
_INNER = "m8!"


def _assert_outside_runtime() -> None:
    with pytest.raises(RuntimeError, match="(?i)(runtime|hook|active)"):
        FeatureGroup.read_credential(_HANDLE)


class RuntimeCredentialConcurrent(DataCreatorFeatureGroup):
    barrier: threading.Barrier | None = None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        before = cls.read_credential(_HANDLE)["material"]
        assert cls.barrier is not None
        cls.barrier.wait(timeout=10)
        assert cls.read_credential(_HANDLE)["material"] == before
        raise ValueError(f"first {_CONCURRENT_VALUES[0]}, second {_CONCURRENT_VALUES[1]}")


def test_concurrent_runs_with_the_same_handle_isolate_reads_and_error_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RuntimeCredentialConcurrent, "barrier", threading.Barrier(2))
    sessions = [
        mloda.prepare(**_request(RuntimeCredentialConcurrent, _collection(value))) for value in _CONCURRENT_VALUES
    ]
    expected = (f"first ***, second {_CONCURRENT_VALUES[1]}", f"first {_CONCURRENT_VALUES[0]}, second ***")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(session.run) for session in sessions]
        for future, message in zip(futures, expected):
            with pytest.raises(ValueError) as exc_info:
                future.result()
            assert exc_info.value.args == (message,)
    _assert_outside_runtime()


class RuntimeCredentialSimpleFailure(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError(cls.read_credential(_HANDLE)["material"])


class RuntimeCredentialLiteralFailure(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError(_INNER)


class RuntimeCredentialNested(DataCreatorFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        outer = cls.read_credential(_HANDLE)["material"]
        inner_result = mloda.run_all(**_request(RuntimeCredentialLookup, _collection(_INNER)))
        assert inner_result == [{RuntimeCredentialLookup.get_class_name(): [_digest(_INNER)]}]
        with pytest.raises(ValueError) as exc_info:
            mloda.run_all(**_request(RuntimeCredentialSimpleFailure, _collection(_INNER)))
        assert _INNER not in str(exc_info.value)
        with pytest.raises(RuntimeError, match="(?i)(runtime|hook|active)"):
            mloda.prepare(**_request(RuntimeCredentialDuringPlanning, _collection(_INNER)))
        assert cls.read_credential(_HANDLE)["material"] == outer
        raise ValueError(f"outer {outer}, inner {_INNER}")


def test_nested_success_failure_and_preparation_restore_the_outer_hook() -> None:
    with pytest.raises(ValueError) as exc_info:
        mloda.run_all(**_request(RuntimeCredentialNested, _collection(_OUTER)))

    assert exc_info.value.args == (f"outer ***, inner {_INNER}",)
    _assert_outside_runtime()


def test_prepared_sessions_can_repeat_after_other_runs_and_failures() -> None:
    first = mloda.prepare(**_request(RuntimeCredentialLookup, _collection(_OUTER)))
    second = mloda.prepare(**_request(RuntimeCredentialLookup, _collection(_INNER)))
    for session, value in ((first, _OUTER), (second, _INNER), (first, _OUTER)):
        result = session.run()
        assert result == [{RuntimeCredentialLookup.get_class_name(): [_digest(value)]}]
        _assert_outside_runtime()

    with pytest.raises(ValueError) as exc_info:
        mloda.run_all(**_request(RuntimeCredentialSimpleFailure, _collection(_INNER)))
    assert _INNER not in str(exc_info.value)
    _assert_outside_runtime()

    with pytest.raises(ValueError) as unread_error:
        mloda.run_all(**_request(RuntimeCredentialLiteralFailure, _collection(_INNER)))
    assert unread_error.value.args == (_INNER,)
    assert first.run() == [{RuntimeCredentialLookup.get_class_name(): [_digest(_OUTER)]}]
    _assert_outside_runtime()


def _stream(entry: str, request: dict[str, Any]) -> Any:
    if entry == "stream_all":
        return mloda.stream_all(**request)
    return mloda.prepare(**request).stream_run()


@pytest.mark.parametrize("entry", ["stream_all", "stream_run"])
def test_interleaved_streams_restore_scope_at_yield_close_and_exhaustion(entry: str) -> None:
    first = _stream(entry, _request(RuntimeCredentialLookup, _collection(_OUTER)))
    second = _stream(entry, _request(RuntimeCredentialLookup, _collection(_INNER)))
    try:
        _assert_outside_runtime()
        assert next(first) == {RuntimeCredentialLookup.get_class_name(): [_digest(_OUTER)]}
        _assert_outside_runtime()
        assert next(second) == {RuntimeCredentialLookup.get_class_name(): [_digest(_INNER)]}
        _assert_outside_runtime()
        first.close()
        assert list(second) == []
        _assert_outside_runtime()
    finally:
        first.close()
        second.close()

    with pytest.raises(ValueError) as exc_info:
        mloda.run_all(**_request(RuntimeCredentialLiteralFailure, _collection(_INNER)))
    assert exc_info.value.args == (_INNER,)


class RuntimeCredentialStreamFailure(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return {Feature(RuntimeCredentialLookup.get_class_name())}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise ValueError(cls.read_credential(_HANDLE)["material"])


@pytest.mark.parametrize("entry", ["stream_all", "stream_run"])
def test_stream_error_after_a_yield_is_redacted_and_releases_scope(entry: str) -> None:
    request = _request(RuntimeCredentialStreamFailure, _collection(_INNER))
    request["features"] = [
        Feature(RuntimeCredentialLookup.get_class_name()),
        Feature(RuntimeCredentialStreamFailure.get_class_name()),
    ]
    request["plugin_collector"] = PluginCollector.enabled_feature_groups(
        {RuntimeCredentialLookup, RuntimeCredentialStreamFailure}
    )
    stream = _stream(entry, request)
    try:
        assert next(stream) == {RuntimeCredentialLookup.get_class_name(): [_digest(_INNER)]}
        _assert_outside_runtime()
        with pytest.raises(ValueError) as exc_info:
            next(stream)
        assert _INNER not in str(exc_info.value)
        _assert_outside_runtime()
    finally:
        stream.close()

    with pytest.raises(ValueError) as unread_error:
        mloda.run_all(**_request(RuntimeCredentialLiteralFailure, _collection(_INNER)))
    assert unread_error.value.args == (_INNER,)


class RuntimeCredentialMetadata(DataCreatorFeatureGroup):
    feature_sets: list[FeatureSet] = []

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        before = (hash(features.options), tuple(hash(feature) for feature in features.get_sorted_features()))
        credential = cls.read_credential(features.get_options_key("credential_handle"))
        after = (hash(features.options), tuple(hash(feature) for feature in features.get_sorted_features()))
        assert after == before
        cls.feature_sets.append(features)
        return {cls.get_class_name(): [_digest(credential["material"])]}


def test_rotating_values_does_not_change_feature_or_plan_identity_or_enter_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RuntimeCredentialMetadata, "feature_sets", [])
    values = ("first-rotated-material-2351", "second-rotated-material-2351")
    identities: list[tuple[int, int, str, str]] = []

    for value in values:
        request = _request(RuntimeCredentialMetadata, _collection(value), credential_handle=_HANDLE)
        feature = request["features"][0]
        initial_identity = (hash(feature), hash(feature.options))
        session = mloda.prepare(**request)
        plan = session.resolved_plan()
        result = session.run()
        assert result == [{RuntimeCredentialMetadata.get_class_name(): [_digest(value)]}]
        assert (hash(feature), hash(feature.options)) == initial_identity
        assert session.runner is not None

        feature_set = RuntimeCredentialMetadata.feature_sets[-1]
        metadata = (
            feature,
            feature.options,
            feature_set,
            feature_set.options,
            session.plan_context,
            session.runner.cfw_register.get_run_context(),
            plan,
            mloda.explain(**request),
        )
        serialized = pickle.dumps(metadata)
        for secret in values:
            assert secret.encode() not in serialized
        identities.append((*initial_identity, plan_structure_hash(plan), plan_content_hash(plan)))

    assert len(RuntimeCredentialMetadata.feature_sets) == 2
    assert identities[0] == identities[1]
    assert len(identities[0][2]) == len(identities[0][3]) == 64
