"""FeatureGroupStep must not upload a dataset that run_calculation already uploaded.

``ComputeFramework.run_calculation`` replaces ``self.data`` with the object id
string returned by ``upload_finished_data`` once a dataset is complete. If
``FeatureGroupStep.execute`` then uploads again it hands that string to
``FlightServer.upload_table``, which fails on ``table.schema``.

Latent in-tree: reaching it needs a step with ``need_to_upload`` set *and* the
``run_calculation`` upload branch taken, and in-tree plans always leave
unconsumed children on marked steps so the branch early-returns first. These
tests drive ``execute`` directly to pin the guard regardless.
"""

from typing import Any, Optional
from unittest.mock import MagicMock

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.user import Feature, Options


class _UploadFeatureGroup(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _step(*, need_to_upload: bool = True) -> FeatureGroupStep:
    features = MagicMock(spec=FeatureSet)
    features.features = set()
    step = FeatureGroupStep(
        feature_group=_UploadFeatureGroup,
        features=features,
        required_uuids=set(),
        compute_framework=MagicMock(),
    )
    step.need_to_upload = need_to_upload
    return step


def _cfw(data: Any, object_ids: list[str]) -> MagicMock:
    """A compute framework whose ``data``/``object_ids`` drive the real predicate."""
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework

    cfw = MagicMock()
    cfw.data = data
    cfw.object_ids = object_ids
    # Bind the real predicate so the test exercises production logic, not a mock.
    cfw.data_is_uploaded_object_id = lambda: ComputeFramework.data_is_uploaded_object_id(cfw)
    return cfw


def _execute(step: FeatureGroupStep, cfw: MagicMock, calculated: Any) -> Any:
    cfw_register = MagicMock()
    cfw_register.get_location.return_value = "grpc://localhost:1234"
    cfw_register.get_runtime_artifacts.return_value = None
    step.run_calculate_feature = MagicMock(return_value=calculated)  # type: ignore[method-assign]
    step.save_artifact = MagicMock()  # type: ignore[method-assign]
    return step.execute(cfw_register, cfw), cfw_register


def test_second_upload_is_skipped_when_data_is_already_an_object_id() -> None:
    """The regression: run_calculation already uploaded and returned the id."""
    object_id = "8f14e45f-ea6d-4b3a-9c2f-000000000001"
    cfw = _cfw(data=object_id, object_ids=[object_id])

    result, cfw_register = _execute(_step(), cfw, calculated=object_id)

    cfw.upload_finished_data.assert_not_called()
    # The dataset IS uploaded, so the flyway registration must still happen.
    cfw_register.add_uuid_flyway_datasets.assert_called_once()
    assert result == object_id


def test_upload_still_happens_for_a_real_table() -> None:
    """The guard must not suppress the normal path."""
    table = {"col": [1, 2, 3]}  # stands in for a framework-native table
    cfw = _cfw(data=table, object_ids=[])

    _result, cfw_register = _execute(_step(), cfw, calculated=table)

    cfw.upload_finished_data.assert_called_once_with("grpc://localhost:1234")
    cfw_register.add_uuid_flyway_datasets.assert_called_once()


def test_a_string_that_was_never_uploaded_still_uploads() -> None:
    """Only an id this framework actually uploaded counts.

    A feature group is free to compute a plain string; that is data, not an
    already-uploaded dataset, so it must not be mistaken for one.
    """
    cfw = _cfw(data="not-an-object-id", object_ids=["8f14e45f-ea6d-4b3a-9c2f-000000000001"])

    _execute(_step(), cfw, calculated="not-an-object-id")

    cfw.upload_finished_data.assert_called_once()


def test_no_upload_when_the_step_is_not_marked() -> None:
    cfw = _cfw(data={"col": [1]}, object_ids=[])

    _execute(_step(need_to_upload=False), cfw, calculated={"col": [1]})

    cfw.upload_finished_data.assert_not_called()


def test_predicate_is_false_for_non_string_data() -> None:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework

    for data in ({"col": [1]}, None, 42, ["8f14e45f"]):
        cfw = _cfw(data=data, object_ids=["8f14e45f"])
        assert ComputeFramework.data_is_uploaded_object_id(cfw) is False
