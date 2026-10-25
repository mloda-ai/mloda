import importlib
import pickle  # nosec B403

import pytest

from mloda.core.abstract_plugins.components.validators.missing_input_validator import MissingInputFeatureError


def _error(missing: tuple[str, ...] = ("x",), available: tuple[str, ...] = ("a", "b")) -> MissingInputFeatureError:
    return MissingInputFeatureError(
        feature_group="MyGroup", feature="my_feature", missing_inputs=missing, available_columns=available
    )


class TestMissingInputFeatureError:
    def test_is_a_value_error_with_a_stable_code(self) -> None:
        error = _error()

        assert isinstance(error, ValueError)
        assert MissingInputFeatureError.code == "missing_input_feature"

    def test_message_names_group_feature_missing_inputs_and_columns(self) -> None:
        message = str(_error(missing=("x", "y"), available=("a", "b")))

        for expected in ("MyGroup", "my_feature", "x", "y", "a", "b"):
            assert expected in message

    def test_pickle_round_trip_keeps_type_and_fields(self) -> None:
        restored = pickle.loads(pickle.dumps(_error(missing=("x", "y"))))  # nosec B301

        assert type(restored) is MissingInputFeatureError
        assert restored.feature_group == "MyGroup"
        assert restored.feature == "my_feature"
        assert restored.missing_inputs == ("x", "y")
        assert restored.available_columns == ("a", "b")
        assert str(restored) == str(_error(missing=("x", "y")))

    def test_join_hint_names_the_renamed_columns(self) -> None:
        message = str(_error(missing=("x",), available=("x_x", "x_y", "other")))

        assert "x_x" in message
        assert "x_y" in message
        assert "join" in message

    def test_no_join_hint_without_suffixed_columns(self) -> None:
        assert "join" not in str(_error(missing=("x",), available=("a", "b")))

    @pytest.mark.parametrize("module", ["mloda.user", "mloda.provider", "mloda.steward"])
    def test_exported_from_the_public_facades(self, module: str) -> None:
        facade = importlib.import_module(module)

        assert facade.MissingInputFeatureError is MissingInputFeatureError
        assert "MissingInputFeatureError" in facade.__all__
