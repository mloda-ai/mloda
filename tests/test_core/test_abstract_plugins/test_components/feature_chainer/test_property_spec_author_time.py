"""Author-time type-error pins for ``PropertySpec`` and ``property_spec`` (issue #694 DoD).

Each invalid construction below is a direct typed call carrying the specific ``# type: ignore[code]``
it triggers. Because the tox gate runs mypy --strict with ``warn_unused_ignores``, an ignore that
stops being needed fails the gate: the ignores themselves pin that mypy catches each mistake, while
pytest pins the matching runtime rejection.

The builder mirrors every case: most in-repo specs and every docs example are written with
``property_spec``, so the guarantee only holds if its parameters are as narrow as the fields.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, property_spec


class TestAuthorTimeTypeErrors:
    """One invalid construction per test; mypy flags it at author time, runtime rejects it too."""

    def test_misspelled_field_is_a_type_error_and_a_runtime_type_error(self) -> None:
        with pytest.raises(TypeError):
            PropertySpec("x", strict_validaton=True)  # type: ignore[call-arg]

    def test_non_bool_flag_is_a_type_error_and_rejected_at_runtime(self) -> None:
        with pytest.raises(ValueError):
            PropertySpec("x", strict_validation="yes")  # type: ignore[arg-type]

    def test_non_callable_validator_is_a_type_error_and_rejected_at_runtime(self) -> None:
        with pytest.raises(ValueError):
            PropertySpec("x", strict_validation=True, element_validator=123)  # type: ignore[arg-type]

    def test_str_allowed_values_is_a_type_error_and_rejected_at_runtime(self) -> None:
        with pytest.raises(ValueError):
            PropertySpec("x", allowed_values="add")  # type: ignore[arg-type]


class TestAuthorTimeTypeErrorsThroughTheBuilder:
    """The same four mistakes through ``property_spec``, whose parameters mirror the fields."""

    def test_misspelled_keyword_is_a_type_error_and_a_runtime_type_error(self) -> None:
        with pytest.raises(TypeError):
            property_spec("x", strct=True)  # type: ignore[call-arg]

    def test_non_bool_strict_is_a_type_error_and_rejected_at_runtime(self) -> None:
        with pytest.raises(ValueError):
            property_spec("x", strict="yes")  # type: ignore[arg-type]

    def test_non_callable_validator_is_a_type_error_and_rejected_at_runtime(self) -> None:
        with pytest.raises(ValueError):
            property_spec("x", strict=True, element_validator=123)  # type: ignore[arg-type]

    def test_str_allowed_values_is_a_type_error_and_rejected_at_runtime(self) -> None:
        with pytest.raises(ValueError):
            property_spec("x", allowed_values="add")  # type: ignore[arg-type]
