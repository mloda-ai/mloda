import warnings

import pytest

from mloda.core.abstract_plugins.components.base_validator import BaseValidator
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook


class _Validator(BaseValidator):
    def validate(self, data):
        pass


class _Extender(Extender):
    def wraps(self):
        return {ExtenderHook.JOIN}

    def __call__(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def test_handle_log_level_warns_deprecation():
    validator = _Validator({}, log_level="warning")
    with pytest.warns(DeprecationWarning, match="handle_log_level"):
        validator.handle_log_level("msg", ValueError("boom"))


def test_handle_log_level_still_raises_on_error():
    validator = _Validator({}, log_level="error")
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            validator.handle_log_level("msg", ValueError("boom"))


def test_feature_options_warns_deprecation():
    with pytest.warns(DeprecationWarning, match="feature_options"):
        assert Extender.feature_options(()) is None
