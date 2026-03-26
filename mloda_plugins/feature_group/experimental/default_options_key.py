import warnings

warnings.warn(
    "Importing DefaultOptionKeys from "
    "'mloda_plugins.feature_group.experimental.default_options_key' is deprecated. "
    "Use 'from mloda.provider import DefaultOptionKeys' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys  # noqa: E402

__all__ = ["DefaultOptionKeys"]
