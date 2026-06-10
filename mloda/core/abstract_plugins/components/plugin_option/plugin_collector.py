import os

from mloda.core.abstract_plugins.feature_group import FeatureGroup

STRICT_MODE_ENV_VAR = "MLODA_PLUGIN_REGISTRY_STRICT"
VALID_STRICT_MODES = ("off", "warn", "strict")


def strict_mode_from_env() -> str:
    """Read the strict mode from the environment, defaulting to "off"."""
    mode = os.environ.get(STRICT_MODE_ENV_VAR, "off")
    _validate_strict_mode(mode)
    return mode


def _validate_strict_mode(mode: str) -> None:
    if mode not in VALID_STRICT_MODES:
        raise ValueError(f"Invalid strict mode {mode!r}. Valid modes are: {', '.join(VALID_STRICT_MODES)}.")


class PluginCollector:
    """
    The PluginCollector class is a helper class with the purpose to disable or enable feature groups.

    This class is useful for rapid prototype development, where you want to disable or enable feature groups,
    when the other, competing feature groups are found.

    Further, this class is useful for testing, where you want to test the behavior of the system with different
    feature groups enabled or disabled.

    Use ``set_allow_redefinition()`` to keep only the most recently defined version of each FeatureGroup
    class when duplicates differ in source — useful when iterating on a FeatureGroup definition in a
    Jupyter cell or via ``importlib.reload``, where the old class object survives in
    ``FeatureGroup.__subclasses__()`` and would otherwise raise a "FeatureGroup redefined" error.
    """

    def __init__(self) -> None:
        self.disabled_feature_group_classes: set[type[FeatureGroup]] = set()
        self.enabled_feature_group_classes: set[type[FeatureGroup]] = set()
        self.allow_redefinition: bool = False
        self.strict_mode: str = strict_mode_from_env()

    def set_allow_redefinition(self, allow: bool = True) -> "PluginCollector":
        """Allow keeping the most recently defined class when duplicates differ in source."""
        self.allow_redefinition = allow
        return self

    def set_strict_mode(self, mode: str) -> "PluginCollector":
        """Set the registry strict mode: "off", "warn", or "strict"."""
        _validate_strict_mode(mode)
        self.strict_mode = mode
        return self

    def add_disabled_feature_group_classes(self, feature_group_cls: set[type[FeatureGroup]]) -> None:
        self.disabled_feature_group_classes.update(feature_group_cls)

    def add_enabled_feature_group_classes(self, feature_group_cls: set[type[FeatureGroup]]) -> None:
        self.enabled_feature_group_classes.update(feature_group_cls)

    def applicable_feature_group_class(self, feature_group_cls: type[FeatureGroup]) -> bool:
        if feature_group_cls in self.disabled_feature_group_classes:
            return False

        # If no feature groups are enabled, all feature groups are enabled.
        if len(self.enabled_feature_group_classes) == 0:
            return True

        if feature_group_cls in self.enabled_feature_group_classes:
            return True
        return False

    @staticmethod
    def disabled_feature_groups(
        feature_group_cls: set[type[FeatureGroup]] | type[FeatureGroup],
    ) -> "PluginCollector":
        if not isinstance(feature_group_cls, set):
            feature_group_cls = {feature_group_cls}

        plugin_collector = PluginCollector()
        plugin_collector.add_disabled_feature_group_classes(feature_group_cls)
        return plugin_collector

    @staticmethod
    def enabled_feature_groups(
        feature_group_cls: set[type[FeatureGroup]] | type[FeatureGroup],
    ) -> "PluginCollector":
        if not isinstance(feature_group_cls, set):
            feature_group_cls = {feature_group_cls}

        plugin_collector = PluginCollector()
        plugin_collector.add_enabled_feature_group_classes(feature_group_cls)
        return plugin_collector
