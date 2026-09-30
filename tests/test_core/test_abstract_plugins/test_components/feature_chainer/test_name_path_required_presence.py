"""Name-path required-presence is mandatory (issue #769).

A feature that matches on its NAME while a required option key is absent (after declared defaults
and name-capture bindings resolve) is a NON-MATCH. The check is unconditional: there is no warn
mode, no off mode, and the env var ``MLODA_NAME_PATH_REQUIRED_PRESENCE`` is ignored entirely.

On the non-match a WARNING names the owning feature group, the feature name, and every missing key;
it references no env var and no mode. Warnings are filtered by the feature_chain_parser logger name
+ WARNING level + the contractual ``required option(s)`` marker substring.

Exemptions (unchanged): a declared default, ``required_when`` (owned by its own guard), the source
key ``in_features`` (name-satisfied), ``deferred_binding=True``, a key bound by a named capture
``(?P<key>...)``, and an ``allow_explicit_none`` key present as None. The config path is unchanged:
a missing required key is a plain non-match there and ``deferred_binding`` does NOT exempt it.

Every fixture carries an "r769" marker in its class name, keys, and values so it cannot collide with
other feature groups in the global registry. Plain (mixin-free) group fixtures carry "Pgp" instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator, PropertySpec, property_spec
from mloda.user import FeatureName, Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

FEATURE_CHAIN_PARSER_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"

# Retired: tests only set it to prove it is ignored, and assert it never appears in messages.
ENV_VAR = "MLODA_NAME_PATH_REQUIRED_PRESENCE"

# The marker substring the non-match warning is contractually required to contain.
MARKER = "required option(s)"

# A name the fixture patterns recognize: "<source>__<capture>". The capture binds the "carried" key.
NAME_PATH_FEATURE = "src__val_r769"
# A separator-free name no fixture pattern captures, so it falls through to the configuration path.
CONFIG_PATH_FEATURE = "config_only_r769"


def _required_presence_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The name-path required-presence WARNINGs, optionally scoped to one class name.

    Filters by logger name, WARNING level, and the ``required option(s)`` marker substring the
    warning is contractually required to contain.
    """
    records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == FEATURE_CHAIN_PARSER_LOGGER
        and MARKER in record.getMessage()
    ]
    if class_name is not None:
        records = [record for record in records if class_name in record.getMessage()]
    return records


class TestMandatoryEnforcement:
    """A missing required key on the name path is a non-match, unconditionally."""

    def test_missing_required_key_is_non_match(self) -> None:
        """The name identifies the group, one required key is absent -> match returns False."""

        class _MissingR769a(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769a>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769a": PropertySpec("required, carried by the name", context=True),
                "missing_r769a": PropertySpec("required, options-only, absent", context=True),
            }

        # Precondition: the missing key is unconditionally required and name-path relevant.
        missing_spec = _MissingR769a.PROPERTY_MAPPING["missing_r769a"]
        assert FeatureChainParser._can_skip_required_check(missing_spec) is False

        result = _MissingR769a.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is False, "a missing required key on the name path must be a non-match"

    def test_non_match_warning_names_group_feature_and_keys(self, caplog: pytest.LogCaptureFixture) -> None:
        """The non-match WARNING names the class, the feature name, and the missing key; no env var, no mode."""

        class _WarnedR769b(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769b>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769b": PropertySpec("required, carried by the name", context=True),
                "missing_r769b": PropertySpec("required, options-only, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _WarnedR769b.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        warnings = _required_presence_warnings(caplog, "_WarnedR769b")
        assert warnings, "the non-match must be announced with a WARNING naming the class"
        message = warnings[0].getMessage()
        assert NAME_PATH_FEATURE in message, "the warning must name the feature"
        assert "missing_r769b" in message, "the warning must name the missing key"
        assert ENV_VAR not in message, "the warning must not reference the retired env var"
        assert "mode" not in message, "the warning must not reference any mode"
        assert result is False, "the warning accompanies the non-match, it does not replace it"

    def test_multiple_missing_keys_all_named(self, caplog: pytest.LogCaptureFixture) -> None:
        """Every missing required key is a non-match reason and every one is named in the warning."""

        class _MultiR769f(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769f>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769f": PropertySpec("carried by the name", context=True),
                "missing_one_r769f": PropertySpec("required, absent", context=True),
                "missing_two_r769f": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _MultiR769f.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is False, "multiple missing required keys must be a non-match"
        warnings = _required_presence_warnings(caplog, "_MultiR769f")
        assert warnings, "the multi-missing non-match must warn"
        rendered = " ".join(record.getMessage() for record in warnings)
        assert "missing_one_r769f" in rendered
        assert "missing_two_r769f" in rendered

    def test_all_required_keys_satisfied_matches_without_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning and a match when every required key is satisfied by name/default/required_when/options."""

        class _SatisfiedR769c(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769c>\w+)$"
            PROPERTY_MAPPING = {
                # required, satisfied by the name binding
                "carried_r769c": PropertySpec("carried by the name", context=True),
                # declared default -> optional, never flagged
                "defaulted_r769c": PropertySpec("has a declared default", context=True, default="d_r769"),
                # required_when -> owned by its own guard, not this check
                "cond_r769c": PropertySpec(
                    "conditionally required", context=True, default=None, required_when=lambda o: False
                ),
                # required, explicitly present in options
                "present_r769c": PropertySpec("required, present in options", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _SatisfiedR769c.match_feature_group_criteria(
                NAME_PATH_FEATURE, Options(context={"present_r769c": "x_r769"})
            )

        assert result is True
        assert not _required_presence_warnings(caplog, "_SatisfiedR769c"), "no key is missing, so nothing is warned"

    def test_present_key_matches(self) -> None:
        """A required key present in options satisfies the check."""

        class _PresentR769e(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769e>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769e": PropertySpec("carried by the name", context=True),
                "present_r769e": PropertySpec("required, present in options", context=True),
            }

        result = _PresentR769e.match_feature_group_criteria(
            NAME_PATH_FEATURE, Options(context={"present_r769e": "y_r769"})
        )

        assert result is True

    def test_name_bound_key_matches(self) -> None:
        """A required key satisfied purely by a named capture from the feature name matches."""

        class _NameBoundR769j(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769j>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769j": PropertySpec("required, carried by the name", context=True),
            }

        result = _NameBoundR769j.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "a name-bound required key is satisfied by the name"


class TestEnvVarIgnored:
    """MLODA_NAME_PATH_REQUIRED_PRESENCE is retired: no value changes the mandatory non-match."""

    @pytest.mark.parametrize("env_value", ["off", "OFF", "0", "false", "no", "warn", "enforce", "banana"])
    def test_env_value_does_not_change_the_verdict(
        self, env_value: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Former modes, off aliases, and garbage values all change nothing: still a warned non-match."""
        monkeypatch.setenv(ENV_VAR, env_value)

        class _EnvIgnoredR769env(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769env>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769env": PropertySpec("required, carried by the name", context=True),
                "missing_r769env": PropertySpec("required, options-only, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _EnvIgnoredR769env.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is False, f"env value {env_value!r} must be ignored: the non-match is mandatory"
        warnings = _required_presence_warnings(caplog, "_EnvIgnoredR769env")
        assert warnings, f"env value {env_value!r} must not silence the non-match warning"
        assert "missing_r769env" in warnings[0].getMessage()


class TestExemptionsUnchanged:
    """The documented exemptions keep the match and stay silent."""

    def test_deferred_binding_key_matches_without_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """``deferred_binding=True`` exempts the key on the name path: match, no warning."""

        class _DeferredR769d(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769d>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769d": PropertySpec("carried by the name", context=True),
                "deferred_r769d": PropertySpec(
                    "required but bound outside the name", context=True, deferred_binding=True
                ),
            }

        with caplog.at_level(logging.WARNING):
            result = _DeferredR769d.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "deferred_binding=True must keep the match"
        assert not _required_presence_warnings(caplog, "_DeferredR769d")

    def test_required_when_key_owned_by_its_own_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        """A required_when key is gated by its own guard, never by the presence check."""

        class _ReqWhenR769i(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769i>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769i": PropertySpec("optional, carried by the name", context=True, default=None),
                "cond_r769i": PropertySpec(
                    "required only when trigger is present",
                    context=True,
                    default=None,
                    required_when=lambda o: o.get("trigger_r769i") is not None,
                ),
            }

        with caplog.at_level(logging.WARNING):
            result = _ReqWhenR769i.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "_ReqWhenR769i"), (
            "required_when is owned by its own guard, not by the presence check"
        )


class TestInFeaturesExcluded:
    """The check must EXCLUDE ``DefaultOptionKeys.in_features``.

    On the name path the source features come from the name prefix (the ``src`` in ``src__op``), so
    the ``in_features`` key is name-satisfied, never missing. All 12 shipped plugins declare
    ``in_features`` as a NO_DEFAULT spec, so without this exclusion every shipped plugin would be a
    false non-match on every name-path feature.
    """

    def test_in_features_not_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        """op captured + in_features excluded -> match True and NO presence warning."""

        class _InFeaturesR769inf(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769inf$"
            PROPERTY_MAPPING = {
                "op_r769inf": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                DefaultOptionKeys.in_features: PropertySpec("source", context=True, strict_validation=False),
            }

        # Precondition: in_features is unconditionally required as a spec, so the exclusion is what
        # keeps it quiet, not a declared default.
        in_features_spec = _InFeaturesR769inf.PROPERTY_MAPPING[DefaultOptionKeys.in_features]
        assert FeatureChainParser._can_skip_required_check(in_features_spec) is False

        with caplog.at_level(logging.WARNING):
            result = _InFeaturesR769inf.match_feature_group_criteria("src__op_r769inf", Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "_InFeaturesR769inf"), (
            "in_features is name-satisfied by the source prefix and must never be flagged"
        )

    def test_genuine_missing_key_is_non_match_in_features_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Contrast: a genuine missing key (not in_features, not deferred) rejects; in_features stays silent."""

        class _InFeaturesContrastR769inf2(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769inf2$"
            PROPERTY_MAPPING = {
                "op_r769inf2": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                DefaultOptionKeys.in_features: PropertySpec("source", context=True, strict_validation=False),
                "genuine_missing_r769inf2": PropertySpec("required, not captured, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _InFeaturesContrastR769inf2.match_feature_group_criteria("src__op_r769inf2", Options())

        assert result is False, "a genuine missing required key must be a non-match"
        warnings = _required_presence_warnings(caplog, "_InFeaturesContrastR769inf2")
        assert warnings, "the genuine missing key must be warned about"
        rendered = " ".join(record.getMessage() for record in warnings)
        assert "genuine_missing_r769inf2" in rendered, "the genuine missing key must be named"
        assert "in_features" not in rendered, "in_features is name-satisfied and must never be reported missing"


class TestConfigPathUnchanged:
    """The config path keeps its long-standing required-presence rule, with no name-path special cases."""

    def test_config_path_rejects_missing_required(self) -> None:
        """A NO_DEFAULT key absent on the config path is a plain non-match."""

        class _ConfigReqR769g(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769g>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769g": PropertySpec("bound by the name on the name path", context=True),
                "req_r769g": PropertySpec("required, options-only", context=True),
            }

        # CONFIG_PATH_FEATURE carries no separator, so the name never identifies the group: the
        # config path decides, and req_r769g (NO_DEFAULT) is absent -> non-match.
        result = _ConfigReqR769g.match_feature_group_criteria(
            CONFIG_PATH_FEATURE, Options(context={"carried_r769g": "x_r769"})
        )

        assert result is False, "config-path required presence must hold"

    def test_deferred_binding_does_not_exempt_config_path(self) -> None:
        """``deferred_binding=True`` exempts only the name path, never the config path."""

        class _ConfigDeferredR769gd(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769gd>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769gd": PropertySpec("bound by the name on the name path", context=True),
                "req_deferred_r769gd": PropertySpec(
                    "required, deferred on the name path only", context=True, deferred_binding=True
                ),
            }

        result = _ConfigDeferredR769gd.match_feature_group_criteria(
            CONFIG_PATH_FEATURE, Options(context={"carried_r769gd": "x_r769"})
        )

        assert result is False, "deferred_binding must not exempt the required key on the config path"

    def test_name_and_config_paths_agree_on_missing_required(self) -> None:
        """Same fixture, same missing key: BOTH paths are a non-match now."""

        class _UnifiedR769h(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769h>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769h": PropertySpec("bound by the name / provided on the config path", context=True),
                "req_only_r769h": PropertySpec("required, options-only, absent", context=True),
            }

        name_result = _UnifiedR769h.match_feature_group_criteria(NAME_PATH_FEATURE, Options())
        config_result = _UnifiedR769h.match_feature_group_criteria(
            CONFIG_PATH_FEATURE, Options(context={"carried_r769h": "x_r769"})
        )

        assert name_result is False, "the name path rejects the missing required key"
        assert config_result is False, "the config path rejects the same missing required key"


class TestShippedPluginsClean:
    """Shipped plugins stay clean on the name path.

    The ``in_features`` exclusion plus the per-plugin ``deferred_binding`` marks keep representative
    shipped plugins matching plain name-path features without a presence warning.
    """

    def test_aggregated_feature_group_name_match_is_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """AggregatedFeatureGroup matches a string name with no presence warning.

        Its only otherwise-flaggable key is in_features, which the exclusion keeps quiet.
        """
        with caplog.at_level(logging.WARNING):
            result = AggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "AggregatedFeatureGroup"), (
            "in_features exclusion must keep AggregatedFeatureGroup clean on the name path"
        )

    def test_time_window_feature_group_name_match_is_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """TimeWindowFeatureGroup matches a valid string name with no presence warning.

        window_function/window_size/time_unit are name-bound via named captures and in_features is
        excluded, so the name-only match reports nothing.
        """
        with caplog.at_level(logging.WARNING):
            result = TimeWindowFeatureGroup.match_feature_group_criteria("temperature__avg_7_day_window", Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "TimeWindowFeatureGroup"), (
            "name-bound captures + in_features exclusion must keep TimeWindowFeatureGroup clean on the name path"
        )

    @pytest.mark.parametrize(
        ("plugin_cls", "key"),
        [
            (TimeWindowFeatureGroup, "window_size"),
            (TimeWindowFeatureGroup, "time_unit"),
            (DimensionalityReductionFeatureGroup, "dimension"),
            (ClusteringFeatureGroup, "k_value"),
            (ForecastingFeatureGroup, "horizon"),
            (ForecastingFeatureGroup, "time_unit"),
        ],
        ids=lambda v: v if isinstance(v, str) else v.__name__,
    )
    def test_multi_capture_plugin_binds_name_values_instead_of_deferring(self, plugin_cls: Any, key: str) -> None:
        """White-box: a named capture binds the key, so it is not deferred_binding."""
        assert plugin_cls.PROPERTY_MAPPING[key].deferred_binding is False


PGP_KEY = "threshold_pgp"
PGP_ROOT_FEATURE = "pgp_root_feature"
PGP_SUPPORTED_FEATURE = "pgp_supported_feature"
PGP_MATCH_DATA_FEATURE = "pgp_match_data_feature"
PGP_OVERRIDE_FEATURE = "pgp_override_feature"
PGP_MISSING_PREFIX = f"required option(s) {PGP_KEY} are absent after declared defaults and name bindings"


class PlainRootPgp(FeatureGroup):
    """Matches through the DataCreator root rule; the required key has no default."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default")}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({PGP_ROOT_FEATURE})


class PlainClassNamePgp(FeatureGroup):
    """Matches through the class-name rule."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default")}


class PlainPrefixPgp(FeatureGroup):
    """Matches through the class-name-prefix rule."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default")}


class PlainSupportedPgp(FeatureGroup):
    """Matches through the feature_names_supported rule."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {PGP_SUPPORTED_FEATURE}


class PlainMatchDataPgp(FeatureGroup, MatchData):
    """Matches through the MatchData rule, via the feature-scope connection option."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default")}

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        return feature_name == PGP_MATCH_DATA_FEATURE


class PlainOverridePgp(FeatureGroup):
    """Overrides the matcher without delegating to the default one."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default")}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PGP_OVERRIDE_FEATURE


class PlainGroupKeyPgp(FeatureGroup):
    """The required key is a group option, so the reason carries no context remedy."""

    PROPERTY_MAPPING = {PGP_KEY: property_spec("required, no default", context=False)}


class PlainExemptPgp(FeatureGroup):
    """Every key is exempt from the presence rule."""

    PROPERTY_MAPPING = {
        "defaulted_none_pgp": property_spec("optional", default=None),
        "cond_pgp": property_spec("conditionally required", default=None, required_when=lambda o: False),
        "deferred_pgp": property_spec("bound outside the name", deferred_binding=True),
        DefaultOptionKeys.in_features: property_spec("source"),
    }


class PlainExemptPlusRequiredPgp(FeatureGroup):
    """The exempt keys sit next to one genuinely required key."""

    PROPERTY_MAPPING = {
        PGP_KEY: property_spec("required, no default"),
        "defaulted_none_pgp": property_spec("optional", default=None),
        "deferred_pgp": property_spec("bound outside the name", deferred_binding=True),
    }


PLAIN_ENTRY_RULES = [
    pytest.param(PlainRootPgp, PGP_ROOT_FEATURE, {}, id="root_data_creator"),
    pytest.param(PlainClassNamePgp, PlainClassNamePgp.get_class_name(), {}, id="class_name"),
    pytest.param(PlainPrefixPgp, f"{PlainPrefixPgp.prefix()}x", {}, id="class_name_prefix"),
    pytest.param(PlainSupportedPgp, PGP_SUPPORTED_FEATURE, {}, id="feature_names_supported"),
    pytest.param(
        PlainMatchDataPgp,
        PGP_MATCH_DATA_FEATURE,
        {PlainMatchDataPgp.get_class_name(): "connection_pgp"},
        id="match_data",
    ),
    pytest.param(PlainOverridePgp, PGP_OVERRIDE_FEATURE, {}, id="non_delegating_override"),
]


@pytest.fixture
def recorded_rejections() -> Iterator[dict[str, MatchRejection]]:
    """An active rejection window, as the engine opens around a candidate's match call."""
    reasons: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    yield reasons
    MATCH_REJECTION_REASONS.reset(token)


class TestPlainGroupPresence:
    """A plain group (no mixin) with a required key needs it on every matching rule."""

    @pytest.mark.parametrize(("group", "feature_name", "group_options"), PLAIN_ENTRY_RULES)
    def test_missing_required_key_is_non_match(
        self, group: type[FeatureGroup], feature_name: str, group_options: dict[str, Any]
    ) -> None:
        assert group.match_feature_group_criteria(feature_name, Options(group=dict(group_options))) is False

    @pytest.mark.parametrize(("group", "feature_name", "group_options"), PLAIN_ENTRY_RULES)
    def test_present_required_key_matches(
        self, group: type[FeatureGroup], feature_name: str, group_options: dict[str, Any]
    ) -> None:
        options = Options(group=dict(group_options), context={PGP_KEY: "5"})

        assert group.match_feature_group_criteria(feature_name, options) is True

    @pytest.mark.parametrize("value", [0, "", False], ids=["zero", "empty_string", "false"])
    def test_present_falsy_value_satisfies_presence(self, value: Any) -> None:
        options = Options(context={PGP_KEY: value})

        assert PlainRootPgp.match_feature_group_criteria(PGP_ROOT_FEATURE, options) is True

    def test_group_option_satisfies_presence(self) -> None:
        options = Options(group={PGP_KEY: "5"})

        assert PlainGroupKeyPgp.match_feature_group_criteria(PlainGroupKeyPgp.get_class_name(), options) is True

    def test_rejection_reason_is_recorded_with_the_context_remedy(
        self, recorded_rejections: dict[str, MatchRejection]
    ) -> None:
        assert PlainRootPgp.match_feature_group_criteria(PGP_ROOT_FEATURE, Options()) is False

        assert list(recorded_rejections) == [PlainRootPgp.get_class_name()]
        rejection = recorded_rejections[PlainRootPgp.get_class_name()]
        assert rejection.stage == "value_rejection"
        assert rejection.reason.startswith(PGP_MISSING_PREFIX)
        assert "Options(context=...)" in rejection.reason
        assert rejection.reason == FeatureChainParser.name_path_presence_rejection_reason(
            Options(), PlainRootPgp.PROPERTY_MAPPING or {}
        )

    def test_group_key_rejection_reason_has_no_context_remedy(
        self, recorded_rejections: dict[str, MatchRejection]
    ) -> None:
        assert PlainGroupKeyPgp.match_feature_group_criteria(PlainGroupKeyPgp.get_class_name(), Options()) is False

        reason = recorded_rejections[PlainGroupKeyPgp.get_class_name()].reason
        assert reason == PGP_MISSING_PREFIX

    def test_non_match_stays_quiet_at_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Root groups are probed for every feature, so a plain non-match must not warn."""
        with caplog.at_level(logging.WARNING):
            result = PlainRootPgp.match_feature_group_criteria(PGP_ROOT_FEATURE, Options())

        assert result is False
        assert not _required_presence_warnings(caplog, "PlainRootPgp")


class TestPlainGroupExemptions:
    """The name-path exemptions carry over: default, required_when, deferred_binding and in_features."""

    def test_exempt_keys_do_not_block_the_match(self) -> None:
        assert PlainExemptPgp.match_feature_group_criteria(PlainExemptPgp.get_class_name(), Options()) is True

    def test_exempt_keys_do_not_mask_a_missing_required_key(
        self, recorded_rejections: dict[str, MatchRejection]
    ) -> None:
        name = PlainExemptPlusRequiredPgp.get_class_name()

        assert PlainExemptPlusRequiredPgp.match_feature_group_criteria(name, Options()) is False
        assert recorded_rejections[name].reason.startswith(PGP_MISSING_PREFIX)
        assert "deferred_pgp" not in recorded_rejections[name].reason
        assert "defaulted_none_pgp" not in recorded_rejections[name].reason
