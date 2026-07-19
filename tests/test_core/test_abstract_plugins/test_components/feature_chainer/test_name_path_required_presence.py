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
other feature groups in the global registry.
"""

from __future__ import annotations

import logging

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec
from mloda.user import Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
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
        """TimeWindowFeatureGroup matches a string name with no presence warning.

        window_size/time_unit are deferred_binding and in_features is excluded, so the name-only
        match reports nothing.
        """
        with caplog.at_level(logging.WARNING):
            result = TimeWindowFeatureGroup.match_feature_group_criteria("temperature__avg_7_days_window", Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "TimeWindowFeatureGroup"), (
            "deferred_binding + in_features exclusion must keep TimeWindowFeatureGroup clean on the name path"
        )

    def test_time_window_deferred_binding_marks_are_present(self) -> None:
        """White-box: window_size and time_unit carry the deferred_binding mark."""
        assert TimeWindowFeatureGroup.PROPERTY_MAPPING["window_size"].deferred_binding is True
        assert TimeWindowFeatureGroup.PROPERTY_MAPPING["time_unit"].deferred_binding is True
