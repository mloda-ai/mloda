"""Name-path required-presence, warn-then-enforce migration (issue #769).

Historically the STRING-NAMED match path never enforced required presence: a key with no declared
default and no ``required_when`` could be absent and the feature still matched, as long as the name
identified the group. The config-based path always enforced it. #769 closes that gap on the name path
behind a warn-then-enforce migration mechanism, with a per-key opt-out (``PropertySpec.deferred_binding``).

Mode is read from the env var ``MLODA_NAME_PATH_REQUIRED_PRESENCE`` (case-insensitive):

- unset OR ``"warn"`` (DEFAULT) -> WARN: a missing required key still MATCHES but logs a WARNING.
- ``"enforce"``                 -> ENFORCE: a missing required key becomes a NON-MATCH (and warns).
- ``"off"``                     -> the check is disabled: no warning, no enforcement (kill-switch).
- any other/invalid value       -> treated as WARN.

A "required, name-path-relevant" key is one where ``FeatureChainParser._can_skip_required_check(spec)``
is False (no declared default AND no ``required_when``) AND ``spec.deferred_binding`` is False. A key is
"missing" when, after name bindings are merged into effective options, its option value is absent (None,
honoring ``allow_explicit_none``). Declared-default keys and ``required_when`` keys are NOT flagged by this
check (``required_when`` has its own guard).

Contract details pinned here:
- The warning names the feature group class and each missing key, and mentions both the
  ``deferred_binding`` opt-out and the ``MLODA_NAME_PATH_REQUIRED_PRESENCE`` env var.
- ``deferred_binding=True`` silences the warning (warn) and keeps the match (enforce) for that key.
- The config-based path is UNCHANGED: a required key absent there is still a non-match in every mode,
  and ``deferred_binding=True`` does NOT exempt it on the config path.
- ``required_when`` gating is owned by its own guard, not by this check.

Every fixture carries an "r769" marker in its class name, keys, and values so it cannot collide with
other feature groups in the global registry. Warnings are filtered by the feature_chain_parser logger
name + WARNING level + the ``MLODA_NAME_PATH_REQUIRED_PRESENCE`` marker substring, exactly like the
sibling universal-matcher and forwarded-mismatch suites filter theirs.
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

ENV_VAR = "MLODA_NAME_PATH_REQUIRED_PRESENCE"

# A name the fixture patterns recognize: "<source>__<capture>". The capture binds the "carried" key.
NAME_PATH_FEATURE = "src__val_r769"
# A separator-free name no fixture pattern captures, so it falls through to the configuration path.
CONFIG_PATH_FEATURE = "config_only_r769"


def _required_presence_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The #769 name-path required-presence WARNINGs, optionally scoped to one class name.

    Filters by logger name, WARNING level, and the env-var marker substring the warning is contractually
    required to mention, mirroring how the sibling suites scope their diagnostics.
    """
    records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == FEATURE_CHAIN_PARSER_LOGGER
        and ENV_VAR in record.getMessage()
    ]
    if class_name is not None:
        records = [record for record in records if class_name in record.getMessage()]
    return records


class TestWarnMode:
    """WARN mode (default): a missing required key still matches, but the gap is announced."""

    def test_missing_required_key_matches_but_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 1: default (unset) mode -> the name-path match returns True and logs a naming WARNING."""
        monkeypatch.delenv(ENV_VAR, raising=False)  # unset == warn

        class _WarnMissingR769a(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769a>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769a": PropertySpec("required, carried by the name", context=True),
                "missing_r769a": PropertySpec("required, options-only, absent", context=True),
            }

        # Preconditions: the missing key is unconditionally required and name-path relevant.
        missing_spec = _WarnMissingR769a.PROPERTY_MAPPING["missing_r769a"]
        assert FeatureChainParser._can_skip_required_check(missing_spec) is False

        with caplog.at_level(logging.WARNING):
            result = _WarnMissingR769a.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "warn mode must not change the verdict: the name match still succeeds"
        warnings = _required_presence_warnings(caplog, "_WarnMissingR769a")
        assert warnings, "warn mode must log a required-presence warning naming the class"
        message = warnings[0].getMessage()
        assert "missing_r769a" in message, "the warning must name the missing key"
        assert "deferred_binding" in message, "the warning must mention the deferred_binding opt-out"
        assert ENV_VAR in message, "the warning must mention the migration env var"

    def test_all_satisfied_no_warning(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """Case 2: no warning when every required key is satisfied by name/default/required_when/options."""
        monkeypatch.delenv(ENV_VAR, raising=False)

        class _WarnSatisfiedR769b(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769b>\w+)$"
            PROPERTY_MAPPING = {
                # required, satisfied by the name binding
                "carried_r769b": PropertySpec("carried by the name", context=True),
                # declared default -> optional, never flagged
                "defaulted_r769b": PropertySpec("has a declared default", context=True, default="d_r769"),
                # required_when -> owned by its own guard, not this check
                "cond_r769b": PropertySpec(
                    "conditionally required", context=True, default=None, required_when=lambda o: False
                ),
                # required, explicitly present in options
                "present_r769b": PropertySpec("required, present in options", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _WarnSatisfiedR769b.match_feature_group_criteria(
                NAME_PATH_FEATURE, Options(context={"present_r769b": "x_r769"})
            )

        assert result is True
        assert not _required_presence_warnings(caplog, "_WarnSatisfiedR769b"), (
            "no key is missing, so nothing must be warned"
        )

    def test_deferred_binding_silences_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 3: ``deferred_binding=True`` on the otherwise-missing required key silences the warning.

        Pre-implementation this fixture raises ``TypeError`` at construction (``deferred_binding`` is not
        yet a PropertySpec field): the expected Part-A-surfacing failure.
        """
        monkeypatch.delenv(ENV_VAR, raising=False)

        class _WarnDeferredR769c(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769c>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769c": PropertySpec("carried by the name", context=True),
                "deferred_r769c": PropertySpec(
                    "required but opted out of the name-path check", context=True, deferred_binding=True
                ),
            }

        with caplog.at_level(logging.WARNING):
            result = _WarnDeferredR769c.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "_WarnDeferredR769c"), (
            "deferred_binding=True must silence the required-presence warning"
        )

    def test_multiple_missing_keys_all_named_in_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 6 (warn): the warning identifies ALL missing required keys, not just the first."""
        monkeypatch.delenv(ENV_VAR, raising=False)

        class _WarnMultiR769f(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769f>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769f": PropertySpec("carried by the name", context=True),
                "missing_one_r769f": PropertySpec("required, absent", context=True),
                "missing_two_r769f": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _WarnMultiR769f.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True
        warnings = _required_presence_warnings(caplog, "_WarnMultiR769f")
        assert warnings, "a multi-missing name-path match must warn"
        rendered = " ".join(record.getMessage() for record in warnings)
        assert "missing_one_r769f" in rendered
        assert "missing_two_r769f" in rendered

    def test_explicit_warn_value_warns(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """The explicit value ``"warn"`` behaves exactly like the unset default."""
        monkeypatch.setenv(ENV_VAR, "warn")

        class _ExplicitWarnR769w(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769w>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769w": PropertySpec("carried by the name", context=True),
                "missing_r769w": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _ExplicitWarnR769w.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True
        assert _required_presence_warnings(caplog, "_ExplicitWarnR769w"), "explicit 'warn' must warn"

    def test_invalid_env_value_behaves_like_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 9: an unrecognized value (e.g. ``"banana"``) falls back to warn behavior."""
        monkeypatch.setenv(ENV_VAR, "banana")

        class _InvalidEnvR769(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769inv>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769inv": PropertySpec("carried by the name", context=True),
                "missing_r769inv": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _InvalidEnvR769.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "an invalid value must not enforce"
        assert _required_presence_warnings(caplog, "_InvalidEnvR769"), "an invalid value must warn"


class TestEnforceMode:
    """ENFORCE mode: a missing required key on the name path becomes a non-match (and still warns)."""

    def test_missing_required_key_is_non_match(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 4: env=enforce turns the same missing-required-key name-path match into False + a WARNING."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _EnforceMissingR769d(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769d>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769d": PropertySpec("required, carried by the name", context=True),
                "missing_r769d": PropertySpec("required, options-only, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _EnforceMissingR769d.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is False, "enforce mode must reject a name-path match with a missing required key"
        warnings = _required_presence_warnings(caplog, "_EnforceMissingR769d")
        assert warnings, "enforce mode must still name the class and missing key"
        assert "missing_r769d" in warnings[0].getMessage()

    def test_deferred_binding_still_matches(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 4 (deferred): ``deferred_binding=True`` matches even under enforce.

        Pre-implementation this raises ``TypeError`` at construction (unknown ``deferred_binding``).
        """
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _EnforceDeferredR769de(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769de>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769de": PropertySpec("carried by the name", context=True),
                "deferred_r769de": PropertySpec("required but opted out", context=True, deferred_binding=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _EnforceDeferredR769de.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "deferred_binding=True must keep the match even under enforce"
        assert not _required_presence_warnings(caplog, "_EnforceDeferredR769de")

    def test_present_key_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case 4 (present): a required key present in options matches under enforce."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _EnforcePresentR769e(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769e>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769e": PropertySpec("carried by the name", context=True),
                "present_r769e": PropertySpec("required, present in options", context=True),
            }

        result = _EnforcePresentR769e.match_feature_group_criteria(
            NAME_PATH_FEATURE, Options(context={"present_r769e": "y_r769"})
        )

        assert result is True

    def test_name_bound_key_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case 4 (name-bound): a required key satisfied purely by the name binding matches under enforce."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _EnforceNameBoundR769j(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769j>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769j": PropertySpec("required, carried by the name", context=True),
            }

        result = _EnforceNameBoundR769j.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "a name-bound required key is satisfied by the name, so enforce still matches"

    def test_multiple_missing_keys_all_named_in_rejection(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 6 (enforce): the rejection warning identifies ALL missing required keys."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _EnforceMultiR769em(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769em>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769em": PropertySpec("carried by the name", context=True),
                "missing_one_r769em": PropertySpec("required, absent", context=True),
                "missing_two_r769em": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _EnforceMultiR769em.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is False
        warnings = _required_presence_warnings(caplog, "_EnforceMultiR769em")
        assert warnings, "an enforced multi-missing rejection must warn"
        rendered = " ".join(record.getMessage() for record in warnings)
        assert "missing_one_r769em" in rendered
        assert "missing_two_r769em" in rendered

    def test_enforce_is_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The mode read is case-insensitive: ``"ENFORCE"`` enforces exactly like ``"enforce"``."""
        monkeypatch.setenv(ENV_VAR, "ENFORCE")

        class _EnforceCaseR769ci(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769ci>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769ci": PropertySpec("carried by the name", context=True),
                "missing_r769ci": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _EnforceCaseR769ci.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is False, "the mode value must be matched case-insensitively"
        assert _required_presence_warnings(caplog, "_EnforceCaseR769ci")


class TestOffMode:
    """OFF mode: the check is a real kill-switch: no warning, no enforcement."""

    def test_missing_required_key_matches_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 5: env=off -> the missing-required-key name-path match returns True and logs no warning."""
        monkeypatch.setenv(ENV_VAR, "off")

        class _OffMissingR769o(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769o>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769o": PropertySpec("carried by the name", context=True),
                "missing_r769o": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _OffMissingR769o.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "off mode must not enforce"
        assert not _required_presence_warnings(caplog, "_OffMissingR769o"), "off mode must not warn"

    def test_off_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """``"OFF"`` disables the check exactly like ``"off"``."""
        monkeypatch.setenv(ENV_VAR, "OFF")

        class _OffCaseR769oc(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769oc>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769oc": PropertySpec("carried by the name", context=True),
                "missing_r769oc": PropertySpec("required, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _OffCaseR769oc.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "_OffCaseR769oc")


class TestConfigPathUnchanged:
    """Turning the name path on must not touch the config path's long-standing required-presence rule."""

    def test_config_path_rejects_missing_required_all_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case 7: a NO_DEFAULT key absent on the config path is a non-match in every mode."""

        class _ConfigReqR769g(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769g>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769g": PropertySpec("bound by the name on the name path", context=True),
                "req_r769g": PropertySpec("required, options-only", context=True),
            }

        for mode in (None, "warn", "enforce", "off"):
            if mode is None:
                monkeypatch.delenv(ENV_VAR, raising=False)
            else:
                monkeypatch.setenv(ENV_VAR, mode)
            # CONFIG_PATH_FEATURE carries no separator, so the name never identifies the group: the
            # config path decides, and req_r769g (NO_DEFAULT) is absent -> non-match, whatever the mode.
            result = _ConfigReqR769g.match_feature_group_criteria(
                CONFIG_PATH_FEATURE, Options(context={"carried_r769g": "x_r769"})
            )
            assert result is False, f"config-path required presence must hold in mode {mode!r}"

    def test_deferred_binding_does_not_exempt_config_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Case 7 (deferred): ``deferred_binding=True`` exempts only the name path, never the config path.

        Pre-implementation this raises ``TypeError`` at construction (unknown ``deferred_binding``).
        """
        monkeypatch.setenv(ENV_VAR, "warn")

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

    def test_name_path_warns_where_config_path_rejects(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 7 (contrast): same fixture, same missing key -> name path warns-and-matches, config path rejects."""
        monkeypatch.delenv(ENV_VAR, raising=False)  # warn

        class _ContrastR769h(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769h>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769h": PropertySpec("bound by the name / provided on the config path", context=True),
                "req_only_r769h": PropertySpec("required, options-only, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            name_result = _ContrastR769h.match_feature_group_criteria(NAME_PATH_FEATURE, Options())
            config_result = _ContrastR769h.match_feature_group_criteria(
                CONFIG_PATH_FEATURE, Options(context={"carried_r769h": "x_r769"})
            )

        assert name_result is True, "the name path (warn mode) still matches despite the missing key"
        assert config_result is False, "the config path rejects the same missing required key"
        warnings = _required_presence_warnings(caplog, "_ContrastR769h")
        assert warnings, "the name-path match must have warned about the missing key"
        assert "req_only_r769h" in warnings[0].getMessage()


class TestRequiredWhenInterplay:
    """A required_when-only gating is owned by its own guard, never by the #769 name-path check."""

    def test_required_when_only_not_warned_in_warn_mode(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 8 (warn): a group whose only requirement is a required_when predicate is not warned."""
        monkeypatch.delenv(ENV_VAR, raising=False)

        class _ReqWhenWarnR769i(FeatureChainParserMixin):
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
            result = _ReqWhenWarnR769i.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "_ReqWhenWarnR769i"), (
            "required_when is owned by its own guard, not by the name-path required-presence check"
        )

    def test_required_when_only_not_enforced_in_enforce_mode(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Case 8 (enforce): the same required_when-only group matches under enforce and is not warned."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _ReqWhenEnforceR769ie(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769ie>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769ie": PropertySpec("optional, carried by the name", context=True, default=None),
                "cond_r769ie": PropertySpec(
                    "required only when trigger is present",
                    context=True,
                    default=None,
                    required_when=lambda o: o.get("trigger_r769ie") is not None,
                ),
            }

        with caplog.at_level(logging.WARNING):
            result = _ReqWhenEnforceR769ie.match_feature_group_criteria(NAME_PATH_FEATURE, Options())

        assert result is True, "a required_when that does not fire must not be enforced by the #769 check"
        assert not _required_presence_warnings(caplog, "_ReqWhenEnforceR769ie")


class TestInFeaturesExcluded:
    """The name-path check must EXCLUDE ``DefaultOptionKeys.in_features``.

    On the name path the source features come from the name prefix (the ``src`` in ``src__op``), so
    the ``in_features`` key is name-satisfied, never missing. All 12 shipped plugins declare
    ``in_features`` as a NO_DEFAULT spec, so without this exclusion every shipped plugin would falsely
    warn on every name-path match. These guards pin the behavior the Green phase must produce: the
    in_features key stays silent while a genuine missing key is still reported.
    """

    def test_warn_mode_in_features_not_reported_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARN mode: op captured + in_features excluded -> match True and NO presence warning."""
        monkeypatch.delenv(ENV_VAR, raising=False)  # warn

        class _InFeaturesExcludedR769inf(FeatureChainParserMixin, FeatureGroup):
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

        # Precondition: in_features is unconditionally required as a spec, so the exclusion is what keeps
        # it quiet, not a declared default.
        in_features_spec = _InFeaturesExcludedR769inf.PROPERTY_MAPPING[DefaultOptionKeys.in_features]
        assert FeatureChainParser._can_skip_required_check(in_features_spec) is False

        with caplog.at_level(logging.WARNING):
            result = _InFeaturesExcludedR769inf.match_feature_group_criteria("src__op_r769inf", Options())

        assert result is True
        warnings = _required_presence_warnings(caplog, "_InFeaturesExcludedR769inf")
        assert not warnings, "in_features is name-satisfied by the source prefix and must never be flagged"

    def test_enforce_mode_in_features_not_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENFORCE mode: op captured + in_features excluded -> still a match (no missing key)."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _InFeaturesEnforcedR769inf(FeatureChainParserMixin, FeatureGroup):
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

        result = _InFeaturesEnforcedR769inf.match_feature_group_criteria("src__op_r769inf", Options())

        assert result is True, "in_features is excluded, so enforce mode still matches a name-path feature"

    def test_in_features_silent_while_genuine_missing_key_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Contrast: a genuine missing key (not in_features, not deferred) still warns; in_features stays silent."""
        monkeypatch.delenv(ENV_VAR, raising=False)  # warn

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

        assert result is True
        warnings = _required_presence_warnings(caplog, "_InFeaturesContrastR769inf2")
        assert warnings, "a genuine missing required key must still warn"
        rendered = " ".join(record.getMessage() for record in warnings)
        assert "genuine_missing_r769inf2" in rendered, "the genuine missing key must be named"
        assert "in_features" not in rendered, "in_features is name-satisfied and must never be reported missing"


class TestShippedPluginsClean:
    """Shipped plugins stay clean on the name path once the #769 migration lands.

    These guards pin the end state: the ``in_features`` exclusion plus the per-plugin
    ``deferred_binding`` marks keep representative shipped plugins from falsely warning on a plain
    name-path match. The white-box asserts pin the core migration (deferred marks) directly; they error
    with AttributeError until ``PropertySpec.deferred_binding`` exists.
    """

    def test_aggregated_feature_group_name_match_is_clean(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARN mode: AggregatedFeatureGroup matches a string name with no presence warning.

        Its only otherwise-flaggable key is in_features, which the exclusion keeps quiet.
        """
        monkeypatch.delenv(ENV_VAR, raising=False)  # warn

        with caplog.at_level(logging.WARNING):
            result = AggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "AggregatedFeatureGroup"), (
            "in_features exclusion must keep AggregatedFeatureGroup clean on the name path"
        )

    def test_time_window_feature_group_name_match_is_clean(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARN mode: TimeWindowFeatureGroup matches a string name with no presence warning.

        window_size/time_unit are marked deferred_binding by Green and in_features is excluded, so the
        name-only match reports nothing.
        """
        monkeypatch.delenv(ENV_VAR, raising=False)  # warn

        with caplog.at_level(logging.WARNING):
            result = TimeWindowFeatureGroup.match_feature_group_criteria("temperature__avg_7_days_window", Options())

        assert result is True
        assert not _required_presence_warnings(caplog, "TimeWindowFeatureGroup"), (
            "deferred_binding + in_features exclusion must keep TimeWindowFeatureGroup clean on the name path"
        )

    def test_time_window_deferred_binding_marks_are_present(self) -> None:
        """White-box: the core migration marks window_size and time_unit as deferred_binding.

        Pre-implementation this errors (``PropertySpec`` has no ``deferred_binding`` attribute yet): the
        expected pre-implementation state.
        """
        assert TimeWindowFeatureGroup.PROPERTY_MAPPING["window_size"].deferred_binding is True
        assert TimeWindowFeatureGroup.PROPERTY_MAPPING["time_unit"].deferred_binding is True
