"""Regression tests for the #769 name-path required-presence review fixes.

Issue #769 (warn-then-enforce name-path required presence) is implemented and green; a deep review
found four minor gaps. This file pins the ACCEPTED fixes. Two groups are RED (they fail against the
committed code and pass once the Green phase lands the fix); two are GUARDs (they pass today and fail
loudly if the pinned behavior ever regresses).

Findings pinned here:

- Finding #2 (RED): an ENFORCE-mode name-path non-match currently vanishes with no explanation. The
  Green phase adds ``FeatureChainParser.name_path_presence_rejection_reason(effective_options,
  property_mapping) -> str | None`` (a reason ONLY in enforce mode when required keys are missing) and
  wires it into ``FeatureChainParserMixin._strict_validation_rejection_reason``, so the diagnostic replay
  that feeds "no feature group matched" explains the missing key. Today the replay returns None.

- Finding #3 (RED): the env var only recognizes ``off`` as the kill-switch. The Green phase also treats
  ``0``, ``false``, ``no`` (case-insensitive) as disabled. Today those aliases fall through to warn.

- Finding #5a (GUARD): a NO_DEFAULT key with ``allow_explicit_none=True`` counts an EXPLICIT None as
  present on the name path (no warning, matches under enforce), while an ABSENT one is still missing.

- Finding #1 (GUARD): every shipped FeatureGroup that declares a PROPERTY_MAPPING stays clean on the
  name path in BOTH warn and enforce mode, because its required keys are name-carried (legacy first-
  capture fallback) or marked ``deferred_binding=True``. This guard fails loudly if that legacy
  positional-binding fallback is ever retired without also marking the affected keys deferred.

Warnings are filtered by the feature_chain_parser logger name + WARNING level + the
``MLODA_NAME_PATH_REQUIRED_PRESENCE`` marker substring, exactly like the sibling
``test_name_path_required_presence.py`` suite. Every fixture carries an "r769rf" marker so it cannot
collide with the r769 fixtures, the rf770 fixtures, or the global registry.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec
from mloda.user import Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

FEATURE_CHAIN_PARSER_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"

ENV_VAR = "MLODA_NAME_PATH_REQUIRED_PRESENCE"


def _required_presence_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The #769 name-path required-presence WARNINGs, optionally scoped to one class name.

    Filters by logger name, WARNING level, and the env-var marker substring the warning is
    contractually required to mention, exactly like the sibling suite.
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


class TestEnforceRejectionReasonSurfaced:
    """Finding #2: an enforce-mode name-path non-match must explain the missing key (RED today).

    The diagnostic replay ``_strict_validation_rejection_reason`` feeds the engine's
    "no feature group matched" message. Today it never runs the presence check, so an enforced
    non-match is silent. The Green phase surfaces the reason ONLY in enforce mode.
    """

    def test_enforce_mode_surfaces_missing_key_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RED: enforce mode must return a reason naming the missing key and the migration env var.

        Pre-implementation this fails: ``_strict_validation_rejection_reason`` returns None because the
        diagnostic replay does not run the name-path presence check.
        """
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _EnforceReasonR769rfen(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfen$"
            PROPERTY_MAPPING = {
                "op_r769rfen": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "missing_r769rfen": PropertySpec("required, options-only, absent on the name path", context=True),
            }

        reason = _EnforceReasonR769rfen._strict_validation_rejection_reason("src__op_r769rfen", Options())

        assert reason is not None, "enforce mode must explain why the name-path candidate did not match"
        assert "missing_r769rfen" in reason, "the reason must name the missing required key"
        assert ENV_VAR in reason, "the reason must mention the migration env var so the user can act"

    def test_warn_mode_reports_no_rejection_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GUARD: warn mode still matches, so there is no rejection reason to report."""
        monkeypatch.setenv(ENV_VAR, "warn")

        class _WarnReasonR769rfwn(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfwn$"
            PROPERTY_MAPPING = {
                "op_r769rfwn": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "missing_r769rfwn": PropertySpec("required, options-only, absent on the name path", context=True),
            }

        reason = _WarnReasonR769rfwn._strict_validation_rejection_reason("src__op_r769rfwn", Options())

        assert reason is None, "warn mode matches, so the presence reason is scoped out"

    def test_off_mode_reports_no_rejection_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GUARD: off mode disables the check, so there is no rejection reason to report."""
        monkeypatch.setenv(ENV_VAR, "off")

        class _OffReasonR769rfof(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfof$"
            PROPERTY_MAPPING = {
                "op_r769rfof": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "missing_r769rfof": PropertySpec("required, options-only, absent on the name path", context=True),
            }

        reason = _OffReasonR769rfof._strict_validation_rejection_reason("src__op_r769rfof", Options())

        assert reason is None, "off mode disables the check, so nothing is reported"

    def test_enforce_present_key_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GUARD: enforce mode with the required key present in options has nothing missing to report."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _PresentReasonR769rfpr(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfpr$"
            PROPERTY_MAPPING = {
                "op_r769rfpr": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "present_r769rfpr": PropertySpec("required, supplied in options", context=True),
            }

        reason = _PresentReasonR769rfpr._strict_validation_rejection_reason(
            "src__op_r769rfpr", Options(context={"present_r769rfpr": "v_r769"})
        )

        assert reason is None, "a present required key leaves nothing missing to explain"

    def test_enforce_deferred_key_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GUARD: enforce mode with a deferred_binding key has nothing missing to report."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _DeferredReasonR769rfdf(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfdf$"
            PROPERTY_MAPPING = {
                "op_r769rfdf": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "deferred_r769rfdf": PropertySpec(
                    "required, bound outside the name", context=True, deferred_binding=True
                ),
            }

        reason = _DeferredReasonR769rfdf._strict_validation_rejection_reason("src__op_r769rfdf", Options())

        assert reason is None, "deferred_binding exempts the key, so nothing is missing"

    def test_enforce_defaulted_key_no_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GUARD: enforce mode with a declared-default key has nothing missing to report."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _DefaultedReasonR769rfdd(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfdd$"
            PROPERTY_MAPPING = {
                "op_r769rfdd": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "defaulted_r769rfdd": PropertySpec("optional via a declared default", context=True, default="d_r769"),
            }

        reason = _DefaultedReasonR769rfdd._strict_validation_rejection_reason("src__op_r769rfdd", Options())

        assert reason is None, "a declared default makes the key optional, so nothing is missing"


class TestOffAliases:
    """Finding #3: 0/false/no (case-insensitive) also disable the check, like off (RED for the aliases)."""

    @pytest.mark.parametrize("env_value", ["0", "false", "FALSE", "no", "off", "OFF"])
    def test_off_aliases_disable_the_check(
        self, env_value: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing-required-key match that WOULD warn matches True and logs NO warning when disabled.

        Pre-implementation the ``0``/``false``/``FALSE``/``no`` cases fail: those values fall through to
        warn, so the name-path match still logs a presence warning. ``off``/``OFF`` already pass (guard).
        """
        monkeypatch.setenv(ENV_VAR, env_value)

        class _OffAliasR769rfal(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769rfal>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769rfal": PropertySpec("required, carried by the name", context=True),
                "missing_r769rfal": PropertySpec("required, options-only, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _OffAliasR769rfal.match_feature_group_criteria("src__val_r769rf", Options())

        assert result is True, f"a disabled mode ({env_value!r}) must not enforce"
        assert not _required_presence_warnings(caplog, "_OffAliasR769rfal"), (
            f"a disabled mode ({env_value!r}) must not warn"
        )

    def test_invalid_value_still_behaves_like_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GUARD: a genuinely invalid value (``banana``) still matches AND warns, never disables the check."""
        monkeypatch.setenv(ENV_VAR, "banana")

        class _BananaR769rfbn(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(?P<carried_r769rfbn>\w+)$"
            PROPERTY_MAPPING = {
                "carried_r769rfbn": PropertySpec("required, carried by the name", context=True),
                "missing_r769rfbn": PropertySpec("required, options-only, absent", context=True),
            }

        with caplog.at_level(logging.WARNING):
            result = _BananaR769rfbn.match_feature_group_criteria("src__val_r769rf", Options())

        assert result is True, "an invalid value must not enforce"
        assert _required_presence_warnings(caplog, "_BananaR769rfbn"), "an invalid value must fall back to warn"


class TestAllowExplicitNoneOnNamePath:
    """Finding #5a (GUARD): allow_explicit_none makes an explicit None count as present on the name path.

    The key is NO_DEFAULT, non-deferred, ``allow_explicit_none=True``; the operation key is name-carried.
    """

    def test_warn_explicit_none_matches_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARN: an explicit None for the allow_explicit_none key counts as present -> no presence warning."""
        monkeypatch.setenv(ENV_VAR, "warn")

        class _AllowNoneWarnR769rfnw(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfnw$"
            PROPERTY_MAPPING = {
                "op_r769rfnw": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "nullable_r769rfnw": PropertySpec(
                    "required, but an explicit None counts as present", context=True, allow_explicit_none=True
                ),
            }

        with caplog.at_level(logging.WARNING):
            result = _AllowNoneWarnR769rfnw.match_feature_group_criteria(
                "src__op_r769rfnw", Options(context={"nullable_r769rfnw": None})
            )

        assert result is True
        assert not _required_presence_warnings(caplog, "_AllowNoneWarnR769rfnw"), (
            "an explicit None must count as present, so the key is not flagged missing"
        )

    def test_warn_absent_warns(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """WARN (contrast): the same key entirely ABSENT is missing and warns, naming the key."""
        monkeypatch.setenv(ENV_VAR, "warn")

        class _AllowNoneAbsentR769rfna(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfna$"
            PROPERTY_MAPPING = {
                "op_r769rfna": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "nullable_r769rfna": PropertySpec(
                    "required, but an explicit None counts as present", context=True, allow_explicit_none=True
                ),
            }

        with caplog.at_level(logging.WARNING):
            result = _AllowNoneAbsentR769rfna.match_feature_group_criteria("src__op_r769rfna", Options())

        assert result is True, "warn mode still matches despite the absent key"
        warnings = _required_presence_warnings(caplog, "_AllowNoneAbsentR769rfna")
        assert warnings, "an absent required key must warn, even when allow_explicit_none is set"
        assert "nullable_r769rfna" in warnings[0].getMessage(), "the warning must name the absent key"

    def test_enforce_explicit_none_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENFORCE: an explicit None counts as present, so the name-path match still succeeds."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _AllowNoneEnforceR769rfne(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfne$"
            PROPERTY_MAPPING = {
                "op_r769rfne": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "nullable_r769rfne": PropertySpec(
                    "required, but an explicit None counts as present", context=True, allow_explicit_none=True
                ),
            }

        result = _AllowNoneEnforceR769rfne.match_feature_group_criteria(
            "src__op_r769rfne", Options(context={"nullable_r769rfne": None})
        )

        assert result is True, "an explicit None is present, so enforce mode still matches"

    def test_enforce_absent_is_non_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENFORCE (contrast): the same key entirely ABSENT is missing, so the match is rejected."""
        monkeypatch.setenv(ENV_VAR, "enforce")

        class _AllowNoneEnforceAbsentR769rfnx(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfnx$"
            PROPERTY_MAPPING = {
                "op_r769rfnx": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "nullable_r769rfnx": PropertySpec(
                    "required, but an explicit None counts as present", context=True, allow_explicit_none=True
                ),
            }

        result = _AllowNoneEnforceAbsentR769rfnx.match_feature_group_criteria("src__op_r769rfnx", Options())

        assert result is False, "an absent required key is a non-match under enforce, allow_explicit_none or not"


# Finding #1 (GUARD): representative valid name per shipped FeatureGroup that declares a PROPERTY_MAPPING.
# Each name matches its PREFIX_PATTERN and every captured token is within the bound key's allowed_values,
# so the group is clean on the name path today (required keys are name-carried or deferred_binding=True).
SHIPPED_NAME_PATH_CASES: list[tuple[type[Any], str]] = [
    (AggregatedFeatureGroup, "sales__sum_aggr"),
    (TimeWindowFeatureGroup, "temperature__avg_7_days_window"),
    (GeoDistanceFeatureGroup, "point1&point2__haversine_distance"),
    (TextCleaningFeatureGroup, "review__cleaned_text"),
    (DimensionalityReductionFeatureGroup, "features__pca_2d"),
    (EncodingFeatureGroup, "category__onehot_encoded"),
    (MissingValueFeatureGroup, "amount__mean_imputed"),
    (ScalingFeatureGroup, "amount__standard_scaled"),
    (ClusteringFeatureGroup, "features__cluster_kmeans_3"),
    (NodeCentralityFeatureGroup, "graph__degree_centrality"),
    (SklearnPipelineFeatureGroup, "data__sklearn_pipeline_preprocessing"),
    (ForecastingFeatureGroup, "sales__linear_forecast_7day"),
]

SHIPPED_IDS = [group.__name__ for group, _ in SHIPPED_NAME_PATH_CASES]


class TestShippedPluginsCleanAcrossAllGroups:
    """Finding #1 (GUARD): every shipped PROPERTY_MAPPING group is clean on the name path in every mode.

    Fails loudly if the legacy positional-binding fallback (or a per-key ``deferred_binding`` mark) is
    ever retired without keeping these groups quiet on the name path.
    """

    @pytest.mark.parametrize("mode", ["warn", "enforce"])
    @pytest.mark.parametrize("group, valid_name", SHIPPED_NAME_PATH_CASES, ids=SHIPPED_IDS)
    def test_shipped_group_name_path_is_clean(
        self,
        group: type[Any],
        valid_name: str,
        mode: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A representative valid name matches with NO presence warning, in both warn and enforce mode."""
        monkeypatch.setenv(ENV_VAR, mode)

        with caplog.at_level(logging.WARNING):
            result = group.match_feature_group_criteria(valid_name, Options())

        assert result is True, f"{group.__name__} must match its representative name {valid_name!r} in {mode} mode"
        assert not _required_presence_warnings(caplog, group.__name__), (
            f"{group.__name__} must stay clean on the name path in {mode} mode "
            f"(required keys are name-carried or deferred_binding=True)"
        )
