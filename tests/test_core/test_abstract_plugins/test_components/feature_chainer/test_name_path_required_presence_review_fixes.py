"""Regression pins for the mandatory name-path required-presence rule (issue #769).

Pinned here:

- ``FeatureChainParser.name_path_presence_rejection_reason`` returns a reason naming the missing
  key(s) UNCONDITIONALLY when keys are missing, None when nothing is missing, and never mentions
  the retired env var. ``FeatureChainParserMixin._strict_validation_rejection_reason`` surfaces it,
  so the resolution-failure report explains the non-match.
- ``MLODA_NAME_PATH_REQUIRED_PRESENCE`` is retired: no value changes the reason or the verdict.
- ``allow_explicit_none``: an explicit None counts as present on the name path; the same key
  entirely absent is a non-match.
- Every shipped FeatureGroup that declares a PROPERTY_MAPPING stays clean on the name path: its
  required keys are name-carried (legacy first-capture fallback) or marked ``deferred_binding=True``.
  This guard fails loudly if that fallback is ever retired without marking the affected keys deferred.

Warnings are filtered by the feature_chain_parser logger name + WARNING level + the contractual
``required option(s)`` marker substring, exactly like the sibling suite. Every fixture carries an
"r769rf" marker so it cannot collide with the r769 fixtures or the global registry.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
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

# Retired: tests only set it to prove it is ignored, and assert it never appears in reasons.
ENV_VAR = "MLODA_NAME_PATH_REQUIRED_PRESENCE"

# The marker substring the non-match warning is contractually required to contain.
MARKER = "required option(s)"


def _required_presence_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The name-path required-presence WARNINGs, optionally scoped to one class name.

    Filters by logger name, WARNING level, and the ``required option(s)`` marker substring the
    warning is contractually required to contain, exactly like the sibling suite.
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


class TestRejectionReasonSurfaced:
    """A name-path presence non-match must explain the missing key in the resolution-failure replay.

    The diagnostic replay ``_strict_validation_rejection_reason`` feeds the engine's
    "no feature group matched" message; the presence reason is reported unconditionally.
    """

    def test_missing_key_reason_surfaced(self) -> None:
        """The replay returns a reason naming the missing key, with no env manipulation."""

        class _ReasonR769rfen(FeatureChainParserMixin, FeatureGroup):
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

        reason = _ReasonR769rfen._strict_validation_rejection_reason("src__op_r769rfen", Options())

        assert reason is not None, "a name-path presence non-match must explain itself"
        assert "missing_r769rfen" in reason, "the reason must name the missing required key"
        assert ENV_VAR not in reason, "the reason must not reference the retired env var"

    @pytest.mark.parametrize("env_value", ["off", "warn", "0", "enforce"])
    def test_env_value_does_not_change_the_reason(self, env_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The env var is retired: any value still yields the same env-var-free reason."""
        monkeypatch.setenv(ENV_VAR, env_value)

        class _ReasonEnvR769rfev(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__(\w+)_r769rfev$"
            PROPERTY_MAPPING = {
                "op_r769rfev": PropertySpec(
                    "operation carried by the positional capture",
                    allowed_values=("op",),
                    context=True,
                    strict_validation=True,
                ),
                "missing_r769rfev": PropertySpec("required, options-only, absent on the name path", context=True),
            }

        reason = _ReasonEnvR769rfev._strict_validation_rejection_reason("src__op_r769rfev", Options())

        assert reason is not None, f"env value {env_value!r} must be ignored: the reason is unconditional"
        assert "missing_r769rfev" in reason
        assert ENV_VAR not in reason, "the reason must not reference the retired env var"

    def test_present_key_no_reason(self) -> None:
        """GUARD: the required key present in options leaves nothing missing to report."""

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

    def test_deferred_key_no_reason(self) -> None:
        """GUARD: a deferred_binding key has nothing missing to report."""

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

    def test_defaulted_key_no_reason(self) -> None:
        """GUARD: a declared-default key has nothing missing to report."""

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


class TestNamePathPresenceRejectionReasonUnit:
    """Direct contract of ``FeatureChainParser.name_path_presence_rejection_reason``."""

    def test_reason_when_key_missing(self) -> None:
        """Missing keys produce a reason unconditionally; the reason names them and no env var."""
        mapping = {"missing_r769rfu": PropertySpec("required, absent", context=True)}

        reason = FeatureChainParser.name_path_presence_rejection_reason(Options(), mapping)

        assert reason is not None, "missing keys must produce a reason unconditionally"
        assert "missing_r769rfu" in reason, "the reason must name the missing key"
        assert ENV_VAR not in reason, "the reason must not reference the retired env var"

    def test_none_when_nothing_missing(self) -> None:
        """Nothing missing means no reason."""
        mapping = {"present_r769rfu2": PropertySpec("required, present", context=True)}

        reason = FeatureChainParser.name_path_presence_rejection_reason(
            Options(context={"present_r769rfu2": "v_r769rf"}), mapping
        )

        assert reason is None


class TestAllowExplicitNoneOnNamePath:
    """allow_explicit_none makes an explicit None count as present on the name path.

    The key is NO_DEFAULT, non-deferred, ``allow_explicit_none=True``; the operation key is
    name-carried.
    """

    def test_explicit_none_counts_as_present(self, caplog: pytest.LogCaptureFixture) -> None:
        """An explicit None for the allow_explicit_none key counts as present: match, no warning."""

        class _AllowNonePresentR769rfnw(FeatureChainParserMixin, FeatureGroup):
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
            result = _AllowNonePresentR769rfnw.match_feature_group_criteria(
                "src__op_r769rfnw", Options(context={"nullable_r769rfnw": None})
            )

        assert result is True, "an explicit None is present, so the match succeeds"
        assert not _required_presence_warnings(caplog, "_AllowNonePresentR769rfnw"), (
            "an explicit None must count as present, so the key is not flagged missing"
        )

    def test_absent_key_is_non_match(self, caplog: pytest.LogCaptureFixture) -> None:
        """Contrast: the same key entirely ABSENT is missing, so the match is rejected and warned."""

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

        assert result is False, "an absent required key is a non-match, allow_explicit_none or not"
        warnings = _required_presence_warnings(caplog, "_AllowNoneAbsentR769rfna")
        assert warnings, "the absent required key must be warned about"
        assert "nullable_r769rfna" in warnings[0].getMessage(), "the warning must name the absent key"


# Representative valid name per shipped FeatureGroup that declares a PROPERTY_MAPPING. Each name
# matches its PREFIX_PATTERN and every captured token is within the bound key's allowed_values, so
# the group is clean on the name path (required keys are name-carried or deferred_binding=True).
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
    """GUARD: every shipped PROPERTY_MAPPING group is clean on the name path.

    Fails loudly if the legacy positional-binding fallback (or a per-key ``deferred_binding`` mark)
    is ever retired without keeping these groups quiet on the name path.
    """

    @pytest.mark.parametrize("group, valid_name", SHIPPED_NAME_PATH_CASES, ids=SHIPPED_IDS)
    def test_shipped_group_name_path_is_clean(
        self,
        group: type[Any],
        valid_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A representative valid name matches with NO presence warning."""
        with caplog.at_level(logging.WARNING):
            result = group.match_feature_group_criteria(valid_name, Options())

        assert result is True, f"{group.__name__} must match its representative name {valid_name!r}"
        assert not _required_presence_warnings(caplog, group.__name__), (
            f"{group.__name__} must stay clean on the name path "
            f"(required keys are name-carried or deferred_binding=True)"
        )
