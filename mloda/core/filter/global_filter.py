from copy import copy, deepcopy
from datetime import datetime, timezone
from itertools import chain
from typing import Any, Optional
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import option_key_is_present
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import is_no_default
from mloda.core.abstract_plugins.components.utils import as_str, safe_field
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options, _isolate_forwarded_value
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.components.utils import contained_raise_reason
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.single_filter import SingleFilter
from mloda.core.prepare.identify_feature_group import matches_feature_group_scope, validate_single_framework_pin
from mloda.core.prepare.resolution_failure_renderer import _candidate_sort_key, near_miss_text
from mloda.core.prepare.resolution_types import Elimination, EliminationStage, rejection_elimination_stage
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


import logging

logger = logging.getLogger(__name__)

# How far down the gate chain each stage sits, so the nearest miss is the drop that got furthest. A matcher
# defect ranks lowest on purpose: a crash says nothing about how far the filter got.
_STAGE_DEPTH: dict[EliminationStage, int] = {
    "matcher_error": 0,
    "input_data": 1,
    "value_rejection": 2,
    "domain": 3,
    "scope": 4,
    "capability": 5,
    "frameworks_not_enabled": 5,
    "framework_pin": 6,
    "links": 7,
}


def _copy_feature_leaf(value: Any) -> Any:
    """Detach a Feature leaf from the host; any other leaf stays shared by reference."""
    return copy(value) if isinstance(value, Feature) else value


class GlobalFilter:
    def __init__(self) -> None:
        """
        This constructor sets up the following attributes:
        1. `filters`: A set to store individual filter objects (`SingleFilter`). Each filter represents a condition
           used to restrict or sort data based on specific features.
        2. `collection`: A dictionary mapping a tuple of feature group types and feature names to a set of filter feature
           names and the uuid to the used single filter. This is used to track which features are associated with which filters for a specific feature group.
           This can be used to check after the fact if a feature is a filter feature for a specific feature group
           e.g. for debugging, logging or quality checks.
        3. `dropped_filters`: maps (feature group, filter feature name) to the `Elimination` naming the gate that
           dropped it and why; a matcher defect outranks a stored near-miss, otherwise the deepest gate reached wins.
        4. `probes`: maps (feature group, feature name, feature uuid) to the filters that probe matched, empty included.
        5. `matched_filter_uuids`: uuids of filters that cleared every gate at least once this setup.

        These attributes provide the foundation for adding, managing, and applying filters across various feature groups
        and features in the context of a data processing pipeline.
        """
        self.filters: set[SingleFilter] = set()
        self.collection: dict[tuple[type[FeatureGroup], FeatureName], set[SingleFilter]] = {}
        self.dropped_filters: dict[tuple[type[FeatureGroup], str], Elimination] = {}
        self.probes: dict[tuple[type[FeatureGroup], FeatureName, UUID], set[SingleFilter]] = {}
        self.matched_filter_uuids: set[UUID] = set()
        self._warned_divergences: set[str] = set()
        # Own state, not dropped_filters: a falsy non-bool is an ordinary non-match, never a recorded drop.
        self._reported_falsy_matches: set[tuple[type[FeatureGroup], str]] = set()
        # WARNING dedupe for defect drops; the ledger itself no longer decides first-ness.
        self._warned_drops: set[tuple[type[FeatureGroup], str]] = set()

    def reset_match_tracking(self) -> None:
        """Every match report is scoped to one engine setup, so a later setup names only what it consulted.
        Each dedupe ledger clears with the facts it guards: one outliving them would lose that report for good."""
        self.matched_filter_uuids.clear()
        self.dropped_filters.clear()
        self._warned_divergences.clear()
        self._warned_drops.clear()
        self._reported_falsy_matches.clear()

    def rehash_stored_filters(self) -> None:
        """Reinsert stored filters whose hashes went stale, so each one is findable in its own set again."""
        # list() forces a rehash; set(value) would reuse the stale stored hashes.
        # Duplicates that became equal again intentionally merge here (same declared filter, same predicate).
        self.collection = {key: set(list(value)) for key, value in self.collection.items()}
        self.probes = {key: set(list(value)) for key, value in self.probes.items()}

    def record_probe(
        self,
        feature_group: type[FeatureGroup],
        filtered_feature_name: FeatureName,
        filtered_feature_uuid: UUID,
        matched_filters: set[SingleFilter],
    ) -> None:
        """Record what a probe matched, empty included."""
        if not self.filters:
            return
        # Rehash via list so hashes are recomputed (update(<set>) reuses the caller's stale ones).
        self.probes.setdefault((feature_group, filtered_feature_name, filtered_feature_uuid), set()).update(
            list(matched_filters)
        )

    def add_filter(
        self, filter_feature: Feature | str, filter_type: str | FilterType, parameter: dict[str, Any]
    ) -> None:
        """
        Adds a `SingleFilter` to the `filters` set based on the provided feature, filter type, and parameters.

        Parameters:
        - filter_feature: The feature or its name used for filtering. It can be a string or a `Feature` object.
            To identify if a filter is used, we need to check if the feature is part of the feature group.
            A `Feature` is stored as a snapshot: identify_matched_filters enriches a per-match deepcopy
            with the filtered feature's options, never the filter stored here.
        - filter_type: The type of filtering operation (e.g., equals, greater than). It can be a string or a `FilterType`.
            This filter_type does not need to match the FilterType, but it should be a string that is meaningful in the concrete
            Featuregroup implementation.
        - parameter: A dictionary of filter-specific options.

        A filter feature pinned to more than one compute framework raises `ComputeFrameworkPinError` here.
        """
        _single_filter = SingleFilter(filter_feature, filter_type, parameter)
        # Validated at declaration time: the raise must not depend on any probe running (#851).
        validate_single_framework_pin(_single_filter.filter_feature)
        self.filters.add(_single_filter)

    def add_filter_to_collection(
        self,
        feature_group: type[FeatureGroup],
        filtered_feature_name: FeatureName,
        single_filter: SingleFilter,
    ) -> None:
        """
        The purpose of the functionality is to store the used filter features for a specific feature group and feature.
        This way we can check after the fact if a feature is a filter feature for a specific feature group.
        """
        if (feature_group, filtered_feature_name) not in self.collection:
            self.collection[(feature_group, filtered_feature_name)] = set([single_filter])
        self.collection[(feature_group, filtered_feature_name)].add(single_filter)

    def identify_matched_filters(
        self,
        feature_group: type[FeatureGroup],
        feat: Feature,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> set[SingleFilter]:
        """
        We need to figure out if the filter feature is a part of the feature class and thus can be used as filter.

        This is quite similar to identifying the feature itself.
        The filter feature's feature_group scope is honored via the canonical predicate, as in feature resolution.
        Differences are in details:
            -   we use the options of the feature to enrich the filter feature options,
            -   we set the compute framework of the feature to determine the one of the filter feature,
            -   we consult the capability hook over the frameworks the filter would ride (its pin, else the feature's),
            -   we do not check links, as this is done earlier already and not needed anymore.

        Each gate records why it closed at its own `continue`, so the gate predicates stay pure for other callers.
        """

        matched_filters: set[SingleFilter] = set()
        for filter in self.filters:
            # We are making a deepcopy so that, we do not change the original filter.
            _filter = deepcopy(filter)
            _filter.filter_feature.options = self.unify_options(feat.options, _filter.filter_feature.options)
            filter_name = str(_filter.filter_feature.name)

            # criteria records its own drops: only it can tell a defect from a decline from a plain non-match.
            if not self.criteria(feature_group, _filter, data_access_collection):
                continue
            if self.domain(_filter, feat.domain, feature_group) is False:
                self._record_near_miss(
                    feature_group, filter_name, "domain", self._domain_reason(_filter, feat, feature_group)
                )
                continue
            if self.feature_group_scope(_filter, feature_group) is False:
                self._record_near_miss(feature_group, filter_name, "scope", "outside the requested feature group scope")
                continue
            supported = self.capability(_filter, feat, feature_group)
            if supported is not None and not supported:
                self._record_near_miss(feature_group, filter_name, "capability", self._capability_reason(_filter, feat))
                continue
            if self.compute_framework(_filter, feat, supported) is False:
                self._record_near_miss(
                    feature_group, filter_name, "framework_pin", self._framework_pin_reason(_filter, feat)
                )
                continue
            # we don't check links, because this is not necessary as this is covered by the feature and feature group before

            # After the gates, against the original declared options: only an attaching filter is described.
            self._warn_on_diverging_options(feature_group, feat.options, filter.filter_feature.options)
            self.matched_filter_uuids.add(filter.uuid)
            matched_filters.add(_filter)
        return matched_filters

    def warn_on_unmatched_filters(self) -> None:
        """Warn once per filter that matched no feature group this setup, naming its nearest miss."""
        for filter in sorted(self.filters, key=lambda f: f.name):
            if filter.uuid in self.matched_filter_uuids:
                continue
            message = f"Filter feature '{filter.name}' matched no feature group."
            # SingleFilter.name reads through to filter_feature.name, which is what the recorders key on.
            nearest = self._nearest_miss(filter.name)
            if nearest is not None:
                message += f" Nearest miss: {nearest}"
            logger.warning(message)

    def _nearest_miss(self, filter_feature_name: str) -> str | None:
        """The captured fact of the deepest gate this filter reached, as the shared near-miss bullet."""
        # The ledger keys on the name, so a name two declared filters share makes every fact under it
        # unattributable to either of them.
        if sum(1 for filter in self.filters if filter.name == filter_feature_name) > 1:
            return None
        captured = [
            (feature_group, elimination)
            for (feature_group, name), elimination in self.dropped_filters.items()
            if name == filter_feature_name
        ]
        if not captured:
            return None
        feature_group, elimination = min(captured, key=self._nearest_miss_key)
        return near_miss_text(feature_group.__name__, elimination.stage, elimination.reason)

    @staticmethod
    def _nearest_miss_key(captured: tuple[type[FeatureGroup], Elimination]) -> tuple[int, str, str]:
        """Deepest gate first, then the renderer's own candidate order, so insertion order decides nothing."""
        feature_group, elimination = captured
        name, module = _candidate_sort_key(feature_group)
        return -_STAGE_DEPTH.get(elimination.stage, 0), name, module

    def unify_options(self, feat_options: Options, filter_options: Options) -> Options:
        """Add the host options the filter feature omits, never rewriting a declared value. Layered on that repair,
        a best-effort detachment of each imported value: the container spine is rebuilt and a Feature leaf copied,
        as far as ``_isolate_forwarded_value`` reaches. ``rehash_stored_filters`` covers the Features it misses."""
        memo: dict[int, Any] = {}
        # Preserve each key's category so context keys do not leak into group (issue #712).
        for key, value in feat_options.group.items():
            if key not in filter_options:
                filter_options.add_to_group(key, _isolate_forwarded_value(value, memo, _copy_feature_leaf))
        for key, value in feat_options.context.items():
            if key not in filter_options:
                filter_options.add_to_context(key, _isolate_forwarded_value(value, memo, _copy_feature_leaf))
        return filter_options

    def _warn_on_diverging_options(
        self, feature_group: Optional[type[FeatureGroup]], feat_options: Options, filter_options: Options
    ) -> None:
        """Report keys the filter feature declares differently, unless intake provably erases the difference."""
        for key, value in chain(feat_options.group.items(), feat_options.context.items()):
            if key not in filter_options:
                continue
            declared = filter_options[key]
            if declared == value:
                continue
            fill = self._intake_fill(feature_group, key, filter_options)
            if self._converges_at_intake(fill, value):
                continue
            if fill is None:
                message = f"Options are not the same. {key} is different. {declared} != {value}"
            else:
                # Name the spec default, which is what the filter feature will actually compute with.
                message = (
                    f"Options are not the same. {key} is different. {declared!r} (intake fills {fill!r}) != {value!r}"
                )
            if message in self._warned_divergences:
                continue
            self._warned_divergences.add(message)
            logger.warning(message)

    @staticmethod
    def _intake_fill(feature_group: Optional[type[FeatureGroup]], key: str, filter_options: Options) -> Any:
        """The value intake materializes for ``key``, None when it fills nothing.

        Mirrors ``FeatureGroup.options_with_defaults``: a concrete spec default fills a key the spec reads
        as absent. Without a resolving group there is no spec to consult, so nothing is suppressed.
        """
        if feature_group is None:
            return None
        spec = (feature_group.PROPERTY_MAPPING or {}).get(key)
        if spec is None or is_no_default(spec.default) or spec.default is None:
            return None
        # Absence, not None-ness: an allow_explicit_none key is present, so its None survives.
        if option_key_is_present(spec, key, filter_options):
            return None
        return spec.default

    @staticmethod
    def _converges_at_intake(intake_fill: Any, feat_value: Any) -> bool:
        """Does intake fill the filter feature with the feature's own value?

        ``default`` is Any, so safe_field keeps an array-like default from raising out of a decision that
        only picks a log line. An uncomparable default is not convergent and keeps warning.
        """
        if intake_fill is None:
            return False
        return safe_field(lambda: bool(intake_fill == feat_value), False)

    def criteria(
        self,
        feature_group: type[FeatureGroup],
        filter: SingleFilter,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        """A raising match hook is a non-match for this filter only, mirroring the resolution seam (#845).

        The shared probe owns the window and the containment; this seam keeps the filter policy: no option
        rollback (the hook sees a per-match deepcopy) and a defect outranks a harvested decline. A matcher
        defect warns whatever its exception type, unlike the resolution seam, because nothing else surfaces
        it here. A typed decline can flip an attachment verdict, since the default hook's owned-reader veto
        gates under the probe's window.

        Mark-or-contain policy: see call_match_hook.
        """
        probe = probe_match_criteria(
            feature_group,
            filter.filter_feature.name,
            filter.filter_feature.options,
            data_access_collection,
        )
        if probe.matcher_error is not None:
            reason = contained_raise_reason(probe.matcher_error)
            self._record_dropped_filter(feature_group, str(filter.filter_feature.name), reason)
            return False
        # value_rejection is excluded: its returned is the containment's synthetic None, not the hook's.
        if probe.value_rejection is None and not probe.matched and not isinstance(probe.returned, bool):
            self._report_falsy_match(feature_group, str(filter.filter_feature.name), probe.returned)
        if probe.rejection is not None:
            self._record_rejected_filter(feature_group, str(filter.filter_feature.name), probe.rejection)
            return False
        return probe.matched

    def _record_near_miss(
        self, feature_group: type[FeatureGroup], filter_feature_name: str, stage: EliminationStage, reason: str
    ) -> None:
        """Record the gate one filter lost at against one feature group; the deepest gate reached keeps the key,
        matching how `_nearest_miss` reads the ledger back."""
        key = (feature_group, filter_feature_name)
        stored = self.dropped_filters.get(key)
        # A stored defect stays pinned to its key, and equal depth keeps the first fact recorded.
        if stored is None or (
            stored.stage != "matcher_error" and _STAGE_DEPTH.get(stored.stage, 0) < _STAGE_DEPTH.get(stage, 0)
        ):
            self.dropped_filters[key] = Elimination(stage=stage, reason=reason)

    def _record_rejected_filter(
        self, feature_group: type[FeatureGroup], filter_feature_name: str, rejection: MatchRejection
    ) -> None:
        """Record a typed decline in the same ledger at DEBUG: a deliberate rejection is a near-miss, not a defect."""
        self._record_near_miss(
            feature_group, filter_feature_name, rejection_elimination_stage(rejection.stage), rejection.reason
        )
        logger.debug(
            "%s rejected filter feature '%s': %s; dropping that filter for this feature group.",
            # A plugin-owned read past the hook call's containment, so it degrades instead of escaping the seam.
            safe_field(lambda: feature_group.get_class_name(), "<unnamed feature group>"),
            filter_feature_name,
            rejection.reason,
        )

    def _record_dropped_filter(self, feature_group: type[FeatureGroup], filter_feature_name: str, reason: str) -> None:
        """Record the drop: defect drops warn once per key and take the key from a stored near-miss."""
        key = (feature_group, filter_feature_name)
        first = key not in self._warned_drops
        self._warned_drops.add(key)
        if first:
            self.dropped_filters[key] = Elimination(stage="matcher_error", reason=reason)
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "%s %s while matching filter feature '%s'; dropping that filter for this feature group.",
            feature_group.get_class_name(),
            reason,
            filter_feature_name,
        )

    def _report_falsy_match(self, feature_group: type[FeatureGroup], filter_feature_name: str, returned: Any) -> None:
        """Report the detached filter: WARNING on a key's first report, DEBUG after, like `_record_dropped_filter`.

        Both fields are plugin-owned reads and this runs past the hook call's containment, so each degrades alone.
        """
        key = (feature_group, filter_feature_name)
        first = key not in self._reported_falsy_matches
        self._reported_falsy_matches.add(key)
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "%s returned a falsy non-bool (%s) while matching filter feature '%s'; that filter is not attached. "
            "Return True explicitly to keep it.",
            safe_field(lambda: feature_group.get_class_name(), "<unnamed feature group>"),
            # The type name only: the value's own __repr__ is plugin code and must not run here.
            safe_field(lambda: type(returned).__name__, "<unreadable type>"),
            filter_feature_name,
        )

    def domain(self, filter: SingleFilter, feature_domain: None | Domain, feature_group: type[FeatureGroup]) -> bool:
        # We have matched already the feature group and the feature.
        # Thus, we take the feature group domain if the feature domain is not set.
        feature_or_group_domain = None
        if feature_domain:
            feature_or_group_domain = feature_domain
        else:
            if feature_group.get_domain() != Domain.get_default_domain():
                feature_or_group_domain = feature_group.get_domain()

        # no domains given -> ok
        if not filter.filter_feature.domain and not feature_or_group_domain:
            return True

        # In case that filter has no domain given, we assume that the feature domain is the one to take.
        # Else the feature group should not have matched the feature domain and thus, we would not be here.
        if not filter.filter_feature.domain and feature_or_group_domain:
            filter.filter_feature.domain = feature_or_group_domain
            return True

        # In case that the filter has a domain and the feature not, it means that the
        # the feature group domain must be equal to the filter feature domain
        if filter.filter_feature.domain and not feature_domain:
            if feature_group.get_domain() == filter.filter_feature.domain:
                return True

        # both domains same -> ok
        if filter.filter_feature.domain == feature_domain:
            return True

        return False

    @staticmethod
    def _domain_reason(filter: SingleFilter, feat: Feature, feature_group: type[FeatureGroup]) -> str:
        """Name the filter feature's declared domain and the domain it lost against."""
        declared = filter.filter_feature.domain
        # Only the drop path reaches this, and the gate reaches it only for a filter that declares a domain.
        assert declared is not None
        if feat.domain is not None:
            compared = feat.domain.name
        else:
            # A plugin-owned read the gate already made; degraded because a log line is not worth a crash.
            # as_str inside the guard: an unvalidated name's own __str__ must not run in the f-string below.
            compared = safe_field(
                lambda: as_str(feature_group.get_domain().name),
                "<unreadable domain>",
                field=f"{feature_group.get_class_name()}.get_domain",
            )
        return f"the filter feature's domain '{declared.name}' does not match '{compared}'"

    def feature_group_scope(self, filter: SingleFilter, feature_group: type[FeatureGroup]) -> bool:
        scope = filter.filter_feature.feature_group_scope
        return scope is None or matches_feature_group_scope(feature_group, scope)

    def capability(
        self, filter: SingleFilter, feat: Feature, feature_group: type[FeatureGroup]
    ) -> set[type[ComputeFramework]] | None:
        """The hook removes rejected frameworks from what the filter would ride, as on the resolution seam.
        Unguarded, unlike `criteria`: a raising hook fails loudly."""
        ride_frameworks = filter.filter_feature.compute_frameworks or feat.compute_frameworks
        if not ride_frameworks:
            return None
        return {
            cfw
            for cfw in ride_frameworks
            if feature_group.supports_compute_framework(filter.filter_feature.name, filter.filter_feature.options, cfw)
        }

    @staticmethod
    def _capability_reason(filter: SingleFilter, feat: Feature) -> str:
        """Name the rejected ride frameworks, worded exactly as the canonical seam words its own drop."""
        # The accepted subset is empty here, so the rejected set is the ride set: no second ask of the hook.
        ride_frameworks = filter.filter_feature.compute_frameworks or feat.compute_frameworks or set()
        rejected_names = sorted(cfw.get_class_name() for cfw in ride_frameworks)
        return f"supports_compute_framework rejected {rejected_names}"

    def compute_framework(
        self, filter: SingleFilter, feat: Feature, supported: set[type[ComputeFramework]] | None = None
    ) -> bool:
        # case that the filter feature has no cf set -> feature defines it
        if not filter.filter_feature.compute_frameworks:
            # Hash-safe: the target is a per-match deepcopy, added to matched_filters only after all mutations.
            # Adoption owns its set and, on the engine path, carries only the hook-accepted subset, mirroring
            # the canonical seam where identified candidates hold only supported frameworks.
            adopted = supported if supported is not None else feat.compute_frameworks
            filter.filter_feature.compute_frameworks = set(adopted) if adopted is not None else None
            return True

        # case that the filter feature has an cf -> the feature framework must be one of the pinned ones.
        # Cardinality is validated at add_filter, so membership degenerates to the single pin's equality.
        if feat.get_compute_framework() in filter.filter_feature.compute_frameworks:
            return True

        return False

    @staticmethod
    def _framework_pin_reason(filter: SingleFilter, feat: Feature) -> str:
        """Name the filter's pinned framework and the one the feature resolved to."""
        # The pin degenerates to one entry, validated at add_filter.
        pinned = filter.filter_feature.get_compute_framework().get_class_name()
        resolved = feat.get_compute_framework().get_class_name()
        return f"pinned compute framework '{pinned}' is not the feature's resolved '{resolved}'"

    def add_time_and_time_travel_filters(
        self,
        event_from: datetime,
        event_to: datetime,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        max_exclusive: bool = True,
        event_time_column: str | Feature = DefaultOptionKeys.reference_time,
        validity_time_column: str | Feature = DefaultOptionKeys.time_travel,
    ) -> None:
        """
        Adds time-based filters (`event_from`, `event_to`) and optionally time-travel filters (`valid_from`, `valid_to`).
        Ensures that both `valid_from` and `valid_to` are provided together, or raises an error.

        This method is useful for filtering data based on time ranges (event) and validity periods (valid).
            Event Time Filter: For historical data (e.g., checking if a customer had a valid contract at the event time), only the event time filter is needed.

            Time Travel Filter: If prior actions (e.g., payments made before the event) are relevant,
            the time travel filter is required.

            Typically, valid_to matches the event timestamp, but in cases like payment plans, where payments occur after creation, some payments may be excluded based on the valid_to data.

        Parameters:
        - event_from (datetime): Start of the time range (with timezone).
        - event_to (datetime): End of the time range (with timezone).
        - valid_from (Optional[datetime]): Start of the validity period (optional, with timezone).
        - valid_to (Optional[datetime]): End of the validity period (optional, with timezone).
        - max_exclusive (bool): If True, the `event_to` and `valid_to` values are treated as exclusive.
        - event_time_column: the column name for the event time filter. Default is DefaultOptionKeys.reference_time.
        - validity_time_column: the column name for the validity time filter. Default is DefaultOptionKeys.time_travel.

        The bounds stored on the created `single_filters` are tz-aware `datetime` objects normalized to UTC,
        so each filter engine can compare them directly against the framework's native temporal column type
        without re-parsing.
        """

        self._add_range_filter(event_time_column, event_from, event_to, max_exclusive)

        # validate that both valid_from and valid_to are provided together
        if (valid_from is not None and valid_to is None) or (valid_from is None and valid_to is not None):
            raise ValueError("Both `valid_from` and `valid_to` must be provided together, or neither should be.")

        if valid_from and valid_to:
            self._add_range_filter(validity_time_column, valid_from, valid_to, max_exclusive)

    def _add_range_filter(
        self, filter_feature: str | Feature, time_from: datetime, time_to: datetime, max_exclusive: bool
    ) -> None:
        _time_from = GlobalFilter._normalize_to_utc(time_from)
        _time_to = GlobalFilter._normalize_to_utc(time_to)
        self.add_filter(
            filter_feature, FilterType.RANGE, {"min": _time_from, "max": _time_to, "max_exclusive": max_exclusive}
        )

    @staticmethod
    def _normalize_to_utc(time_with_tz: datetime) -> datetime:
        """Validate tz-aware datetime and normalize to UTC for filtering."""
        if time_with_tz.tzinfo is None:
            raise ValueError(f"Timezone information is missing in {time_with_tz}")

        return time_with_tz.astimezone(timezone.utc)
