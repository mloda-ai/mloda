"""One home for the match-hook call: the containment, the marked-abort re-raise and the return coercion.

Both match seams held their own try around ``match_feature_group_criteria`` and drifted apart twice (#991).
Each keeps only its own recording and rollback now, driven by the outcome this helper hands back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import is_match_abort

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.feature_group import FeatureGroup


@dataclass(frozen=True)
class MatchHookOutcome:
    """One match-hook call: the coerced verdict, the raw return, and the raise that was contained."""

    matched: bool
    returned: Any
    error: Optional[Exception]


def call_match_hook(
    feature_group: type[FeatureGroup],
    feature_name: FeatureName | str,
    options: Options,
    data_access_collection: Optional[DataAccessCollection] = None,
) -> MatchHookOutcome:
    """Ask one candidate's match hook: a raise out of it is a non-match for that candidate only (#845).

    Policy: mark a raise with escalate_match_abort when it reports a misconfiguration, a contradiction or a
    framework defect; leave it contained when it is one candidate's own judgment or own defect. Every raise
    the match path reaches must say which, at the raise: tests/test_core/test_prepare/test_match_abort_sweep.py
    enforces that. A mark only survives if every handler in between re-raises it (``safe_field`` does not). An
    option-write conflict during reader selection escalates as a contradiction, decided at the reader-selection
    raise.

    The contained exception is handed back rather than judged here: each seam records and rolls back its own way.
    """
    try:
        # bool() inside the try: reading a plugin's return is itself a plugin call (#927).
        returned = feature_group.match_feature_group_criteria(feature_name, options, data_access_collection)
        return MatchHookOutcome(bool(returned), returned, None)
    except Exception as exc:  # noqa: BLE001  (contained: one broken matcher must not poison the whole run)
        if is_match_abort(exc):
            raise
        # The outcome outlives this except block, and the traceback's frames pin the plugin class and its raw
        # return. with_traceback, not the attribute: a frozen-dataclass exception rejects the assignment.
        return MatchHookOutcome(False, None, exc.with_traceback(None))
