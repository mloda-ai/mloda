"""Definition-time diagnostic for a "universal configuration matcher" PROPERTY_MAPPING (issue #771).

A FeatureGroup whose PROPERTY_MAPPING has ZERO unconditionally-required keys, declares no capturing
pattern, and inherits the mixin's configuration matcher will match ANY feature name with EMPTY
options (Finding 10 of #750). That silent over-matching is what this diagnostic warns about at
class-definition time. The warning names the escape hatch ``ALLOW_UNIVERSAL_MATCHER`` and the class.

Authors silence it legitimately by: declaring an unconditionally-required key, supplying a genuinely
discriminating ``match_feature_group_criteria``, gating with ``required_when`` predicates that fire
for empty options, or opting in with ``ALLOW_UNIVERSAL_MATCHER = True``.

Requiredness (final PropertySpec semantics, already in the repo):
- ``default=NO_DEFAULT`` AND ``required_when is None``  -> unconditionally required.
- any declared ``default`` (including ``default=None``)  -> optional.
- ``required_when=<predicate>``                          -> conditionally required (not unconditional).
``FeatureChainParser._can_skip_required_check(spec)`` is True for the optional-or-conditional cases.

Every fixture carries a "u771" marker in its class name, keys, and values so it cannot collide in the
global plugin registry; the captureless test (test_captureless_no_binding.py) uses "c772" the same way.
The universal fixtures declare NO PREFIX_PATTERN/SUFFIX_PATTERN (a pure-config matcher), so the ONLY
diagnostic in scope is the universal-matcher warning, never the captureless one.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

FEATURE_CHAIN_PARSER_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"

# A name that no fixture pattern would ever recognize, used as the "unrelated probe".
UNRELATED_NAME_U771 = "some_unrelated_feature_u771"

# Case 5's two mutually-exclusive conditional keys (modeled on the sklearn PIPELINE_NAME/PIPELINE_STEPS
# pattern): each is required only when the other is absent, so EMPTY options leaves both required.
COND_KEY_A_U771 = "pipe_a_u771e"
COND_KEY_B_U771 = "pipe_b_u771e"


def _universal_matcher_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The ALLOW_UNIVERSAL_MATCHER definition-time warnings, optionally scoped to one class name.

    Filters exactly like the captureless test filters RECOGNITION_ONLY_PATTERN records: by logger
    name, WARNING level, and the marker substring in the rendered message.
    """
    records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == FEATURE_CHAIN_PARSER_LOGGER
        and "ALLOW_UNIVERSAL_MATCHER" in record.getMessage()
    ]
    if class_name is not None:
        records = [record for record in records if class_name in record.getMessage()]
    return records


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Its tests define local FeatureGroup subclasses (some whose matcher raises or matches any name). Those
    class objects sit in reference cycles, lingering in FeatureGroup.__subclasses__() until a GC cycle runs;
    while they linger, other tests that enumerate via get_all_subclasses(FeatureGroup) (e.g. test_resolve_feature)
    trip over them. Force a collection and assert none of this module's classes remain.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


class TestUniversalMatcherWarns:
    """The guard warns when an inherited config matcher matches any feature name with empty options."""

    def test_inherited_all_declared_default_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 1: every key declares ``default=None``, no pattern -> universal matcher -> WARNS."""
        with caplog.at_level(logging.WARNING):

            class _InheritedAllDefaultU771a(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "opt_a_u771a": PropertySpec("optional a", default=None),
                    "opt_b_u771a": PropertySpec("optional b", default=None),
                }

            assert "opt_a_u771a" in _InheritedAllDefaultU771a.PROPERTY_MAPPING

        warnings = _universal_matcher_warnings(caplog, "_InheritedAllDefaultU771a")
        assert warnings, "expected an ALLOW_UNIVERSAL_MATCHER warning naming the class"

    def test_declared_default_value_is_optional_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 6: a concrete declared default (not just None) still counts as optional -> WARNS."""
        with caplog.at_level(logging.WARNING):

            class _DeclaredDefaultU771f(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "defaulted_u771f": PropertySpec(
                        "declared concrete default",
                        default="x_u771",
                        strict_validation=True,
                        allowed_values=("x_u771",),
                    ),
                }

            # Precondition: a concrete declared default makes the key optional, not required.
            spec = _DeclaredDefaultU771f.PROPERTY_MAPPING["defaulted_u771f"]
            assert FeatureChainParser._can_skip_required_check(spec) is True

        warnings = _universal_matcher_warnings(caplog, "_DeclaredDefaultU771f")
        assert warnings, "a declared default is optional, so the class is still a universal matcher"

    def test_passthrough_override_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 4: an override that only delegates via super() stays universal -> WARNS.

        This is the DoD "distinguish genuine custom from pass-through" case: a pass-through matcher is
        as universal as the inherited one, so the guard must still fire.
        """
        with caplog.at_level(logging.WARNING):

            class _PassThroughU771d(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771d": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return super().match_feature_group_criteria(feature_name, options, data_access_collection)

            # Precondition: the pass-through override still matches an unrelated name with empty options.
            assert _PassThroughU771d.match_feature_group_criteria("anything_u771d", Options()) is True

        warnings = _universal_matcher_warnings(caplog, "_PassThroughU771d")
        assert warnings, "a pass-through override is still a universal matcher and must warn"

    def test_named_capture_optional_pattern_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 10: a named-capture pattern plus an all-optional mapping is universal via config -> WARNS.

        An unrelated name WITHOUT a chain separator reaches the configuration path, where the sole key is
        optional, so the class matches it with empty options: it is a universal matcher. The definition-time
        probe must confirm universality on a name the pattern does NOT capture. Today's ``__``-bearing probe
        IS captured by the pattern, the captured value fails strict validation, the match returns False, and
        no warning fires. Fixed once the probe stops using a name that contains the chain separator.
        """
        with caplog.at_level(logging.WARNING):

            class _NamedOptionalPatternU771j(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__(?P<mode_u771j>\w+)$"
                PROPERTY_MAPPING = {
                    "mode_u771j": PropertySpec(
                        "mode", allowed_values=("special_u771j",), default=None, strict_validation=True
                    ),
                }

            # Precondition (holds now and after the fix): a name with NO chain separator reaches the config
            # path, where the optional key lets the class match with empty options -> universal.
            assert _NamedOptionalPatternU771j.match_feature_group_criteria("plainunrelatedu771j", Options()) is True

        warnings = _universal_matcher_warnings(caplog, "_NamedOptionalPatternU771j")
        assert warnings, "a named-capture pattern with an all-optional mapping is still a universal matcher"

    def test_empty_mapping_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 11: an explicit empty PROPERTY_MAPPING validates vacuously -> universal matcher -> WARNS.

        An empty mapping enforces nothing, so with empty options it matches any feature name: it is the most
        universal shape there is. Today the guard early-returns on the empty dict and stays silent; it must
        warn once that early return is removed.
        """
        with caplog.at_level(logging.WARNING):

            class _EmptyMappingU771k(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {}

            # Precondition: an empty mapping validates vacuously, so an unrelated name matches -> universal.
            assert _EmptyMappingU771k.match_feature_group_criteria("unrelatedu771k", Options()) is True

        warnings = _universal_matcher_warnings(caplog, "_EmptyMappingU771k")
        assert warnings, "an empty PROPERTY_MAPPING is vacuously universal and must warn"

    def test_warns_exactly_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 15 (pin, once-count): the definition-time warning fires exactly once per class.

        Only ``FeatureChainParserMixin.__init_subclass__`` runs this diagnostic. Mirrors the captureless
        suite's once-count and guards against a future regression that also wires the guard into
        ``FeatureGroup.__init_subclass__``, which would double the warning through the super() chain.
        """
        with caplog.at_level(logging.WARNING):

            class _OnceCountU771o(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771o": PropertySpec("optional", default=None)}

            assert "opt_u771o" in _OnceCountU771o.PROPERTY_MAPPING

        assert len(_universal_matcher_warnings(caplog, "_OnceCountU771o")) == 1


class TestUniversalMatcherDoesNotWarn:
    """The guard stays quiet whenever the class is NOT a universal matcher, or opts out explicitly."""

    def test_unconditionally_required_key_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 2: one unconditionally-required key (NO_DEFAULT, no required_when) -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _RequiredKeyU771b(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "required_u771b": PropertySpec(
                        "required op", allowed_values=("agg_u771b",), strict_validation=True
                    ),
                    "opt_u771b": PropertySpec("optional", default=None),
                }

            # Precondition: the first key is unconditionally required (cannot be skipped).
            required_spec = _RequiredKeyU771b.PROPERTY_MAPPING["required_u771b"]
            assert FeatureChainParser._can_skip_required_check(required_spec) is False

        assert not _universal_matcher_warnings(caplog), "a required key means the matcher is not universal"

    def test_genuine_custom_matcher_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 3: a real, discriminating override rejects the unrelated probe -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _GenuineCustomU771c(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771c": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return str(feature_name) == "specific_u771c"

            # Precondition: the custom matcher really discriminates (True only for its own name).
            assert _GenuineCustomU771c.match_feature_group_criteria("specific_u771c", Options()) is True
            assert _GenuineCustomU771c.match_feature_group_criteria(UNRELATED_NAME_U771, Options()) is False

        assert not _universal_matcher_warnings(caplog), "a genuinely custom matcher is not universal"

    def test_conditional_requirement_fires_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 5: mutually-exclusive required_when keys make empty options a non-match -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _ConditionalU771e(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    COND_KEY_A_U771: PropertySpec(
                        "a, required when b is absent",
                        default=None,
                        required_when=(lambda options: options.get(COND_KEY_B_U771) is None),
                    ),
                    COND_KEY_B_U771: PropertySpec(
                        "b, required when a is absent",
                        default=None,
                        required_when=(lambda options: options.get(COND_KEY_A_U771) is None),
                    ),
                }

            # Precondition: with EMPTY options at least one conditional key is required, so the
            # config match fails and the class is therefore NOT a universal matcher.
            assert _ConditionalU771e.match_feature_group_criteria(UNRELATED_NAME_U771, Options()) is False

        assert not _universal_matcher_warnings(caplog), "a conditional requirement that fires is not universal"

    def test_escape_hatch_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 7: ALLOW_UNIVERSAL_MATCHER = True opts out of the diagnostic -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _EscapeHatchU771g(FeatureChainParserMixin, FeatureGroup):
                ALLOW_UNIVERSAL_MATCHER = True
                PROPERTY_MAPPING = {"opt_u771g": PropertySpec("optional", default=None)}

            assert _EscapeHatchU771g.ALLOW_UNIVERSAL_MATCHER is True

        assert not _universal_matcher_warnings(caplog), "the escape hatch must silence the diagnostic"

    def test_self_referential_required_when_no_spurious_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 12: a self-referential required_when must NOT trigger a spurious class-definition warning.

        An all-conditional mapping whose ``required_when`` references the class's own not-yet-bound name is a
        common authoring style (sklearn's pipeline uses
        ``lambda o: o.get(SklearnPipelineFeatureGroup.KEY) is None``). During ``__init_subclass__`` the class
        name is unbound, so calling the predicate raises NameError. Today the universal-matcher probe runs the
        installed guard, which triggers the predicate; ``check_required_when`` catches the NameError and logs a
        misleading "required_when predicate ... raised ..." warning. A conditional mapping is not a universal
        matcher and must not be probed at class-definition time.
        """
        with caplog.at_level(logging.WARNING):

            class _SelfRefConditionalU771l(FeatureChainParserMixin, FeatureGroup):
                COND_KEY_U771L = "cond_u771l"
                OTHER_KEY_U771L = "other_u771l"
                PROPERTY_MAPPING = {
                    "cond_u771l": PropertySpec(
                        "cond",
                        default=None,
                        required_when=(lambda o: o.get(_SelfRefConditionalU771l.OTHER_KEY_U771L) is None),
                    ),
                    "other_u771l": PropertySpec(
                        "other",
                        default=None,
                        required_when=(lambda o: o.get(_SelfRefConditionalU771l.COND_KEY_U771L) is None),
                    ),
                }

            assert "cond_u771l" in _SelfRefConditionalU771l.PROPERTY_MAPPING

        # 1) A conditional mapping is not warned as a universal matcher (holds now and after the fix).
        assert not _universal_matcher_warnings(caplog, "_SelfRefConditionalU771l")
        # 2) The probe must not run the required_when predicate at class-definition time. Doing so today
        # raises NameError (the class name is unbound during __init_subclass__) and logs this spurious record.
        spurious = [
            record
            for record in caplog.records
            if record.name == FEATURE_CHAIN_PARSER_LOGGER
            and "required_when" in record.getMessage()
            and "raised" in record.getMessage()
        ]
        assert not spurious, "the guard's probe must not run required_when predicates at class definition"

    def test_required_key_with_always_true_matcher_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 13 (pin): an unconditionally-required key keeps the class out of scope even when a custom
        matcher returns True for everything.

        The diagnostic is scoped to the MAPPING SHAPE (all-optional), not "is this matcher universal in
        general". A required key means the class is out of #771 scope, whatever the matcher answers, so the
        probe is never even reached.
        """
        with caplog.at_level(logging.WARNING):

            class _RequiredKeyCustomTrueU771m(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "req_u771m": PropertySpec("required", allowed_values=("v_u771m",), strict_validation=True),
                }

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return True

            # Precondition: the key is unconditionally required, so the mapping shape is not universal.
            required_spec = _RequiredKeyCustomTrueU771m.PROPERTY_MAPPING["req_u771m"]
            assert FeatureChainParser._can_skip_required_check(required_spec) is False

        assert not _universal_matcher_warnings(caplog, "_RequiredKeyCustomTrueU771m"), (
            "a required key keeps the class out of scope even if the matcher returns True for everything"
        )

    def test_probe_exception_contained_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 14 (pin): a matcher that raises on the probe is contained -> no warning, no crash.

        The class-definition probe must never let a matcher exception escape and abort the class body, and a
        raising matcher is not treated as universal.
        """
        with caplog.at_level(logging.WARNING):

            class _RaisingMatcherU771n(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771n": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    raise RuntimeError("boom_u771n")

            # The class body completes: the probe exception did not escape class definition.
            assert "opt_u771n" in _RaisingMatcherU771n.PROPERTY_MAPPING

        assert not _universal_matcher_warnings(caplog, "_RaisingMatcherU771n"), (
            "a probe exception is contained, so the class is not warned as universal"
        )

    def test_contained_probe_logs_str_not_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        """The contained-probe debug record stores str(exc), never the exception object.

        A record holding the exception pins its traceback, frame and the probed class, defeating the
        registry-pollution cleanup under DEBUG logging (the utils.safe_field discipline).
        """
        with caplog.at_level(logging.DEBUG, logger=FEATURE_CHAIN_PARSER_LOGGER):

            class _DebugRaiserU771y(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771y": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    raise RuntimeError("boom_u771y")

            assert "opt_u771y" in _DebugRaiserU771y.PROPERTY_MAPPING

        records = [
            record
            for record in caplog.records
            if "universal-matcher probe" in record.getMessage() and "_DebugRaiserU771y" in record.getMessage()
        ]
        assert records, "expected the contained-probe debug record"
        assert all(not isinstance(arg, BaseException) for record in records for arg in record.args or ()), (
            "the contained-probe record must store str(exc), not the exception, so it cannot pin the class"
        )

    def test_probe_exception_with_failing_str_is_contained(self, caplog: pytest.LogCaptureFixture) -> None:
        """A probe exception whose __str__ also raises is still contained: class creation must not abort.

        str(exc) is a log argument evaluated eagerly regardless of level, so a failing __str__ would escape
        containment even with DEBUG off. The diagnostic degrades to the exception type name instead.
        """

        class _FailingStrErrorU771x(Exception):
            def __str__(self) -> str:
                raise RuntimeError("str_boom_u771x")

        with caplog.at_level(logging.WARNING):

            class _FailingStrMatcherU771x(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771x": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    raise _FailingStrErrorU771x()

            # The class body completes: neither the probe exception nor its failing __str__ escaped.
            assert "opt_u771x" in _FailingStrMatcherU771x.PROPERTY_MAPPING

    def test_probe_exception_with_hostile_getattribute_is_contained(self, caplog: pytest.LogCaptureFixture) -> None:
        """A probe exception whose instance __getattribute__ rejects __str__ is still contained.

        str(exc) uses type-level __str__ lookup, bypassing a hostile __getattribute__; an explicit exc.__str__
        lookup would raise before the guard and abort class creation.
        """

        class _HostileGetattrErrorU771w(Exception):
            def __getattribute__(self, name: str) -> Any:
                if name == "__str__":
                    raise RuntimeError("getattr_boom_u771w")
                return super().__getattribute__(name)

        with caplog.at_level(logging.WARNING):

            class _HostileGetattrMatcherU771w(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771w": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    raise _HostileGetattrErrorU771w()

            # The class body completes: the hostile attribute lookup did not escape.
            assert "opt_u771w" in _HostileGetattrMatcherU771w.PROPERTY_MAPPING


class TestUniversalMatcherMotivation:
    """Behavioral pins that document WHY the guard exists and keep shipped plugins out of scope."""

    def test_inherited_all_optional_matches_unrelated_name(self) -> None:
        """Case 8: without the escape hatch, an all-optional inherited matcher claims an unrelated name.

        This is the motivation for the diagnostic and holds both before and after implementation: the
        guard only warns, it does not change matching behavior.
        """

        class _UniversalMotivationU771h(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {"opt_u771h": PropertySpec("optional", default=None)}

        assert _UniversalMotivationU771h.match_feature_group_criteria(UNRELATED_NAME_U771, Options()) is True

    def test_shipped_aggregated_feature_group_is_not_universal(self) -> None:
        """Case 9: a representative shipped plugin has an unconditionally-required key -> out of scope.

        Pins that shipped plugins do not trip the diagnostic; passes before and after implementation.
        """
        assert any(
            not FeatureChainParser._can_skip_required_check(spec)
            for spec in AggregatedFeatureGroup.PROPERTY_MAPPING.values()
        ), "AggregatedFeatureGroup must keep an unconditionally-required key so it is not a universal matcher"
