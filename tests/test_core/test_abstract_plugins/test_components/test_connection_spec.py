"""ConnectionSpec / ConnectionSource: a picklable recipe and its live-vs-spec wrapper.

Covers construction, matches(), equality/hashing, pickling, repr, and DataAccessCollection
accepting a ConnectionSpec as a connection entry. See the connection resolution seam design.
"""

from __future__ import annotations

import pickle  # nosec B403

from mloda.core.abstract_plugins.components.connection_spec import ConnectionSource, ConnectionSpec
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class _SpecFramework(ComputeFramework):
    pass


class _SpecSubFramework(_SpecFramework):
    pass


class _UnrelatedSpecFramework(ComputeFramework):
    pass


class _PicklableLive:
    """Picklable stand-in for a live connection handle."""


class TestConnectionSpecConstruction:
    def test_stores_framework_and_params(self) -> None:
        spec = ConnectionSpec(_SpecFramework, database="x")
        assert spec.framework is _SpecFramework
        assert spec.params == {"database": "x"}

    def test_no_params_yields_empty_dict(self) -> None:
        spec = ConnectionSpec(_SpecFramework)
        assert spec.params == {}


class TestConnectionSpecMatchesWithClass:
    def test_subclass_matches(self) -> None:
        spec = ConnectionSpec(_SpecFramework)
        assert spec.matches(_SpecSubFramework) is True

    def test_unrelated_class_does_not_match(self) -> None:
        spec = ConnectionSpec(_SpecFramework)
        assert spec.matches(_UnrelatedSpecFramework) is False


class TestConnectionSpecMatchesWithString:
    def test_named_class_matches(self) -> None:
        spec = ConnectionSpec("_SpecFramework")
        assert spec.matches(_SpecFramework) is True

    def test_subclass_of_named_class_matches(self) -> None:
        spec = ConnectionSpec("_SpecFramework")
        assert spec.matches(_SpecSubFramework) is True

    def test_unrelated_class_does_not_match(self) -> None:
        spec = ConnectionSpec("_SpecFramework")
        assert spec.matches(_UnrelatedSpecFramework) is False


class TestConnectionSpecEqualityAndHashing:
    def test_equal_framework_and_params_are_equal(self) -> None:
        assert ConnectionSpec(_SpecFramework, database="x") == ConnectionSpec(_SpecFramework, database="x")

    def test_differing_params_are_not_equal(self) -> None:
        assert ConnectionSpec(_SpecFramework, database="x") != ConnectionSpec(_SpecFramework, database="y")

    def test_equal_specs_hash_equal(self) -> None:
        left = ConnectionSpec(_SpecFramework, database="x")
        right = ConnectionSpec(_SpecFramework, database="x")
        assert hash(left) == hash(right)

    def test_dict_valued_param_stays_hashable(self) -> None:
        spec = ConnectionSpec(_SpecFramework, config={"a": 1})
        collected = {spec}
        assert spec in collected


class TestConnectionSpecPickle:
    def test_round_trip_preserves_framework_and_params(self) -> None:
        spec = ConnectionSpec(_SpecFramework, database="x")
        restored = pickle.loads(pickle.dumps(spec))  # nosec B301
        assert restored.framework is _SpecFramework
        assert restored.params == {"database": "x"}


class TestConnectionSpecRepr:
    def test_repr_contains_framework_name_and_params(self) -> None:
        spec = ConnectionSpec(_SpecFramework, database="x")
        text = repr(spec)
        assert "_SpecFramework" in text
        assert "database" in text
        assert "x" in text


class TestConnectionSourceConstruction:
    def test_live_source(self) -> None:
        obj = _PicklableLive()
        source = ConnectionSource(live=obj)
        assert source.live is obj
        assert source.spec is None
        assert source.live_dropped is False

    def test_spec_source(self) -> None:
        spec = ConnectionSpec(_SpecFramework)
        source = ConnectionSource(spec=spec)
        assert source.spec is spec


class TestConnectionSourceFromEntry:
    def test_from_entry_with_spec(self) -> None:
        spec = ConnectionSpec(_SpecFramework)
        source = ConnectionSource.from_entry(spec)
        assert source.spec is spec
        assert source.live is None

    def test_from_entry_with_live_object(self) -> None:
        obj = _PicklableLive()
        source = ConnectionSource.from_entry(obj)
        assert source.live is obj
        assert source.spec is None


class TestConnectionSourcePickle:
    def test_live_only_source_drops_live_and_marks_dropped(self) -> None:
        obj = _PicklableLive()
        source = ConnectionSource(live=obj)
        restored = pickle.loads(pickle.dumps(source))  # nosec B301
        assert restored.live is None
        assert restored.spec is None
        assert restored.live_dropped is True

    def test_spec_source_survives_round_trip(self) -> None:
        spec = ConnectionSpec(_SpecFramework, database="x")
        source = ConnectionSource(spec=spec)
        restored = pickle.loads(pickle.dumps(source))  # nosec B301
        assert restored.spec == spec
        assert restored.live_dropped is False


class TestDataAccessCollectionAcceptsConnectionSpec:
    def test_resolve_returns_the_spec(self) -> None:
        spec = ConnectionSpec(_SpecFramework, database="x")
        dac = DataAccessCollection(connections={spec})
        assert dac.resolve("connection") is spec
