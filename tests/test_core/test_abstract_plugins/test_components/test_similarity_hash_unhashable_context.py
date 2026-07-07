"""Regression tests for similarity hashing over an unhashable INHERITED context value.

Feature.similarity_hash / base_similarity_hash now fold INHERITED context VALUES into
the grouping hash via _split_context_hashable -> _make_hashable. _make_hashable only
unwraps dict/list/tuple/set; a context value that is itself unhashable and NOT a
container (an object whose __hash__ is None) reaches the outer hash(...) untouched and
raises TypeError. Pre-PR the context was never hashed, so propagated/inherited
unhashable context values are a regression: grouping must not crash on them.
"""

from mloda.user import Feature, Options


class _UnhashableContextValue:
    """A tiny context value that is intentionally unhashable (defines __eq__, no __hash__).

    Defining __eq__ without __hash__ makes Python set __hash__ to None, so hash(instance)
    raises TypeError: unhashable type. This mimics real-world context payloads (e.g. mutable
    config objects) that a consumer may propagate into an input feature's context.
    """

    def __init__(self, token: str) -> None:
        self.token = token

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _UnhashableContextValue) and self.token == other.token


def _feature_with_inherited_unhashable_context(key: str, token: str) -> Feature:
    feature = Feature("grouping_feature", options=Options(group={"data_source": "prod"}))
    feature.options.context[key] = _UnhashableContextValue(token)
    feature.options.inherited_context_keys = frozenset({key})
    return feature


def test_similarity_hash_does_not_crash_on_unhashable_inherited_context() -> None:
    """similarity_hash over an inherited, unhashable, non-container context value returns an int."""
    feature = _feature_with_inherited_unhashable_context("payload", "acme")

    result = feature.similarity_hash(frozenset({"payload"}))

    assert isinstance(result, int)


def test_base_similarity_hash_does_not_crash_on_unhashable_inherited_context() -> None:
    """base_similarity_hash (lenient None-typed grouping) must also survive the unhashable value."""
    feature = _feature_with_inherited_unhashable_context("payload", "acme")

    result = feature.base_similarity_hash(frozenset({"payload"}))

    assert isinstance(result, int)
