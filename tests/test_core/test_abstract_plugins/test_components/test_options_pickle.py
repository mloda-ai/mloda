"""Options.__getstate__: strip non_forwarded_group_keys from .group only when Options is actually
pickled (pickle.dumps), never on normal in-process access, and never on copy/deepcopy (which use
the existing __copy__/__deepcopy__ overrides, not __getstate__)."""

import copy
import pickle  # nosec B403

from mloda.user import Options


class TestOptionsGetstatePickleStripsMarkedKeys:
    def test_pickled_copy_drops_marked_key_from_group(self) -> None:
        options = Options(group={"conn": "handle", "kg_backend": "neo4j"})
        options.mark_non_forwarded("conn")

        restored: Options = pickle.loads(pickle.dumps(options))  # nosec B301

        assert "conn" not in restored.group
        assert restored.group["kg_backend"] == "neo4j"

    def test_pickled_copy_context_unaffected(self) -> None:
        options = Options(group={"conn": "handle"}, context={"trace_id": "abc"})
        options.mark_non_forwarded("conn")

        restored: Options = pickle.loads(pickle.dumps(options))  # nosec B301

        assert restored.context == {"trace_id": "abc"}

    def test_pickled_copy_multiple_marked_keys_all_dropped(self) -> None:
        options = Options(group={"conn": "handle", "other_conn": "handle2", "kg_backend": "neo4j"})
        options.mark_non_forwarded("conn")
        options.mark_non_forwarded("other_conn")

        restored: Options = pickle.loads(pickle.dumps(options))  # nosec B301

        assert "conn" not in restored.group
        assert "other_conn" not in restored.group
        assert restored.group["kg_backend"] == "neo4j"


class TestOptionsGetstatePickleNoMarkedKeys:
    def test_no_marked_keys_group_fully_intact(self) -> None:
        options = Options(group={"kg_backend": "neo4j", "top_k": 5}, context={"trace_id": "abc"})

        restored: Options = pickle.loads(pickle.dumps(options))  # nosec B301

        assert restored.group == options.group
        assert restored.context == options.context


class TestOptionsGetstateDoesNotMutateLiveObject:
    def test_original_keeps_marked_value_after_pickling(self) -> None:
        """Pickling produces a snapshot; the original, live Options must be untouched."""
        options = Options(group={"conn": "handle", "kg_backend": "neo4j"})
        options.mark_non_forwarded("conn")

        pickle.dumps(options)

        assert options.get("conn") == "handle"
        assert options.group["conn"] == "handle"

    def test_original_non_forwarded_group_keys_unchanged_after_pickling(self) -> None:
        options = Options(group={"conn": "handle"})
        options.mark_non_forwarded("conn")

        pickle.dumps(options)

        assert options.non_forwarded_group_keys == frozenset({"conn"})


class TestOptionsGetstateDoesNotAffectCopyModule:
    """copy.copy/copy.deepcopy use __copy__/__deepcopy__, which take priority over __getstate__."""

    def test_copy_preserves_marked_key(self) -> None:
        options = Options(group={"conn": "handle", "kg_backend": "neo4j"})
        options.mark_non_forwarded("conn")

        copied = copy.copy(options)

        assert copied.group["conn"] == "handle"

    def test_deepcopy_preserves_marked_key(self) -> None:
        options = Options(group={"conn": "handle", "kg_backend": "neo4j"})
        options.mark_non_forwarded("conn")

        copied = copy.deepcopy(options)

        assert copied.group["conn"] == "handle"
