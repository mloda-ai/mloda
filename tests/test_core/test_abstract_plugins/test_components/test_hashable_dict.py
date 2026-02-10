"""Tests for HashableDict nested structure hashing.

These tests verify that HashableDict can properly hash dictionaries
containing nested structures like dicts, lists, and sets.
"""

import pytest

from mloda.core.abstract_plugins.components.hashable_dict import HashableDict


class TestHashableDictNestedHashing:
    """Tests for hashing HashableDict with nested structures."""

    def test_hash_with_nested_dict_values(self) -> None:
        """HashableDict should handle nested dict values.

        Currently fails with: TypeError: unhashable type: 'dict'
        """
        data = {"outer": {"inner_key": "inner_value"}}
        hashable = HashableDict(data)

        result = hash(hashable)

        assert isinstance(result, int)

    def test_hash_with_list_values(self) -> None:
        """HashableDict should handle list values.

        Currently fails with: TypeError: unhashable type: 'list'
        """
        data = {"items": [1, 2, 3]}
        hashable = HashableDict(data)

        result = hash(hashable)

        assert isinstance(result, int)

    def test_hash_with_set_values(self) -> None:
        """HashableDict should handle set values.

        Currently fails with: TypeError: unhashable type: 'set'
        """
        data = {"tags": {1, 2, 3}}
        hashable = HashableDict(data)

        result = hash(hashable)

        assert isinstance(result, int)

    def test_hash_with_mixed_nested_structures(self) -> None:
        """HashableDict should handle complex mixed nested structures.

        Currently fails with: TypeError: unhashable type: 'dict'
        """
        data = {
            "config": {"nested": {"deep": "value"}},
            "items": [1, 2, {"nested_in_list": True}],
            "simple": "string",
        }
        hashable = HashableDict(data)

        result = hash(hashable)

        assert isinstance(result, int)

    def test_equal_nested_dicts_produce_equal_hashes(self) -> None:
        """Equal HashableDicts with nested structures should have equal hashes.

        Currently fails with: TypeError: unhashable type: 'dict'
        """
        data1 = {"outer": {"inner": "value"}}
        data2 = {"outer": {"inner": "value"}}

        hashable1 = HashableDict(data1)
        hashable2 = HashableDict(data2)

        assert hash(hashable1) == hash(hashable2)
        assert hashable1 == hashable2

    def test_different_nested_dicts_produce_different_hashes(self) -> None:
        """Different HashableDicts with nested structures should have different hashes.

        Currently fails with: TypeError: unhashable type: 'dict'
        """
        data1 = {"outer": {"inner": "value1"}}
        data2 = {"outer": {"inner": "value2"}}

        hashable1 = HashableDict(data1)
        hashable2 = HashableDict(data2)

        assert hash(hashable1) != hash(hashable2)
        assert hashable1 != hashable2


class TestHashableDictSimpleHashing:
    """Tests for hashing HashableDict with simple values (should already work)."""

    def test_hash_with_simple_values(self) -> None:
        """HashableDict should handle simple hashable values."""
        data = {"key": "value", "number": 42}
        hashable = HashableDict(data)

        result = hash(hashable)

        assert isinstance(result, int)

    def test_equal_simple_dicts_produce_equal_hashes(self) -> None:
        """Equal HashableDicts with simple values should have equal hashes."""
        data1 = {"key": "value"}
        data2 = {"key": "value"}

        hashable1 = HashableDict(data1)
        hashable2 = HashableDict(data2)

        assert hash(hashable1) == hash(hashable2)
        assert hashable1 == hashable2
