"""order_ordered_ids_by_relation must not silently drop an entry when two ids collide on one position."""

from collections import OrderedDict
from uuid import uuid4

from mloda.core.prepare.resolve_links import LinkTrekker


def test_order_ordered_ids_by_relation_keeps_both_ids_that_collide_on_one_position() -> None:
    trekker = LinkTrekker()
    uuid_a = uuid4()
    uuid_b = uuid4()
    uuid_c = uuid4()

    trekker.order = OrderedDict(
        [
            (uuid_a, set()),
            (uuid_b, set()),
            (uuid_c, {uuid_a, uuid_b}),
        ]
    )

    trekker.order_ordered_ids_by_relation()

    assert list(trekker.order.keys()) == [uuid_c, uuid_a, uuid_b]


def test_order_ordered_ids_by_relation_respects_precedence_among_colliding_ids() -> None:
    """Colliding entries (A, B) that both precede D must keep their own relative precedence order."""
    trekker = LinkTrekker()
    uuid_a = uuid4()
    uuid_b = uuid4()
    uuid_c = uuid4()
    uuid_d = uuid4()
    uuid_z = uuid4()

    trekker.order = OrderedDict(
        [
            (uuid_c, {uuid_a}),
            (uuid_a, {uuid_z}),
            (uuid_b, {uuid_a}),
            (uuid_d, {uuid_a, uuid_b}),
        ]
    )

    trekker.order_ordered_ids_by_relation()

    order_index = {uuid_id: pos for pos, uuid_id in enumerate(trekker.order.keys())}

    assert order_index[uuid_c] < order_index[uuid_a]
    assert order_index[uuid_b] < order_index[uuid_a]
    assert order_index[uuid_d] < order_index[uuid_a]
    assert order_index[uuid_d] < order_index[uuid_b]


def test_order_ordered_ids_by_relation_does_not_reorder_an_already_valid_order() -> None:
    """An id with no relation to anything must not be pulled ahead of a later, unrelated-but-required successor."""
    trekker = LinkTrekker()
    uuid_p = uuid4()
    uuid_q = uuid4()
    uuid_r = uuid4()

    trekker.order = OrderedDict(
        [
            (uuid_p, {uuid_q}),
            (uuid_q, set()),
            (uuid_r, set()),
        ]
    )

    trekker.order_ordered_ids_by_relation()

    assert list(trekker.order.keys()) == [uuid_p, uuid_q, uuid_r]


def test_order_ordered_ids_by_relation_treats_a_self_edge_as_a_no_op() -> None:
    """A self-referencing entry imposes no real ordering constraint and must not be pushed after unrelated ids."""
    trekker = LinkTrekker()
    uuid_k = uuid4()
    uuid_m = uuid4()

    trekker.order = OrderedDict(
        [
            (uuid_k, {uuid_k}),
            (uuid_m, set()),
        ]
    )

    trekker.order_ordered_ids_by_relation()

    assert list(trekker.order.keys()) == [uuid_k, uuid_m]
