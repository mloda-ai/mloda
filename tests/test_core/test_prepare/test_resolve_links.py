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

    assert uuid_a in trekker.order, f"uuid_a was overwritten by the colliding position bug; got: {trekker.order}"
    assert uuid_b in trekker.order
    assert uuid_c in trekker.order
    assert len(trekker.order) == 3
