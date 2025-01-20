from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from uuid import UUID

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.prepare.graph.graph import Graph
from mloda_core.abstract_plugins.components.link import Link


LinkFrameworkTrekker = Tuple[Link, Type[ComputeFrameWork], Type[ComputeFrameWork]]


class LinkTrekker:
    """This class is used to keep track of Links and which children depend on this link."""

    def __init__(self) -> None:
        self.data: Dict[LinkFrameworkTrekker, Set[UUID]] = defaultdict(set)
        self.data_ordered: OrderedDict[LinkFrameworkTrekker, Set[UUID]] = OrderedDict()

        self.order: OrderedDict[UUID, Set[UUID]] = OrderedDict()

    def update(self, key: LinkFrameworkTrekker, value: UUID) -> None:
        self.data[key].add(value)

    def invert_link(
        self, link: Link, left_cfw: Type[ComputeFrameWork], right_cfw: Type[ComputeFrameWork], uuid: UUID
    ) -> None:
        """
        The purpose of this function is to invert left and right of an index during a run.
        """

        # if data is not there, we plug it behind the existing link
        if (link, right_cfw, left_cfw) not in self.data_ordered:
            position = self.get_position(link, left_cfw, right_cfw)
            self.insert_at_position(key=(link, right_cfw, left_cfw), value={uuid}, position=position + 1)
        else:
            self.data_ordered[(link, right_cfw, left_cfw)].add(uuid)

        # adjust self.data
        self.data[(link, right_cfw, left_cfw)].add(uuid)

        # drop the old link. however, as they are one object, we only need to drop it in one of data or data_ordered
        self.data[(link, left_cfw, right_cfw)].remove(uuid)
        if len(self.data[(link, left_cfw, right_cfw)]) == 0:
            del self.data[(link, left_cfw, right_cfw)]
            del self.data_ordered[(link, left_cfw, right_cfw)]

    def get_position(self, link: Link, left_cfw: Type[ComputeFrameWork], right_cfw: Type[ComputeFrameWork]) -> int:
        for i, (k, _) in enumerate(self.data_ordered.items()):
            if k == (link, left_cfw, right_cfw):
                return i

        raise ValueError("Link not found in data ordered!")

    def insert_at_position(self, key: LinkFrameworkTrekker, value: Set[UUID], position: int) -> None:
        items = list(self.data_ordered.items())
        items.insert(position, (key, value))
        self.data_ordered = OrderedDict(items)

    def get_ordered_data(self) -> OrderedDict[LinkFrameworkTrekker, Set[UUID]]:
        """
        We ensure that the data is ordered by the order of the links.
        If we would have:

        links = {Link("inner", left, right_1), Link("inner", right_1, right_2)}

        This would result in the following order in creating the queue:
             1.   (Link("inner", right_1, right_2): {uuid2}
             2.   (Link("inner", left, right_1), left, right_1): {uuid1},
        """

        self.order_links_by_frameworks()
        self.order_ordered_ids_by_relation()
        self.create_data_ordered()
        return self.data_ordered

    def drop_dependency_in_case_of_circular_dependencies(self) -> None:
        def adjust_order(data: Dict[LinkFrameworkTrekker, Set[UUID]], k_out: UUID, k_int: UUID) -> None:
            found_out = None
            found_in = None

            for link_fw, dependant_features in data.items():
                if link_fw[0].uuid == k_out:
                    found_out = (k_out, dependant_features)

                if link_fw[0].uuid == k_int:
                    found_in = (k_in, dependant_features)

            if found_out is None or found_in is None:
                raise ValueError("Link not found in data!")

            if len(found_out[1]) >= len(found_in[1]):
                k_to_drop = found_out[0]
                k_not_to_drop = found_in[0]
            elif len(found_out[1]) < len(found_in[1]):
                k_to_drop = found_in[0]
                k_not_to_drop = found_out[0]
            else:
                raise ValueError("This should not happen!")

            for k_order, v_order in self.order.items():
                if k_not_to_drop != k_order:
                    continue

                if k_to_drop in v_order:
                    v_order.remove(k_to_drop)
                    break

        for k_out, v_out in self.order.items():
            for k_in, v_in in self.order.items():
                if k_out == k_in:
                    continue

                if k_out in v_in:
                    if k_in in v_out:
                        adjust_order(self.data, k_out, k_in)

    def order_ordered_ids_by_relation(self) -> None:
        """
        We have currently this:
        odict_items([(UUID('cad2bdde-3028-47c3-a79b-53110c51a2dd'), {UUID('022d7fe1-2ef6-4d9a-83aa-832738575920')}),
                    (UUID('022d7fe1-2ef6-4d9a-83aa-832738575920'), {UUID('11b118b5-d714-48ff-87ab-190ecd286f61')})])
        or
        odict_items([(UUID('022d7fe1-2ef6-4d9a-83aa-832738575920'), {UUID('11b118b5-d714-48ff-87ab-190ecd286f61')}),
                    (UUID('cad2bdde-3028-47c3-a79b-53110c51a2dd'), {UUID('022d7fe1-2ef6-4d9a-83aa-832738575920')})])

        The second case is not ok, as this would mean that the first link depends on the second link. This is not the case.
        Therefore, we reorder this in here.
        """

        new_order: OrderedDict[UUID, Set[UUID]] = OrderedDict()
        pos_marker: Dict[int, Tuple[UUID, Set[UUID]]] = {}

        for o_pos, (o_uuid, o_set) in enumerate(self.order.items()):
            latest_position = None

            for i_pos, (i_uuid, i_set) in enumerate(self.order.items()):
                if o_pos >= i_pos:
                    continue

                if o_uuid not in i_set:
                    continue

                latest_position = i_pos

            if latest_position is None:
                new_order[o_uuid] = o_set
            else:
                # remembering latest position
                if latest_position in pos_marker:
                    for i in range(latest_position, len(pos_marker)):
                        if i not in pos_marker:
                            latest_position = i + latest_position
                            break
                pos_marker[latest_position] = (o_uuid, o_set)

        if len(pos_marker.keys()):
            max_latest_pos = max(pos_marker.keys())

            for i in range(max_latest_pos + 1):
                if i in pos_marker:
                    uuid, uuid_set = pos_marker[i]
                    new_order[uuid] = uuid_set
                    new_order.move_to_end(uuid)

            self.order = new_order

    def order_links_by_frameworks(self) -> None:
        for trekker, uuids in self.data.items():
            link = trekker[0]
            left_framework = trekker[1]
            right_framework = trekker[2]

            for other_trekker, other_uuids in self.data.items():
                other_link = other_trekker[0]
                other_left_framework = other_trekker[1]
                _ = other_trekker[2]

                if link.uuid == other_link.uuid:
                    continue

                if right_framework == other_left_framework == left_framework:
                    continue

                if right_framework == other_left_framework:
                    if other_link.uuid not in self.order:
                        self.order[other_link.uuid] = {link.uuid}
                    else:
                        self.order[other_link.uuid].add(link.uuid)

        self.drop_dependency_in_case_of_circular_dependencies()

    def create_data_ordered(self) -> None:
        # fill ordered dict by priority

        for o_id, oo_uuids in self.order.items():
            for k, v in self.data.items():
                if k[0].uuid == o_id:
                    self.data_ordered[k] = v

        # fill rest independent of order
        for k, v in self.data.items():
            if k not in self.data_ordered:
                self.data_ordered[k] = v

        if len(self.data.items()) != len(self.data_ordered.items()):
            raise ValueError("Data ordered is not the same length as data!")


class ResolveLinks:
    def __init__(self, graph: Graph, links: Optional[Set[Link]] = None) -> None:
        self.graph = graph
        self.links = links
        self.link_trekker = LinkTrekker()

    def add_links_to_queue(self) -> List[Union[UUID, LinkFrameworkTrekker]]:
        queue = self.graph.queue

        queue_with_link: List[Union[UUID, LinkFrameworkTrekker]] = []
        already_joined: Set[LinkFrameworkTrekker] = set()

        ordered = self.link_trekker.get_ordered_data()

        for uuid in queue:
            for link, link_uuids in ordered.items():
                if link in already_joined:
                    continue

                for link_id in link_uuids:
                    if uuid == link_id:
                        queue_with_link.append(link)
                        already_joined.add(link)

            queue_with_link.append(uuid)

        return queue_with_link

    def resolve_links(self) -> None:
        self.graph.set_direct_parents_for_each_child()
        self.graph.set_all_parents_for_each_child()
        self.graph.set_root_parents_by_direct_()

        self.go_through_each_child_and_its_parents_and_look_for_links()
        self.validate_link_trekker()

    def validate_link_trekker(self) -> None:
        for link_fw_trekker, _ in self.link_trekker.data.items():
            for other_fw_trekker, _ in self.link_trekker.data.items():
                if link_fw_trekker == other_fw_trekker:
                    continue

                link, _, _ = link_fw_trekker
                other, _, _ = other_fw_trekker

                # if feature group match
                if (
                    link.left_feature_group.get_class_name() == other.left_feature_group.get_class_name()
                    and link.right_feature_group.get_class_name() == other.right_feature_group.get_class_name()
                ):
                    # case join different
                    if link.jointype != other.jointype:
                        raise Exception(
                            f"Link {link} and {other} have the same feature groups, but different join types!"
                        )

    def get_link_trekker(self) -> LinkTrekker:
        return self.link_trekker

    def go_through_each_child_and_its_parents_and_look_for_links(self) -> None:
        if not self.links:
            return

        for link in self.links:
            for child, parents in self.graph.parent_to_children_mapping.items():
                for parent_in in parents:
                    for parent_out in parents:
                        if parent_in == parent_out:
                            continue

                        r_left = self.graph.get_nodes()[parent_in]
                        r_right = self.graph.get_nodes()[parent_out]

                        if link.matches(
                            other_left_feature_group=r_left.feature_group_class,
                            other_right_feature_group=r_right.feature_group_class,
                        ):
                            key = self.create_link_trekker_key(
                                link, r_left.feature.compute_frameworks, r_right.feature.compute_frameworks
                            )
                            self.set_link_trekker(key, child)

    def set_link_trekker(self, link_trekker_key: LinkFrameworkTrekker, uuid: UUID) -> None:
        self.link_trekker.update(link_trekker_key, uuid)

    def create_link_trekker_key(
        self,
        link: Link,
        left_frameworks: Optional[Set[Type[ComputeFrameWork]]] = None,
        right_frameworks: Optional[Set[Type[ComputeFrameWork]]] = None,
    ) -> LinkFrameworkTrekker:
        if left_frameworks is None or right_frameworks is None:
            raise ValueError("Left or right frameworks are not set!")
        left_framework = next(iter(left_frameworks))
        right_framework = next(iter(right_frameworks))
        return (link, left_framework, right_framework)
