from copy import copy, deepcopy
from typing import Any, Generator, List, Optional, Set, Tuple, Type, Dict, Union
from uuid import UUID

from mloda_core.abstract_plugins.components.index.index import Index

from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import (
    ApiInputDataCollection,
)
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda_core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.global_filter import GlobalFilter
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.prepare.joinstep_collection import JoinStepCollection
from mloda_core.prepare.graph.graph import Graph
from mloda_core.prepare.resolve_graph import PlannedQueue
from mloda_core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda_core.core.step.feature_group_step import FeatureGroupStep
from mloda_core.core.step.join_step import JoinStep
from mloda_core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.link import JoinType, Link
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class ExecutionPlan:
    def __init__(
        self,
        global_filter: Optional[GlobalFilter] = None,
        api_input_data_collection: Optional[ApiInputDataCollection] = None,
    ) -> None:
        self.tfs_collecion: Set[TransformFrameworkStep] = set()
        self.joinstep_collection = JoinStepCollection()
        self.global_filter = global_filter
        self.api_input_data_collection = api_input_data_collection

        # Helper variable
        self.feature_set_collections: List[Set[UUID]] = []

    def __iter__(self) -> Generator[Union[TransformFrameworkStep, JoinStep, FeatureGroupStep], None, None]:
        yield from self.execution_plan

    def __len__(self) -> int:
        return len(self.execution_plan)

    def create_execution_plan(self, queue: PlannedQueue, graph: Graph, link_trekker: LinkTrekker) -> None:
        child_links = self.invert_link_trekker(link_trekker)
        pre_execution_plan = self.add_feature_group_step(queue, graph.parent_to_children_mapping, child_links)
        fw_execution_plan = self.add_joinstep(pre_execution_plan, link_trekker, graph)
        self.execution_plan = self.add_tfs(fw_execution_plan, graph)

    def add_feature_group_step(
        self,
        queue: PlannedQueue,
        parent_to_children_mapping: Dict[UUID, Set[UUID]],
        child_links: Dict[UUID, Set[LinkFrameworkTrekker]],
    ) -> List[Union[LinkFrameworkTrekker, FeatureGroupStep]]:
        pre_execution_plan: List[Union[LinkFrameworkTrekker, FeatureGroupStep]] = []

        for element in queue:
            if isinstance(element[0], Link):
                pre_execution_plan.append(element)
                continue

            elif issubclass(element[0], AbstractFeatureGroup):
                if not isinstance(element[1], set):
                    raise ValueError(f"Element {element} is not a valid element.")

                links_pre_calulated = self.retrieve_links_which_must_be_calculated_before(element[1], child_links)
                feature_group_steps = self.run_feature_group(element, parent_to_children_mapping, links_pre_calulated)
                for fg_step in feature_group_steps.values():
                    pre_execution_plan.append(fg_step)

            else:
                raise ValueError(f"Element {element} is not a valid element.")
        return pre_execution_plan

    def add_joinstep(
        self,
        pre_execution_plan: List[Union[LinkFrameworkTrekker, FeatureGroupStep]],
        link_trekker: LinkTrekker,
        graph: Graph,
    ) -> List[Union[JoinStep, FeatureGroupStep]]:
        fw_execution_plan: List[Union[JoinStep, FeatureGroupStep]] = []

        for pex in pre_execution_plan:
            if isinstance(pex, tuple):
                js = self.run_link(pex, link_trekker, graph, pre_execution_plan)
                if js is not None:
                    fw_execution_plan.append(js)
            else:
                fw_execution_plan.append(pex)

        fw_execution_plan = self.handle_append_or_union_joinstep(fw_execution_plan)

        return fw_execution_plan

    def handle_append_or_union_joinstep(
        self,
        fw_execution_plan: List[Union[JoinStep, FeatureGroupStep]],
    ) -> List[Union[JoinStep, FeatureGroupStep]]:
        """
        This part is for the case that we have a join step with append or union.

        Example:
        UUID1 - UUID2 : UUID2 - UUID3 -> UUID1 must wait for UUID2 completion
        -> we add this to the required_uuids of the join step of UUID1

        We use two loops to make sure that we have the correct order.
        1) We map the left framework uuid to the link uuid
        2) We use the mapping to update the required_uuids of the join step
        """

        map_left_framework_uuid_to_link_uuid: Dict[UUID, Set[UUID]] = defaultdict(set)

        # Map the left framework uuid to the link uuid
        for fw in fw_execution_plan:
            if isinstance(fw, JoinStep) and fw.link.jointype in (JoinType.APPEND, JoinType.UNION):
                if len(fw.left_framework_uuids) > 1:
                    raise ValueError("This should not happen.")
                map_left_framework_uuid_to_link_uuid[next(iter(fw.left_framework_uuids))].add(fw.link.uuid)

        # Use the mapping to update the required_uuids of the join step
        for fw in fw_execution_plan:
            if isinstance(fw, JoinStep) and fw.link.jointype in (JoinType.APPEND, JoinType.UNION):
                if len(fw.right_framework_uuids) > 1:
                    raise ValueError("This should not happen.")

                right_framework_uuid = next(iter(fw.right_framework_uuids))
                required = map_left_framework_uuid_to_link_uuid.get(right_framework_uuid)
                if required is not None:
                    fw.required_uuids.update(required)

        return fw_execution_plan

    def fill_tfs_by_joinstep(self, ep: JoinStep) -> TransformFrameworkStep:
        """
        We switch here only the feature group, as the other is already switched during run_link
        """

        if ep.link.jointype == JoinType.RIGHT:
            from_feature_group = ep.link.left_feature_group
            to_feature_group = ep.link.right_feature_group
        else:
            from_feature_group = ep.link.right_feature_group
            to_feature_group = ep.link.left_feature_group

        return TransformFrameworkStep(
            from_framework=ep.right_framework,
            to_framework=ep.left_framework,
            required_uuids=deepcopy(ep.required_uuids),
            from_feature_group=from_feature_group,
            to_feature_group=to_feature_group,
            link_id=ep.link.uuid,
            right_framework_uuids=ep.right_framework_uuids,
        )

    def add_tfs(
        self, execution_plan: List[Union[JoinStep, FeatureGroupStep]], graph: Graph
    ) -> List[Union[TransformFrameworkStep, JoinStep, FeatureGroupStep]]:
        new_execution_plan: List[Union[TransformFrameworkStep, JoinStep, FeatureGroupStep]] = []

        left_join_frameworks: Set[JoinStep] = {ep for ep in execution_plan if isinstance(ep, JoinStep)}
        need_to_upload_collector: Set[UUID] = set()

        for ep in execution_plan:
            if isinstance(ep, JoinStep):
                if ep.left_framework != ep.right_framework:
                    new_tfs = self.fill_tfs_by_joinstep(ep)

                    if new_tfs not in self.tfs_collecion:
                        self.tfs_collecion.add(new_tfs)
                        new_execution_plan.append(new_tfs)
                        ep.required_uuids.add(new_tfs.uuid)

                    need_to_upload_collector.update(ep.right_framework_uuids)

                    # We are updating the required uuids after the tfs is added as this makes sure, that the TFS can run in parallel before the join.
                    ep.required_uuids.update(self.joinstep_collection.get_required_join_uuids(ep))
                else:
                    # We need to do two things:
                    # 1) right feature group of the join step needs to know of the link, so that the cfw can be used by the joinstep
                    # 2) The child feature using this join needs to know which cfw to use. We use the tfs vehicle for this.
                    store_val = None

                    for inner_ep in execution_plan:
                        if isinstance(inner_ep, FeatureGroupStep):
                            # 1) We do 1 here:
                            for uuid in inner_ep.get_uuids():
                                if uuid in ep.right_framework_uuids:
                                    # add the link uuid to the children_if_root of the right feature group
                                    inner_ep.add_value_to_children_if_root(ep.link.uuid)

                                    # add to upload as this is the right feature group gets accessed in mp by other process
                                    need_to_upload_collector.update(ep.right_framework_uuids)
                                    break

                                if uuid in ep.left_framework_uuids:
                                    # add the link uuid to the children_if_root of the left feature group

                                    store_val = uuid

                            if store_val is None:
                                continue

                            # Check if any element of ep.left_framework_uuids is in inner_ep.required_uuids
                            # same for right framework
                            if any(elem in inner_ep.required_uuids for elem in ep.left_framework_uuids) and any(
                                elem in inner_ep.required_uuids for elem in ep.right_framework_uuids
                            ):
                                if ep.link.jointype in (JoinType.APPEND, JoinType.UNION):
                                    self.set_store_value_to_left_most_index_and_update_feature_group(
                                        inner_ep, store_val
                                    )
                                else:
                                    inner_ep.tfs_ids = {store_val}
                                    inner_ep.features.any_uuid = (
                                        store_val  # Resets the any_uuid to one of the left side
                                    )

            elif isinstance(ep, FeatureGroupStep):
                if ep.features.any_uuid is None:
                    raise ValueError(f"Feature group {ep.feature_group} has no uuid.")

                parents = graph.parent_to_children_mapping[ep.features.any_uuid]
                parent_parents = self.get_parent_parents(parents, graph)

                for parent in parents:
                    match = set()
                    parent_node_property = graph.get_nodes()[parent]

                    for js in left_join_frameworks:
                        found = js.matched(ep.compute_framework, parent_node_property.feature.uuid)
                        if found:
                            match.add(found)
                            break
                    if match:
                        # We add the uuid of the joinstep to the required_uuids of the feature group.
                        ep.required_uuids.union(match)
                        continue

                    # We only want to add TFS for direct parents and not for parent parents.
                    if parent in parent_parents:
                        continue

                    if ep.compute_framework != parent_node_property.feature.get_compute_framework():
                        new_tfs = TransformFrameworkStep(
                            from_framework=parent_node_property.feature.get_compute_framework(),
                            to_framework=ep.compute_framework,
                            required_uuids={parent},
                            from_feature_group=parent_node_property.feature_group_class,
                            to_feature_group=ep.feature_group,
                        )
                        if new_tfs not in self.tfs_collecion:
                            self.tfs_collecion.add(new_tfs)
                            new_execution_plan.append(new_tfs)
                            ep.required_uuids.add(new_tfs.uuid)

                        # We update the any_uuid of the feature group to the uuid of the TFS.
                        # This way we make sure that the TFS is used later.
                        ep.tfs_ids.add(new_tfs.uuid)

                        need_to_upload_collector.add(parent)

            else:
                raise ValueError(f"Element {ep} is not a valid element.")
            new_execution_plan.append(ep)

            # We define that every parent of a transform framework step needs to be uploaded.
            # This step is only relevant for multi processing.
            for _ep in new_execution_plan:
                if isinstance(_ep, FeatureGroupStep):
                    if _ep.features.any_uuid in need_to_upload_collector:
                        _ep.need_to_upload = True

        # 1.7.2024
        # print()
        # for ep in new_execution_plan:
        #    print("--------")
        #    if isinstance(ep, FeatureGroupStep):
        #        print("FGS", ep.feature_group.get_class_name(), ep.features.get_all_feature_ids())
        #        print(ep.features.get_all_names())
        #        print(ep.children_if_root)
        #        # print(next(iter(ep.features.features)).compute_frameworks)
        #        print(ep.required_uuids)
        #    elif isinstance(ep, TransformFrameworkStep):
        #        print("TFS")
        #        print(ep.from_feature_group.get_class_name(), " -> ", ep.to_feature_group.get_class_name())
        #        print(ep.from_framework.get_class_name(), " -> ", ep.to_framework.get_class_name())
        #        print(ep.required_uuids)
        #    elif isinstance(ep, JoinStep):
        #        print("JOIN")
        #        print(ep.link.uuid)
        #        print(ep.left_framework_uuids)
        #        print(ep.right_framework_uuids)
        #       print(ep.right_framework.get_class_name(), " -> ", ep.left_framework.get_class_name())
        #        print(ep.required_uuids)
        # print("###############################")

        return new_execution_plan

    def set_store_value_to_left_most_index_and_update_feature_group(
        self, inner_ep: FeatureGroupStep, store_val: UUID
    ) -> None:
        """
        Sets the `store_val` to the left-most index and updates the given feature group step.

        This is during runtime used to identify correct compute framework.

        Args:
            inner_ep (FeatureGroupStep): The step to update.
            store_val (UUID): The value to set as the latest UUID.
        """
        joinsteps = self.joinstep_collection.collection

        # Step 1: Identify all left-most and right-most indexes
        left_indexes: Set[Index] = set()
        right_indexes: Set[Index] = set()

        for js, _ in joinsteps.items():
            # Skip if the index does not belong to the FeatureGroupStep.
            if js.link.left_feature_group != inner_ep.feature_group:
                continue

            if not left_indexes:
                # Initialize with the first left and right indexes
                left_indexes.add(js.link.left_index)
                right_indexes.add(js.link.right_index)
                continue

            elif js.link.left_index in right_indexes:
                # If the left index is already in the right set, update both
                right_indexes.add(js.link.left_index)
                right_indexes.add(js.link.right_index)
                continue
            else:
                # Otherwise, add new left and right indexes
                left_indexes.add(js.link.left_index)
                right_indexes.add(js.link.right_index)

        # Step 2: Reduce to a single left-most index (Should be the only one left)
        for js, _ in joinsteps.items():
            _right = js.link.right_index
            # Use a copy of left_indexes to safely modify the set
            for left_index in list(left_indexes):
                if left_index == _right:
                    left_indexes.remove(left_index)

        if len(left_indexes) == 0:
            return

        if len(left_indexes) > 1:
            raise ValueError("Expected exactly one left-most index, but found multiple or none.")

        left_most_index = next(iter(left_indexes))  # Extract the single left-most index

        # Step 3: Update the relevant fields in `inner_ep` based on conditions
        right_memory_index: Set[Index] = set()

        for js, _ in joinsteps.items():
            # Skip if the left index is already in the memory index
            if right_memory_index:
                if js.link.left_index in (right_memory_index):
                    continue

            # Initialize the memory index with the first right index
            if not right_memory_index:
                right_memory_index.add(js.link.right_index)

            # Update the UUIDs only if the conditions are met: it is a left most index and is part of the joinstep left framework uuid
            if store_val == next(iter(js.left_framework_uuids)) and left_most_index == js.link.left_index:
                inner_ep.tfs_ids = {store_val}
                inner_ep.features.any_uuid = store_val

    def get_parent_parents(self, parents: Set[UUID], graph: Graph) -> Set[UUID]:
        parent_parents = set()
        for parent in parents:
            parent_parent = graph.parent_to_children_mapping[parent]
            if len(parent_parent) > 0:
                parent_parents.update(parent_parent)
        return parent_parents

    def run_link(
        self,
        link_fw: LinkFrameworkTrekker,
        link_trekker: LinkTrekker,
        graph: Graph,
        pre_execution_plan: List[Union[LinkFrameworkTrekker, FeatureGroupStep]],
    ) -> JoinStep | None:
        link = link_fw[0]
        left_framework = link_fw[1]
        right_framework = link_fw[2]

        # Switch left and right index if join type is right, as the algorithm does not care about right or left.
        if link.jointype == JoinType.RIGHT:
            left_framework = link_fw[2]
            right_framework = link_fw[1]

        # This gets the id of the children which needs the link to be calculated.
        children_uuids: Set[UUID] = set()

        for stored_links, uuids in link_trekker.data.items():
            if link_fw == stored_links:
                children_uuids.update(uuids)
            # this part is not working!

        if len(children_uuids) == 0:
            # This is the case if we invert right/left index.
            left_framework = link_fw[2]
            right_framework = link_fw[1]

            for stored_links, uuids in link_trekker.data.items():
                if (link, left_framework, right_framework) == stored_links:
                    children_uuids.update(uuids)

            if len(children_uuids) == 0:
                raise ValueError(f"Link {link} has no matching uuids.")

        children_uuids = self.reduce_children_to_one_level(children_uuids, graph)

        # This gets the parent ids of the joinstep, which needs to be calulated before the link.
        required_uuids: Set[UUID] = set()
        for uuid in children_uuids:
            required_uuids.update(graph.parent_to_children_mapping[uuid])

        # This filters the required_uuids to only the one with the final compute framework.
        left_framework_uuids: Set[UUID] = set()
        right_framework_uuids: Set[UUID] = set()

        for uuid in required_uuids:
            if graph.get_nodes()[uuid].feature.get_compute_framework() == left_framework:
                left_framework_uuids.add(uuid)

            if graph.get_nodes()[uuid].feature.get_compute_framework() == right_framework:
                right_framework_uuids.add(uuid)

        # The order shows which items should be added first.
        # Thus, we need to make sure that higher orderered links are calculated first.
        for k, v in link_trekker.order.items():
            if link.uuid in v:
                required_uuids.add(k)

        # Potential  -> This should be the feature uuid of the child of the joinstep. Can this be more than 1?
        # This part can be dropped if we have more tests.
        # if len(children_uuids) > 1:
        #    raise ValueError("This is not supported yet.")

        # This part is for handling specific join cases. Currently, we only deal with equal feature groups.
        for children_uuid in children_uuids:
            children_fw = graph.get_nodes()[children_uuid].feature.get_compute_framework()

            # This runs with the assumption that children_uuids is exactly 1.
            # result = True
            result = self.is_valid_join_step(link_fw, children_fw, children_uuid, graph)
            if result is False:
                return None
            elif result is True:
                pass
            else:
                left_framework_uuids, right_framework_uuids = result

        if link.jointype in (JoinType.APPEND, JoinType.UNION):
            js = self.create_joinstep_in_case_of_append_or_union(
                link, link_fw, required_uuids, graph, pre_execution_plan
            )
        else:
            js = JoinStep(
                link, left_framework, right_framework, required_uuids, left_framework_uuids, right_framework_uuids
            )

        # This makes sure that we do not write on the same datasets due to overlapping joins at once.
        self.joinstep_collection.add(js)
        return js

    def find_fg_per_uuid(
        self, pre_execution_plan: List[Union[LinkFrameworkTrekker, FeatureGroupStep]], uuid: UUID
    ) -> Type[AbstractFeatureGroup]:
        """
        This function finds the feature group per UUID in the pre_execution_plan.

        This can certainly be optimized, but for now, this is the easiest.
        """
        for element in pre_execution_plan:
            if isinstance(element, FeatureGroupStep):
                if uuid in element.get_uuids():
                    return element.feature_group
        raise ValueError(f"Feature group for UUID {uuid} not found.")

    def create_joinstep_in_case_of_append_or_union(
        self,
        link: Link,
        link_fw: LinkFrameworkTrekker,
        required_uuids: Set[UUID],
        graph: Graph,
        pre_execution_plan: List[Union[LinkFrameworkTrekker, FeatureGroupStep]],
    ) -> JoinStep:
        """
        Create a JoinStep for APPEND or UNION operations in the framework execution plan.

        This function identifies the left and right feature UUIDs required for a join operation,
        validates the frameworks and indices, and constructs a JoinStep.
        """

        # Unpack link-related data
        left_index, right_index = link.left_index, link.right_index
        left_feature_group, right_feature_group = link.left_feature_group, link.right_feature_group

        # Initialize variables for feature UUIDs and frameworks
        left_feature_uuid = None
        right_feature_uuid = None
        left_framework, right_framework = link_fw[1], link_fw[2]

        # Identify the left and right feature UUIDs
        for uuid in required_uuids:
            # Skip non-feature UUIDs
            if uuid not in graph.get_nodes():
                continue

            # Get the feature, its index and feature groups
            feature = graph.get_nodes()[uuid].feature
            feature_feature_group = self.find_fg_per_uuid(pre_execution_plan, uuid)
            feature_index = feature.index
            if feature_index is None:
                continue

            # Match the left index and feature group
            if left_index == feature_index and feature_feature_group == left_feature_group:
                if left_feature_uuid is not None:
                    raise ValueError(f"Are the indexes for append or union set double? {left_index}")
                left_framework = feature.get_compute_framework()
                left_feature_uuid = uuid

            # Match the right index and feature group
            if right_index == feature_index and feature_feature_group == right_feature_group:
                if right_feature_uuid is not None:
                    raise ValueError(f"Are the indexes for append or union set double? {right_index}")
                right_feature_uuid = uuid
                right_framework = feature.get_compute_framework()

        # Validate that both feature UUIDs are identified
        if left_feature_uuid is None or right_feature_uuid is None:
            raise ValueError(
                f"Are the indexes for the append or union set correctly? {left_index.index, right_index.index}"
            )

        # Sanity check for framework consistency
        if link_fw[1] != left_framework:
            raise ValueError(
                f"Left framework is not the same as the left framework of the link. {left_framework}. This is a sanity check!"
            )
        if link_fw[2] != right_framework:
            raise ValueError(
                f"Right framework is not the same as the right framework of the link. {right_framework}. This is a sanity check!"
            )

        return JoinStep(
            link=link,
            left_framework=left_framework,
            right_framework=right_framework,
            required_uuids={left_feature_uuid, right_feature_uuid},
            left_framework_uuids={left_feature_uuid},
            right_framework_uuids={right_feature_uuid},
        )

    def reduce_children_to_one_level(self, children_uuids: Set[UUID], graph: Graph) -> Set[UUID]:
        """
        We reduce the children to one level. This is needed for the joinstep creation.
        """

        new_children_uuids: Set[UUID] = copy(children_uuids)
        for child in children_uuids:
            child_of_child = graph.adjacency_list[child]

            for c_o_c in child_of_child:
                if c_o_c in children_uuids:
                    new_children_uuids.remove(c_o_c)

        return new_children_uuids

    def is_valid_join_step(
        self,
        link_fw: LinkFrameworkTrekker,
        children_fw: type[ComputeFrameWork],
        children_uuid: UUID,
        graph: Graph,
    ) -> bool | Tuple[Set[UUID], Set[UUID]]:
        """Identify if the join is valid. If not, this marks it as invalid and returns False."""

        # Check that we handle links with equal feature groups specifically!
        if link_fw[0].left_feature_group == link_fw[0].right_feature_group:
            result = self.case_link_equal_feature_groups(link_fw, children_fw, children_uuid, graph)
            if result is False:
                return False
            return result

        # Check that we handle links where left cfw == children cfw
        if link_fw[1] == children_fw:
            result = self.case_link_fw_is_equal_to_children_fw(link_fw, children_uuid, graph)
            if result is False:
                return False
            return result
        return True

    def case_link_fw_is_equal_to_children_fw(
        self, link_fw: LinkFrameworkTrekker, children_uuid: UUID, graph: Graph
    ) -> bool | Tuple[Set[UUID], Set[UUID]]:
        # check that we only support non-right joins for equal feature groups
        if link_fw[0].jointype == JoinType.RIGHT:
            raise Exception(f"Right joins are not supported for equal feature groups in the join. link: {link_fw[0]}")

        # get feature which could be left
        parents = graph.parent_to_children_mapping[children_uuid]
        local_feature_set_collection = deepcopy(self.feature_set_collections)
        feature_set_collection_per_uuid = self.find_feature_uuids(parents, local_feature_set_collection)

        if len(feature_set_collection_per_uuid) == 0:
            raise ValueError("Feature set collection per uuid is None. This should not happen.")

        unique_solution_counter = 0
        left_uuids = None
        right_uuids = None

        for uuid, uuid_complete in feature_set_collection_per_uuid.items():
            # get the feature set collection, where feature cfw = left link cfw
            if link_fw[1] != graph.nodes[uuid].feature.get_compute_framework():
                continue

            if link_fw[0].left_feature_group != graph.nodes[uuid].feature_group_class:
                continue

            # loop over all other feature set collections
            for _uuid, _uuid_complete in feature_set_collection_per_uuid.items():
                if uuid == _uuid:
                    continue

                # get the feature set collection, where feature cfw = right link cfw
                if link_fw[2] != graph.nodes[_uuid].feature.get_compute_framework():
                    continue

                if link_fw[0].right_feature_group != graph.nodes[_uuid].feature_group_class:
                    continue

                if left_uuids is None:
                    left_uuids = uuid_complete
                    right_uuids = _uuid_complete
                    unique_solution_counter += 1
                    continue

                if left_uuids == uuid_complete and right_uuids == _uuid_complete:
                    continue

                unique_solution_counter += 1

        if unique_solution_counter == 1:
            if left_uuids is None or right_uuids is None:
                raise ValueError("This should not happen.")
            return (left_uuids, right_uuids)
        elif unique_solution_counter == 0:
            return False
        else:
            raise ValueError(
                "There are more than one solution for the join. This should not happen. If you have this occurence, please check your logic, but you can also contact the developers, as we skipped this algorithm part for now."
            )

    def case_link_equal_feature_groups(
        self,
        link_fw: LinkFrameworkTrekker,
        children_fw: type[ComputeFrameWork],
        children_uuid: UUID,
        graph: Graph,
    ) -> bool | Tuple[Set[UUID], Set[UUID]]:
        """
        If we have equal feature groups in the link object, this creates an interesting scenario.

        The algorithm does not know in which order it should join these features.
        We handle this case with some assumptions:

        1) We only support non-right joins for equal feature groups.
        2) Left join cfw should be the child cfw and the left feature cfw.
        3) We only support one solution for the join.

        I have for now not thought if this is algorithmically enough for all cases.
        If that is the case, we might need to adjust the graph algorithm part.

        To date, my first concern is that people use this framework.
        If you find a use case needing different support here, please contact mloda developers.
        """

        # check that we only support non-right joins for equal feature groups
        if link_fw[0].jointype == JoinType.RIGHT:
            raise Exception(f"Right joins are not supported for equal feature groups in the join. link: {link_fw[0]}")

        # check that the compute framework of the child_fw is similar to the left cfw as this is the target cfw
        if link_fw[1] != children_fw:
            return False

        # get feature which could be left
        parents = graph.parent_to_children_mapping[children_uuid]
        local_feature_set_collection = deepcopy(self.feature_set_collections)
        feature_set_collection_per_uuid = self.find_feature_uuids(parents, local_feature_set_collection)

        if len(feature_set_collection_per_uuid) == 0:
            raise ValueError("Feature set collection per uuid is None. This should not happen.")

        unique_solution_counter = 0
        left_uuids = None
        right_uuids = None

        for uuid, uuid_complete in feature_set_collection_per_uuid.items():
            # get the feature set collection, where feature cfw = left link cfw
            if link_fw[1] != graph.nodes[uuid].feature.get_compute_framework():
                continue

            if link_fw[0].left_pointer is not None:
                if not self.check_pointer(link_fw[0].left_pointer, link_fw, graph, uuid):
                    continue

            # loop over all other feature set collections
            for _uuid, _uuid_complete in feature_set_collection_per_uuid.items():
                if uuid == _uuid:
                    continue

                # get the feature set collection, where feature cfw = right link cfw
                if link_fw[2] != graph.nodes[_uuid].feature.get_compute_framework():
                    continue

                if link_fw[0].right_pointer is not None:
                    if not self.check_pointer(
                        link_fw[0].right_pointer,
                        link_fw,
                        graph,
                        _uuid,
                    ):
                        continue
                # This should be the only solution
                left_uuids = uuid_complete
                right_uuids = _uuid_complete
                unique_solution_counter += 1

        # handle append, union
        if link_fw[0].jointype in (JoinType.APPEND, JoinType.UNION):
            if left_uuids is None or right_uuids is None:
                raise ValueError(
                    "This should not happen. Did you set an index for the append or union? Are the features unique? Link and Hash are not unique properties. In this, case, set an arbritarys options."
                )
            if unique_solution_counter > 0:
                return (left_uuids, right_uuids)
            else:
                return False

        if unique_solution_counter == 1:
            if left_uuids is None or right_uuids is None:
                raise ValueError("This should not happen.")
            return (left_uuids, right_uuids)
        elif unique_solution_counter == 0:
            return False
        else:
            raise ValueError(
                "There are more than one solution for the join. This should not happen. If you have this occurence, please check your logic, but you can also contact the developers, as we skipped this algorithm part for now."
            )

    def check_pointer(
        self, pointer_dict: Dict[str, Any], link_fw: LinkFrameworkTrekker, graph: Graph, uuid: UUID
    ) -> bool:
        if link_fw[0].right_pointer is None:
            raise ValueError("This should not happen. If one pointer is set, the other should be set as well.")

        if link_fw[0].left_pointer is None:
            raise ValueError("This should not happen. If one pointer is set, the other should be set as well.")

        for k, v in graph.nodes[uuid].feature.options.data.items():
            for _k, _v in pointer_dict.items():
                if k == _k and v == _v:
                    return True
        return False

    def find_feature_uuids(
        self, parents: Set[UUID], local_feature_set_collection: List[Set[UUID]]
    ) -> Dict[UUID, Set[UUID]]:
        """
        We group the feature_uuids by the feature_set_collection, which represent features of one concrete feature group (step).
        """
        feature_set_collection_per_uuid = defaultdict(set)
        already_used_parents = set()
        for parent in parents:
            if parent in already_used_parents:
                continue
            for feature_uuids in local_feature_set_collection:
                if parent in feature_uuids:
                    feature_set_collection_per_uuid[parent].update(feature_uuids)
                    already_used_parents.update(feature_uuids)
        return feature_set_collection_per_uuid

    def run_feature_group(
        self,
        feature_group_features: Tuple[Type[AbstractFeatureGroup], Set[Feature]],
        parent_to_children_mapping: Dict[UUID, Set[UUID]],
        pre_required_uuids: Set[UUID],
    ) -> Dict[int, FeatureGroupStep]:
        feature_group, features = feature_group_features[0], feature_group_features[1]
        features_grouped_by_framework_and_options = self.group_features_by_compute_framework_and_options(features)

        fg_steps = {}

        root_parent_children_mapping = self.get_parent_children_mapping(parent_to_children_mapping)

        for f_hash, features in features_grouped_by_framework_and_options.items():
            pre_calculated = self.retrieve_nodes_which_must_be_calculated_before(features, parent_to_children_mapping)
            pre_calculated.update(copy(pre_required_uuids))

            cf = next(iter(features)).get_compute_framework()

            children_if_root = set()
            for feature in features:
                if feature.uuid in root_parent_children_mapping:
                    children_if_root.update(root_parent_children_mapping[feature.uuid])

            feature_set = FeatureSet()
            for feature in features:
                feature_set.add(feature)
                feature.name

            self.feature_set_collections.append(feature_set.get_all_feature_ids())

            self.add_artifact_to_feature_set(feature_group, feature_set)
            self.add_single_filters_to_feature_set(feature_group, feature_set)

            feature_group_step = FeatureGroupStep(
                feature_group,
                feature_set,
                pre_calculated,
                cf,
                children_if_root,
                self.prepare_api_input_data(feature_group, feature_set),
            )

            # TODO
            # data type step
            fg_steps[f_hash] = feature_group_step
        return fg_steps

    def prepare_api_input_data(
        self, feature_group: Type[AbstractFeatureGroup], feature_set: FeatureSet
    ) -> Union[bool, BaseApiData]:
        if not isinstance(feature_group.input_data(), ApiInputData):
            return False

        if self.api_input_data_collection is None:
            raise ValueError(
                f"Feature group {feature_group} has an api input data class, but no api_input_data_collection was given."
            )

        if feature_set.get_name_of_one_feature().name is None:
            raise ValueError(f"Feature group {feature_group} has no feature set name.")

        api_input_name, matching_cls = self.api_input_data_collection.get_name_cls_by_matching_column_name(
            feature_set.get_name_of_one_feature().name
        )

        if matching_cls is None:
            raise ValueError(f"Feature group {feature_group} has no matching api data class for feature.")

        matching_cls_initialized = matching_cls(
            api_input_name, feature_set.get_name_of_one_feature().name, feature_set.get_options()
        )

        return matching_cls_initialized

    def add_artifact_to_feature_set(self, feature_group: Type[AbstractFeatureGroup], feature_set: FeatureSet) -> None:
        if feature_group.artifact() is None:
            return

        feature_set.add_artifact_name()

    def add_single_filters_to_feature_set(
        self, feature_group: Type[AbstractFeatureGroup], feature_set: FeatureSet
    ) -> None:
        if self.global_filter is None:
            return

        if len(self.global_filter.collection.keys()) == 0:
            return

        relevant_filters: Set[SingleFilter] = set()

        for (
            filtered_feature_group,
            filtered_feature_name,
        ), single_filters in self.global_filter.collection.items():
            # check for correct feature group
            if filtered_feature_group == feature_group:
                # check if filter feature is a feature of this feature set
                for feature in feature_set.features:
                    if feature.name == filtered_feature_name:
                        if len(relevant_filters) == 0:
                            relevant_filters = single_filters
                        else:
                            if relevant_filters != single_filters:
                                raise ValueError(
                                    f"""Feature group {feature_group} has different filters for different features {filtered_feature_name}.
                                      This is currently not allowed. Please make sure that all features of the same feature group have the same filters.
                                      If this has a business use case, where this does not make sense, please contact the developers.
                                      """
                                )

        feature_set.add_filters(relevant_filters)

    def get_parent_children_mapping(self, parent_to_children_mapping: Dict[UUID, Set[UUID]]) -> Dict[UUID, Set[UUID]]:
        inverted_dict: Dict[UUID, Set[UUID]] = {}
        for key, values in parent_to_children_mapping.items():
            for value in values:
                if value not in inverted_dict:
                    inverted_dict[value] = set()
                inverted_dict[value].add(key)

        return inverted_dict

    def invert_link_trekker(self, link_trekker: LinkTrekker) -> Dict[UUID, Set[LinkFrameworkTrekker]]:
        new_dict: Dict[UUID, Set[LinkFrameworkTrekker]] = defaultdict(set)

        for link, uuids in link_trekker.data.items():
            for uuid in uuids:
                new_dict[uuid].add(link)

        return new_dict

    def retrieve_links_which_must_be_calculated_before(
        self, features: Set[Feature], child_links: Dict[UUID, Set[LinkFrameworkTrekker]]
    ) -> Set[UUID]:
        new_set: Set[UUID] = set()

        for feature in features:
            if feature.uuid in child_links:
                new_set.update({link[0].uuid for link in child_links[feature.uuid]})
        return new_set

    def retrieve_nodes_which_must_be_calculated_before(
        self, features: Set[Feature], parent_to_children_mapping: Dict[UUID, Set[UUID]]
    ) -> Set[UUID]:
        new_set: Set[UUID] = set()
        for feature in features:
            if feature.uuid in parent_to_children_mapping:
                new_set.update(parent_to_children_mapping[feature.uuid])
        return new_set

    def group_features_by_compute_framework_and_options(self, features: Set[Feature]) -> Dict[int, Set[Feature]]:
        hash_collector: Dict[int, Set[Feature]] = defaultdict(set)

        # get features with same compute framework and options
        for feature in features:
            f_hash = feature.has_similarity_properties()
            hash_collector[f_hash].add(feature)

        return hash_collector
