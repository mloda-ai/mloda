from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.index.index import Index


def create_index_feature(index: Index, feature_group: AbstractFeatureGroup, feature: Feature) -> Feature:
    if feature.domain:
        domain = feature.domain.name
    else:
        domain = None

    cfw = feature.get_compute_framework().get_class_name()

    new_index_feature = Feature(
        name=index.index[0],
        options=feature.options,
        compute_framework=cfw,
        domain=domain,
    )

    new_index_feature.name = feature_group.set_feature_name(feature.options, new_index_feature.name)
    return new_index_feature
