from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.index.index import Index


def create_index_feature(index: Index, feature_group: FeatureGroup, feature: Feature) -> Feature:
    if feature.domain:
        domain = feature.domain.name
    else:
        domain = None

    new_index_feature = Feature(
        name=index.index[0],
        options=feature.options,
        domain=domain,
    )

    new_index_feature.input_data_match = feature.input_data_match
    if feature.compute_frameworks is not None:
        new_index_feature.compute_frameworks = set(feature.compute_frameworks)
        new_index_feature.framework_pinned = feature.framework_pinned
    new_index_feature.name = feature_group.set_feature_name(feature.options, new_index_feature.name)
    return new_index_feature
