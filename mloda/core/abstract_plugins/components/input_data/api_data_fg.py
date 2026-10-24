"""Source group serving the api_data sets of a request; subclass and override load_neutral for other transport."""

from collections.abc import Collection, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.api_data_scope import ApiDataRef, read_api_data
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.options import Options


class ApiDataFG(SourceFeatureGroup):
    """Serves the sets given as api_data={key: {column: values}}; Options({"ApiDataFG": key}) narrows to one set."""

    CLAIM_ROUTES = (ClaimRoute("api_data", NamePolicy.CHECKED, True),)

    @classmethod
    def _spec(cls, data_access_collection: DataAccessCollection | None) -> Mapping[str, tuple[str, ...]]:
        return {} if data_access_collection is None else data_access_collection.api_data_columns

    @classmethod
    def is_pointed(
        cls, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> bool:
        """Also pointed when the name is a column of a set, so another group computing it stays ambiguous."""
        if super().is_pointed(feature_name, options, data_access_collection):
            return True
        if cls.foreign_pointer(options) is not None or cls.has_query_text(options):
            return False
        base_name = cls.get_column_base_feature(feature_name)
        return any(base_name in columns for columns in cls._spec(data_access_collection).values())

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        spec = cls._spec(data_access_collection)
        pointers = [(name, value) for name, value in options.items() if name in cls.pointer_keys()]
        if pointers:
            pointer_key, key = pointers[0]
            if not isinstance(key, str) or key not in spec:
                cls._abort(
                    ValueError(f"{pointer_key}={key!r} is not an api_data set key; known keys: {sorted(spec)}"),
                    deferrable=True,
                )
                return []
            spec = {key: spec[key]}
        return [SourceMatch(f"api_data:{key}", ApiDataRef(key, columns)) for key, columns in spec.items()]

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        return match.access.columns if isinstance(match.access, ApiDataRef) else None

    @classmethod
    def api_values(cls, match: SourceMatch) -> Any:
        """The run's raw values for the matched api_data set; for load_neutral overrides."""
        if not isinstance(match.access, ApiDataRef):
            raise ValueError(f"{cls.get_class_name()}.api_values needs an api_data match, got {match.source!r}.")
        return read_api_data(match.access.key)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        return cls.api_values(match)
