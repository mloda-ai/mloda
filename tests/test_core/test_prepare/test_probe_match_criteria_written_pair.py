"""probe_match_criteria pops the reserved reader key once and hands the pair back on the outcome (``written``)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    RESERVED_READER_OPTION_KEY,
)
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import (
    ToySourceBase,
    toy_dac,
)

WRITER_FEATURE = "probe_written_writer_feat_1001"
DECLINER_FEATURE = "probe_written_decliner_feat_1001"
RAISER_FEATURE = "probe_written_raiser_feat_1001"
ABORTER_FEATURE = "probe_written_aborter_feat_1001"
ACCESS = "probe_written_access_1001"
FORMAT_FEATURE = "probe_written_format_feat_1001"


class ProbeWrittenFw1001(ComputeFramework):
    pass


class ProbeWrittenWriterFG1001(FeatureGroup):
    """Writes the pair the way a source group does, then matches."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {WRITER_FEATURE}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {ProbeWrittenFw1001}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) not in cls.feature_names_supported():
            return False
        options.add_to_group(RESERVED_READER_OPTION_KEY, (cls, ACCESS))
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ProbeWrittenDeclinerFG1001(ProbeWrittenWriterFG1001):
    """Writes the pair, then declines."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {DECLINER_FEATURE}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) not in cls.feature_names_supported():
            return False
        options.add_to_group(RESERVED_READER_OPTION_KEY, (cls, ACCESS))
        return False


class ProbeWrittenRaiserFG1001(ProbeWrittenWriterFG1001):
    """Writes the pair, then raises a contained error."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {RAISER_FEATURE}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) not in cls.feature_names_supported():
            return False
        options.add_to_group(RESERVED_READER_OPTION_KEY, (cls, ACCESS))
        raise RuntimeError("contained defect after the write")


class ProbeWrittenAborterFG1001(ProbeWrittenWriterFG1001):
    """Writes the pair, then raises a marked abort."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {ABORTER_FEATURE}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) not in cls.feature_names_supported():
            return False
        options.add_to_group(RESERVED_READER_OPTION_KEY, (cls, ACCESS))
        raise escalate_match_abort(RuntimeError("marked abort after the write"))


class ProbeWrittenFormatFG1001(ToySourceBase):
    """Serves only its own column, so process-wide discovery cannot collide."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("probe_written_1001", NamePolicy.CHECKED, True),)

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        return [column for column in match.access if column == FORMAT_FEATURE]


def test_matching_writer_key_moves_onto_the_outcome() -> None:
    options = Options()

    outcome = probe_match_criteria(ProbeWrittenWriterFG1001, WRITER_FEATURE, options)

    assert outcome.matched is True
    assert RESERVED_READER_OPTION_KEY not in options
    assert outcome.written == (ProbeWrittenWriterFG1001, ACCESS)


def test_non_match_leaves_no_key_and_no_written_pair() -> None:
    options = Options()

    outcome = probe_match_criteria(ProbeWrittenDeclinerFG1001, DECLINER_FEATURE, options)

    assert outcome.matched is False
    assert outcome.written is None
    assert RESERVED_READER_OPTION_KEY not in options


def test_contained_abort_leaves_no_key_and_no_written_pair() -> None:
    options = Options()

    outcome = probe_match_criteria(ProbeWrittenRaiserFG1001, RAISER_FEATURE, options)

    assert outcome.matched is False
    assert outcome.matcher_error is not None
    assert outcome.written is None
    assert RESERVED_READER_OPTION_KEY not in options


def test_marked_abort_propagates_and_leaves_no_key_on_the_options() -> None:
    options = Options()

    with pytest.raises(RuntimeError, match="marked abort after the write"):
        probe_match_criteria(ProbeWrittenAborterFG1001, ABORTER_FEATURE, options)

    assert RESERVED_READER_OPTION_KEY not in options


def test_source_group_write_moves_onto_the_outcome() -> None:
    options = Options()
    access = toy_dac(probe_written_handle_1001={FORMAT_FEATURE: [1]})

    outcome = probe_match_criteria(ProbeWrittenFormatFG1001, FORMAT_FEATURE, options, access)

    assert outcome.matched is True
    assert RESERVED_READER_OPTION_KEY not in options
    assert outcome.written == (ProbeWrittenFormatFG1001, SourceMatch("probe_written_handle_1001:toy", None))
