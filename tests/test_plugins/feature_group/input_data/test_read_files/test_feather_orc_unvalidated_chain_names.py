"""FeatherReader and OrcReader don't override get_column_names, so they decline
chain/column-separated names. Matching never calls load_data, so no pyarrow install is needed.
"""

from collections.abc import Iterator

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.provider import CHAIN_SEPARATOR
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one direct matcher call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


class TestShippedUnvalidatedReadersDeclineChainSeparatedNames:
    def test_feather_reader_declines_chain_separated_name(self, rejection_window: dict[str, MatchRejection]) -> None:
        result = FeatherReader.match_read_file_data_access(["dummy.feather"], [f"a{CHAIN_SEPARATOR}b"])

        assert result is None
        stored = rejection_window[FeatherReader.get_class_name()]
        assert "get_column_names" in stored.reason
        assert f"a{CHAIN_SEPARATOR}b" in stored.reason

    def test_orc_reader_declines_chain_separated_name(self) -> None:
        assert OrcReader.match_read_file_data_access(["dummy.orc"], [f"a{CHAIN_SEPARATOR}b"]) is None

    def test_feather_reader_still_assumes_plain_names(self) -> None:
        assert FeatherReader.match_read_file_data_access(["dummy.feather"], ["a"]) == "dummy.feather"

    def test_orc_reader_still_assumes_plain_names(self) -> None:
        assert OrcReader.match_read_file_data_access(["dummy.orc"], ["a"]) == "dummy.orc"
