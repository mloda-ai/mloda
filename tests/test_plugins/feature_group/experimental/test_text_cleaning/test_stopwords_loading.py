"""Stopword loading preserves offline text cleaning in both backends."""

from collections.abc import Callable
from unittest.mock import Mock, call

import pandas as pd
import pytest

from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaning
from mloda_plugins.feature_group.experimental.text_cleaning.python_dict import PythonDictTextCleaning


@pytest.fixture(params=["pandas", "python_dict"])
def remove_stopwords(request: pytest.FixtureRequest) -> Callable[[list[str]], list[str]]:
    if request.param == "python_dict":
        return PythonDictTextCleaning._remove_stopwords

    def remove_from_series(text: list[str]) -> list[str]:
        return list(PandasTextCleaning._remove_stopwords(pd.Series(text)))

    return remove_from_series


@pytest.fixture
def nltk_io(monkeypatch: pytest.MonkeyPatch) -> Mock:
    nltk = pytest.importorskip("nltk")
    io = Mock()
    io.words.return_value = ["the", "is", "and"]
    io.download.return_value = True
    # Bypass LazyCorpusLoader.__getattr__ while covering every alias of this corpus.
    monkeypatch.setitem(vars(nltk.corpus.stopwords), "words", io.words)
    monkeypatch.setattr(nltk, "download", io.download)
    return io


def test_remove_stopwords_uses_installed_corpus_without_download(
    remove_stopwords: Callable[[list[str]], list[str]], nltk_io: Mock
) -> None:
    result = remove_stopwords(["The River is wide", "and the", ""])

    assert result == ["River wide", "", ""]
    nltk_io.words.assert_called_once_with("english")
    nltk_io.download.assert_not_called()


def test_remove_stopwords_downloads_only_after_missing_corpus(
    remove_stopwords: Callable[[list[str]], list[str]], nltk_io: Mock
) -> None:
    nltk_io.words.side_effect = [LookupError("missing stopwords"), ["the", "is", "and"]]

    result = remove_stopwords(["The River is wide", "and the", ""])

    assert result == ["River wide", "", ""]
    assert nltk_io.mock_calls == [
        call.words("english"),
        call.download("stopwords", quiet=True),
        call.words("english"),
    ]


def test_remove_stopwords_preserves_text_on_corpus_error(
    remove_stopwords: Callable[[list[str]], list[str]], nltk_io: Mock
) -> None:
    nltk_io.words.side_effect = OSError("unreadable corpus")
    text = ["The  River is wide", "and the", ""]

    assert remove_stopwords(text) == text
    nltk_io.download.assert_not_called()


@pytest.mark.parametrize("download_error", [OSError("offline"), None], ids=["download-error", "still-missing"])
def test_remove_stopwords_preserves_text_when_download_cannot_supply_corpus(
    remove_stopwords: Callable[[list[str]], list[str]], nltk_io: Mock, download_error: OSError | None
) -> None:
    nltk_io.words.side_effect = LookupError("missing stopwords")
    nltk_io.download.side_effect = download_error
    text = ["The  River is wide", "and the", ""]

    assert remove_stopwords(text) == text
    nltk_io.download.assert_called_once_with("stopwords", quiet=True)
