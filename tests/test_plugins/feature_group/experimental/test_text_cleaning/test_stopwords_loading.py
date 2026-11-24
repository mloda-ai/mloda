"""Stopword loading keeps offline text cleaning in both backends and stays off plugin import."""

import subprocess  # nosec B404
import sys
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


_IMPORT_SCRIPT = """
import sys

import mloda_plugins.feature_group.experimental.text_cleaning.python_dict
import mloda_plugins.feature_group.experimental.text_cleaning.pandas

leaked = sorted(name for name in sys.modules if name == "nltk" or name.startswith("nltk."))
assert not leaked, "importing the text cleaning backends pulled nltk into sys.modules: " + repr(leaked)

print("ok")
"""


@pytest.mark.timeout(30)
def test_importing_the_backends_does_not_import_nltk() -> None:
    # Fixed argv, no shell, no user input.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=25,
    )

    assert result.returncode == 0, f"nltk must be imported lazily by the backends.\nstderr:\n{result.stderr}"
    assert result.stdout.strip() == "ok"


def test_remove_stopwords_without_nltk_returns_text_unchanged(
    remove_stopwords: Callable[[list[str]], list[str]], nltk_io: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "nltk", None)
    text = ["The River is wide", "and the", ""]

    result = remove_stopwords(text)

    assert result == text
    nltk_io.words.assert_not_called()
