"""Pandas text cleaning."""

from __future__ import annotations

import string

import unicodedata

try:
    import nltk
    from nltk.corpus import stopwords

    nltk_available = True
except ImportError:
    nltk = None
    stopwords = None
    nltk_available = False


try:
    import pandas as pd
except ImportError:
    pd = None


from typing import Any

from mloda.provider import FeatureSet, Implementation
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup


class PandasTextCleaning(Implementation):
    """Pandas implementation of the text cleaning operations."""

    feature_group = TextCleaningFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(cls, group: type[TextCleaningFeatureGroup], data: Any, features: FeatureSet) -> Any:
        return group.clean(
            data,
            features,
            source_text=cls._get_source_text,
            apply=cls._apply_operation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
        )

    @classmethod
    def _get_source_text(cls, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """
        Get the source text from the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to get

        Returns:
            The source text as a pandas Series
        """
        # Treat missing values as empty text, then convert to string if not already
        return data[feature_name].fillna("").astype(str)

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: pd.Series) -> pd.DataFrame:
        """
        Add the cleaning result to the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to add
            result: The cleaning result as a pandas Series

        Returns:
            The updated DataFrame with the cleaning result added
        """
        data[feature_name] = result
        return data

    @classmethod
    def _apply_operation(cls, data: pd.DataFrame, text: pd.Series, operation: str) -> pd.Series:
        """
        Apply a cleaning operation to the text.

        Args:
            data: The pandas DataFrame (for context)
            text: The text to clean as a pandas Series
            operation: The operation to apply

        Returns:
            The cleaned text as a pandas Series

        Raises:
            ValueError: If the operation is not supported
        """
        if operation == "normalize":
            return cls._normalize_text(text)
        elif operation == "remove_stopwords":
            return cls._remove_stopwords(text)
        elif operation == "remove_punctuation":
            return cls._remove_punctuation(text)
        elif operation == "remove_special_chars":
            return cls._remove_special_chars(text)
        elif operation == "normalize_whitespace":
            return cls._normalize_whitespace(text)
        elif operation == "remove_urls":
            return cls._remove_urls(text)
        else:
            raise ValueError(f"Unsupported cleaning operation: {operation}")

    @classmethod
    def _normalize_text(cls, text: pd.Series) -> pd.Series:
        """
        Normalize text by converting to lowercase and removing accents.

        Args:
            text: The text to normalize

        Returns:
            The normalized text
        """
        # Convert to lowercase - this should work regardless of NLTK availability
        result = text.str.lower()

        # Remove accents if unicodedata is available
        if unicodedata:

            def remove_accents(input_str: str) -> str:
                nfkd_form = unicodedata.normalize("NFKD", input_str)
                return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

            result = result.apply(remove_accents)

        return result

    @classmethod
    def _remove_stopwords(cls, text: pd.Series) -> pd.Series:
        """
        Remove common stopwords from text.

        Args:
            text: The text to process

        Returns:
            The text with stopwords removed
        """
        if not nltk_available:
            # If NLTK is not available, return the original text
            return text

        try:
            # Download stopwords if not already downloaded
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))

            def remove_stopwords_from_text(input_str: str) -> str:
                words = input_str.split()
                filtered_words = [word for word in words if word.lower() not in stop_words]
                return " ".join(filtered_words)

            return text.apply(remove_stopwords_from_text)
        except Exception:
            # If there's an error, return the original text
            return text

    @classmethod
    def _remove_punctuation(cls, text: pd.Series) -> pd.Series:
        """
        Remove punctuation from text.

        Args:
            text: The text to process

        Returns:
            The text with punctuation removed
        """
        translator = str.maketrans("", "", string.punctuation)
        return text.apply(lambda x: x.translate(translator))

    @classmethod
    def _remove_special_chars(cls, text: pd.Series) -> pd.Series:
        """
        Remove special characters from text.

        Args:
            text: The text to process

        Returns:
            The text with special characters removed
        """
        # Keep alphanumeric characters and whitespace
        pattern = r"[^a-zA-Z0-9\s]"
        return text.str.replace(pattern, "", regex=True)

    @classmethod
    def _normalize_whitespace(cls, text: pd.Series) -> pd.Series:
        """
        Normalize whitespace in text.

        Args:
            text: The text to process

        Returns:
            The text with normalized whitespace
        """
        # Replace multiple whitespace characters with a single space
        return text.str.replace(r"\s+", " ", regex=True).str.strip()

    @classmethod
    def _remove_urls(cls, text: pd.Series) -> pd.Series:
        """
        Remove URLs and email addresses from text.

        Args:
            text: The text to process

        Returns:
            The text with URLs and email addresses removed
        """
        # URL pattern
        url_pattern = r"https?://\S+|www\.\S+"
        # Email pattern
        email_pattern = r"\S+@\S+\.\S+"

        # Remove URLs and emails
        result = text.str.replace(url_pattern, "", regex=True)
        result = result.str.replace(email_pattern, "", regex=True)

        return result
