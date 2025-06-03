"""
PythonDict implementation for text cleaning feature groups.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any, Dict, List, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup

# Optional NLTK support - gracefully handle if not available
try:
    import nltk
    from nltk.corpus import stopwords

    nltk_available = True
except ImportError:
    nltk = None
    stopwords = None
    nltk_available = False


class PythonDictTextCleaningFeatureGroup(TextCleaningFeatureGroup):
    """
    PythonDict implementation for text cleaning feature groups.

    This implementation uses pure Python operations on List[Dict[str, Any]] data structures
    to perform text cleaning operations without external dependencies (except optional NLTK).
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PythonDictFramework}

    @classmethod
    def _check_source_feature_exists(cls, data: List[Dict[str, Any]], feature_name: str) -> None:
        if not data:
            raise ValueError("Data cannot be empty")

        # Check if feature exists in any row
        feature_exists = any(feature_name in row for row in data)
        if not feature_exists:
            raise ValueError(f"Feature '{feature_name}' not found in the data")

    @classmethod
    def _get_source_text(cls, data: List[Dict[str, Any]], feature_name: str) -> List[str]:
        """
        Get the source text from the data.

        Args:
            data: The List[Dict] data structure
            feature_name: The name of the feature to get

        Returns:
            The source text as a list of strings
        """
        # Convert to string if not already and handle None values
        return [str(row.get(feature_name, "")) if row.get(feature_name) is not None else "" for row in data]

    @classmethod
    def _add_result_to_data(
        cls, data: List[Dict[str, Any]], feature_name: str, result: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Add the cleaning result to the data.

        Args:
            data: The List[Dict] data structure
            feature_name: The name of the feature to add
            result: The cleaning result as a list of strings

        Returns:
            The updated data with the cleaning result added
        """
        if len(result) != len(data):
            raise ValueError(f"Result length {len(result)} does not match data length {len(data)}")

        for i, row in enumerate(data):
            row[feature_name] = result[i]

        return data

    @classmethod
    def _apply_operation(cls, data: List[Dict[str, Any]], text: List[str], operation: str) -> List[str]:
        """
        Apply a cleaning operation to the text.

        Args:
            data: The List[Dict] data structure (for context)
            text: The text to clean as a list of strings
            operation: The operation to apply

        Returns:
            The cleaned text as a list of strings

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
    def _normalize_text(cls, text: List[str]) -> List[str]:
        """
        Normalize text by converting to lowercase and removing accents.

        Args:
            text: The text to normalize

        Returns:
            The normalized text
        """
        result = []
        for input_str in text:
            # Convert to lowercase
            normalized = input_str.lower()

            # Remove accents using unicodedata
            try:
                nfkd_form = unicodedata.normalize("NFKD", normalized)
                normalized = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
            except Exception:
                # If unicodedata fails, just use lowercase
                pass  # nosec B110

            result.append(normalized)

        return result

    @classmethod
    def _remove_stopwords(cls, text: List[str]) -> List[str]:
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

            result = []
            for input_str in text:
                words = input_str.split()
                filtered_words = [word for word in words if word.lower() not in stop_words]
                result.append(" ".join(filtered_words))

            return result
        except Exception:
            # If there's an error, return the original text
            return text

    @classmethod
    def _remove_punctuation(cls, text: List[str]) -> List[str]:
        """
        Remove punctuation from text.

        Args:
            text: The text to process

        Returns:
            The text with punctuation removed
        """
        translator = str.maketrans("", "", string.punctuation)
        return [input_str.translate(translator) for input_str in text]

    @classmethod
    def _remove_special_chars(cls, text: List[str]) -> List[str]:
        """
        Remove special characters from text.

        Args:
            text: The text to process

        Returns:
            The text with special characters removed
        """
        # Keep alphanumeric characters and whitespace
        pattern = r"[^a-zA-Z0-9\s]"
        return [re.sub(pattern, "", input_str) for input_str in text]

    @classmethod
    def _normalize_whitespace(cls, text: List[str]) -> List[str]:
        """
        Normalize whitespace in text.

        Args:
            text: The text to process

        Returns:
            The text with normalized whitespace
        """
        # Replace multiple whitespace characters with a single space
        result = []
        for input_str in text:
            normalized = re.sub(r"\s+", " ", input_str).strip()
            result.append(normalized)

        return result

    @classmethod
    def _remove_urls(cls, text: List[str]) -> List[str]:
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

        result = []
        for input_str in text:
            # Remove URLs and emails
            cleaned = re.sub(url_pattern, "", input_str)
            cleaned = re.sub(email_pattern, "", cleaned)
            result.append(cleaned)

        return result
