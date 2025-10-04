"""
Test to check if NLTK is available in the test environment.
"""

import os
import pytest


@pytest.mark.skipif(
    os.getenv("SKIP_TEXT_CLEANING_INSTALLATION_TEST", "false").lower() == "true",
    reason="Text cleaning installation test is disabled by environment variable",
)
def test_nltk_availability() -> None:
    """Test if NLTK is available in the test environment."""
    try:
        import nltk
        from nltk.corpus import stopwords

        print(f"NLTK version: {nltk.__version__}")
        assert nltk.__version__, "NLTK version not found"

    except ImportError:
        pytest.fail("NLTK is not installed but is required for this test environment")
