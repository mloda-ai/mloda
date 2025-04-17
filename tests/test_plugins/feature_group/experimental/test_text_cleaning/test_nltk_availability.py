"""
Test to check if NLTK is available in the test environment.
"""


def test_nltk_availability() -> None:
    """Test if NLTK is available in the test environment."""
    try:
        import nltk
        from nltk.corpus import stopwords

        print(f"NLTK version: {nltk.__version__}")
        assert nltk.__version__, "NLTK version not found"

    except ImportError:
        assert False, "NLTK is not installed"
