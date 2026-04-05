REPORT_URL = "https://github.com/mloda-ai/mloda/issues"


def internal_invariant_error(
    invariant: str,
    actual_values: str = "",
    hint: str = "",
) -> str:
    """Build a consistent message for internal invariant violations.

    Args:
        invariant: What invariant was expected.
        actual_values: The actual values that violated the invariant.
        hint: Optional extra guidance for the developer.
    """
    parts = [f"Internal error: {invariant}"]
    if actual_values:
        parts.append(f"Actual state: {actual_values}")
    if hint:
        parts.append(hint)
    parts.append(f"Please report this issue at {REPORT_URL} with the full traceback.")
    return "\n".join(parts)
