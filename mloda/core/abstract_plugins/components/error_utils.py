REPORT_URL = "https://github.com/mloda-ai/mloda/issues"


class MlodaRunError(Exception):
    """Raised by mlodaAPI when a worker step failed and the original exception
    object could not be preserved (e.g. the internal critical error_out path,
    or a non-picklable exception crossing a process boundary)."""


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
