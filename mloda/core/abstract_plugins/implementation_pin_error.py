"""ImplementationPinError: a Feature(implementation=...) pin that cannot be honoured."""


class ImplementationPinError(ValueError):
    """Raised at plan time when an implementation pin is unknown, unavailable, ambiguous or conflicting."""
