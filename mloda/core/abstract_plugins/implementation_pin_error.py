"""ImplementationPinError: a Feature(implementation=...) pin that cannot be honoured."""


class ImplementationPinError(ValueError):
    """Raised at plan time when an implementation pin is unknown, unavailable, ambiguous or conflicting."""


class VariantBlockedPinError(ImplementationPinError):
    """An implementation pin blocked because its framework follows a rule variant the options do not allow."""

    def __init__(
        self,
        message: str,
        *,
        feature: str,
        feature_group: str,
        variants: tuple[str, ...],
        allow_variants: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.feature = feature
        self.feature_group = feature_group
        self.variants = variants
        self.allow_variants = allow_variants
