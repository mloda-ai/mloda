"""Result type returned by conformance checks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check against one subject."""

    check: str
    subject: str
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures
