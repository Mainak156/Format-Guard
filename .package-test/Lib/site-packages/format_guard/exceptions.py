class FormatGuardError(Exception):
    """Base exception for Format Guard."""


class ValidationFailedError(FormatGuardError):
    """Raised when LLM output cannot be validated."""


class RepairFailedError(FormatGuardError):
    """Raised when invalid output cannot be repaired."""