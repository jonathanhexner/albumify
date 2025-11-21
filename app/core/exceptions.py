"""Custom exceptions for the application."""


class AppError(Exception):
    """Base exception for application errors."""
    pass


class AgentError(AppError):
    """Raised when agent operations fail."""
    pass


class ImageLoadError(AppError):
    """Raised when image loading fails."""
    pass


class ValidationError(AppError):
    """Raised when validation fails."""
    pass
