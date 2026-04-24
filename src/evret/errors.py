"""Evret exception hierarchy."""

from __future__ import annotations


class EvretError(Exception):
    """Base exception for Evret."""


class EvretValidationError(ValueError, EvretError):
    """Raised when user input or data format is invalid."""


class OptionalDependencyError(ImportError, EvretError):
    """Raised when an optional dependency required by a feature is missing."""
