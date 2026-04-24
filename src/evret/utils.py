"""Internal utility helpers for validation and normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypeVar

from evret.errors import EvretValidationError

T = TypeVar("T")


def require_non_empty_str(value: object, field_name: str) -> str:
    """Return a stripped string value and validate it is non-empty."""
    normalized = str(value).strip()
    if not normalized:
        raise EvretValidationError(f"{field_name} must be a non-empty string")
    return normalized


def require_positive_int(value: object, field_name: str) -> int:
    """Return an integer and validate it is strictly positive."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvretValidationError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise EvretValidationError(f"{field_name} must be a positive integer")
    return parsed


def require_file_exists(path: str | Path, field_name: str = "path") -> Path:
    """Return a `Path` and validate it points to an existing file."""
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise EvretValidationError(f"{field_name} file not found: {resolved_path}")
    return resolved_path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directory for a file path and return normalized path."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def normalize_unique_non_empty_strings(values: Iterable[object]) -> list[str]:
    """Normalize iterable values to deduplicated, non-empty stripped strings."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def normalize_str_int_mapping(values: Mapping[object, object]) -> dict[str, int]:
    """Normalize mapping keys to non-empty strings and values to ints."""
    normalized: dict[str, int] = {}
    for key, value in values.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[key_text] = int(value)
    return normalized


def has_duplicates(values: Iterable[T]) -> bool:
    """Return whether iterable contains duplicate values."""
    seen: set[T] = set()
    for value in values:
        if value in seen:
            return True
        seen.add(value)
    return False


def find_duplicates(values: Iterable[T]) -> set[T]:
    """Return the subset of values that appear more than once."""
    seen: set[T] = set()
    duplicates: set[T] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
            continue
        seen.add(value)
    return duplicates
