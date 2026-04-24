import pytest

from evret.utils import (
    ensure_parent_dir,
    find_duplicates,
    normalize_str_int_mapping,
    normalize_unique_non_empty_strings,
    require_file_exists,
    require_non_empty_str,
    require_positive_int,
)


def test_require_non_empty_str_strips_and_validates() -> None:
    assert require_non_empty_str("  alpha  ", "name") == "alpha"

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        require_non_empty_str("   ", "name")


def test_require_positive_int_validates_type_and_sign() -> None:
    assert require_positive_int("5", "k") == 5

    with pytest.raises(ValueError, match="k must be an integer"):
        require_positive_int("abc", "k")
    with pytest.raises(ValueError, match="k must be a positive integer"):
        require_positive_int(0, "k")


def test_require_file_exists_returns_path(tmp_path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("x", encoding="utf-8")

    assert require_file_exists(file_path, "dataset") == file_path
    with pytest.raises(ValueError, match="dataset file not found"):
        require_file_exists(tmp_path / "missing.txt", "dataset")


def test_ensure_parent_dir_creates_directories(tmp_path) -> None:
    file_path = tmp_path / "nested" / "x.json"
    returned = ensure_parent_dir(file_path)

    assert returned == file_path
    assert (tmp_path / "nested").is_dir()


def test_normalize_unique_non_empty_strings_filters_and_dedupes() -> None:
    values = [" doc_1 ", "", "doc_2", "doc_1", "   "]

    assert normalize_unique_non_empty_strings(values) == ["doc_1", "doc_2"]


def test_normalize_str_int_mapping_strips_keys_and_casts_values() -> None:
    mapping = {" doc_1 ": "2", "": 1, "doc_2": 3}

    assert normalize_str_int_mapping(mapping) == {"doc_1": 2, "doc_2": 3}


def test_find_duplicates_returns_duplicate_values() -> None:
    assert find_duplicates(["a", "b", "a", "c", "b"]) == {"a", "b"}
    assert find_duplicates([1, 2, 3]) == set()
