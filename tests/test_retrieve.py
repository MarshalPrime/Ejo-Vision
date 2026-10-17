import numpy as np
import pytest

from ejo_vision.retrieve import best_past_match, load_picture_book, save_picture_book, top_past_matches


def test_match_uses_only_the_past():
    book = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)
    times = np.array([0, 10, 30], dtype=np.int64)
    # Query is a copy of page 0, asked at t=20. Page 2 is the same picture but in the future.
    index, score = best_past_match(book, times, np.array([1.0, 0.0]), query_t=20, min_age_ns=0)
    assert index == 0
    assert score == 1.0


def test_second_best_is_reported():
    book = np.array([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]], dtype=float)
    times = np.array([0, 5, 8], dtype=np.int64)
    first, score1, second, score2 = top_past_matches(
        book, times, np.array([1.0, 0.0]), query_t=10, min_age_ns=0
    )
    assert first == 0
    assert score1 == 1.0
    assert second == 1
    assert score2 < score1


def test_picture_book_roundtrip(tmp_path):
    desc = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    times = np.array([10, 20], dtype=np.int64)
    x = np.array([1.5, 2.5])
    y = np.array([-1.0, 3.0])
    yaw = np.array([0.1, -0.2])
    path = tmp_path / "book.npz"
    save_picture_book(path, desc, times, x, y, yaw)
    loaded = load_picture_book(path)
    np.testing.assert_array_equal(loaded["descriptors"], desc)
    np.testing.assert_array_equal(loaded["timestamp_ns"], times)
    np.testing.assert_allclose(loaded["x"], x)
    np.testing.assert_allclose(loaded["y"], y)
    np.testing.assert_allclose(loaded["yaw"], yaw)


def test_picture_book_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        save_picture_book(tmp_path / "book.npz", np.zeros((0, 2)), [], [], [], [])


def test_recent_pages_are_excluded():
    book = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    times = np.array([0, 9], dtype=np.int64)
    index, score = best_past_match(book, times, np.array([1.0, 0.0]), query_t=10, min_age_ns=5)
    assert index == 0
    assert score == 1.0
