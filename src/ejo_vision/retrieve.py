"""Tiny grayscale thumbnail used to find a similar past frame.

The vector is mean-centered and scaled to length 1, so a dot product is a
similarity between -1 and 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

THUMB_W = 48
THUMB_H = 24
SIM_MIN = 0.90
MIN_AGE_NS = 8_000_000_000
BOOK_EVERY = 5
# Last two complete laps. Not used to choose SIM_MIN.
DEV_END_S = 896.135417986
VAL_END_S = 1120.864154466


def thumbnail_descriptor(gray: np.ndarray) -> np.ndarray | None:
    import cv2

    small = cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    norm = float(np.linalg.norm(small))
    if norm < 1e-6:
        return None
    return (small / norm).ravel()


def best_past_match(
    book_desc: np.ndarray,
    book_t: np.ndarray,
    query_desc: np.ndarray,
    query_t: int,
    min_age_ns: int,
) -> tuple[int | None, float]:
    """Nearest earlier page. Pages newer than query_t - min_age_ns are ignored."""
    index, score, _second_index, _second_score = top_past_matches(
        book_desc, book_t, query_desc, query_t, min_age_ns
    )
    return index, score


def top_past_matches(
    book_desc: np.ndarray,
    book_t: np.ndarray,
    query_desc: np.ndarray,
    query_t: int,
    min_age_ns: int,
) -> tuple[int | None, float, int | None, float]:
    """Best and second-best earlier pages."""
    if len(book_desc) == 0:
        return None, -np.inf, None, -np.inf
    sim = book_desc @ query_desc
    sim = np.where(book_t.astype(np.int64) <= int(query_t) - int(min_age_ns), sim, -np.inf)
    if len(sim) == 1:
        score = float(sim[0])
        return (0, score, None, -np.inf) if np.isfinite(score) else (None, -np.inf, None, -np.inf)
    first, second = np.argpartition(-sim, 1)[:2]
    if sim[second] > sim[first]:
        first, second = second, first
    score1, score2 = float(sim[first]), float(sim[second])
    if not np.isfinite(score1):
        return None, -np.inf, None, -np.inf
    if not np.isfinite(score2):
        return int(first), score1, None, -np.inf
    return int(first), score1, int(second), score2


def save_picture_book(
    path: str | Path,
    descriptors: np.ndarray,
    timestamp_ns: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
) -> None:
    """Store development-lap thumbnails and the training poses that label them."""
    path = Path(path)
    desc = np.asarray(descriptors, dtype=np.float32)
    t_ns = np.asarray(timestamp_ns, dtype=np.int64)
    px = np.asarray(x, dtype=np.float64)
    py = np.asarray(y, dtype=np.float64)
    pyaw = np.asarray(yaw, dtype=np.float64)
    n = len(t_ns)
    if n == 0 or not (n == len(desc) == len(px) == len(py) == len(pyaw)):
        raise ValueError("Picture book arrays must be non-empty and the same length.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        descriptors=desc,
        timestamp_ns=t_ns,
        x=px,
        y=py,
        yaw=pyaw,
        dev_end_s=np.float64(DEV_END_S),
    )


def load_picture_book(path: str | Path) -> dict[str, np.ndarray]:
    """Load a book written by save_picture_book. The file has no live ground truth."""
    with np.load(path) as book:
        return {
            "descriptors": np.array(book["descriptors"], dtype=np.float32),
            "timestamp_ns": np.array(book["timestamp_ns"], dtype=np.int64),
            "x": np.array(book["x"], dtype=np.float64),
            "y": np.array(book["y"], dtype=np.float64),
            "yaw": np.array(book["yaw"], dtype=np.float64),
        }
