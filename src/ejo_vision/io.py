"""Image-sequence loading. No dataset-specific folder layout is assumed."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_image_paths(image_dir: str | Path) -> list[Path]:
    """Return image paths sorted by filename.

    Lexicographic sort is safe for zero-padded names such as `000001.png`.
    It is not safe for `frame_9.png` vs `frame_10.png`.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
