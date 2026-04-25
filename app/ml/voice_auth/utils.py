import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def temporary_audio_file(file_bytes: bytes, suffix: str) -> Iterator[Path]:
    """Write uploaded audio bytes to a temporary file and remove it afterwards."""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(file_bytes)
            temp.flush()
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def iter_audio_files(dataset_dir: Path, allowed_extensions: set[str]) -> list[tuple[Path, str]]:
    """Return all audio files under dataset/<label>/... as (path, label)."""
    if not dataset_dir.exists():
        raise ValueError(
            f"Dataset directory not found: {dataset_dir}. Expected folders like "
            "dataset/human/ and dataset/ai/."
        )
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset path is not a directory: {dataset_dir}")

    samples: list[tuple[Path, str]] = []
    for label_dir in sorted(dataset_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                samples.append((path, label))
    return samples
