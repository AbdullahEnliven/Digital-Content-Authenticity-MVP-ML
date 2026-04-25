from pathlib import Path


def iter_text_files(dataset_dir: Path) -> list[tuple[Path, str]]:
    """Return all .txt files under dataset/<label>/... as (path, label)."""
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
        for path in sorted(label_dir.rglob("*.txt")):
            if path.is_file():
                samples.append((path, label))
    return samples


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file with a permissive fallback for imperfect datasets."""
    return path.read_text(encoding="utf-8", errors="replace")
