"""Tests for the PairDataset class."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset import PairDataset, PairImageDataset
from src.transforms import GainTransform


def test_generate_pairs_creates_graded_and_ungraded_files(tmp_path: Path) -> None:
    """generate_pairs should produce one graded and one ungraded file per input."""
    raw_dir = tmp_path / "raw"
    graded_dir = tmp_path / "graded"
    ungraded_dir = tmp_path / "ungraded"
    raw_dir.mkdir()

    fake_image = np.full((4, 4, 3), 100, dtype=np.uint8)
    Image.fromarray(fake_image).save(raw_dir / "test_image.jpg")

    transform = GainTransform(red=1.5, green=1.0, blue=0.5)
    dataset = PairDataset(
        raw_dir=raw_dir,
        graded_dir=graded_dir,
        ungraded_dir=ungraded_dir,
        transform=transform,
    )

    count = dataset.generate_pairs()

    assert count == 1
    assert (graded_dir / "test_image.jpg").exists()
    assert (ungraded_dir / "test_image.jpg").exists()


def test_pair_image_dataset_loads_pairs() -> None:
    dataset = PairImageDataset(
        ungraded_dir=Path("data/ungraded"),
        graded_dir=Path("data/graded"),
    )
    assert len(dataset) == 12

    ungraded, graded = dataset[0]
    assert ungraded.shape[0] == 3  # RGB channels
    assert graded.shape[0] == 3
    assert ungraded.shape == graded.shape