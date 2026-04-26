"""Tests for the PairGenerator class."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.generator import PairGenerator


def test_pair_generator_run_returns_correct_count(tmp_path: Path) -> None:
    """run() should return the number of pairs generated."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    fake_image = np.full((4, 4, 3), 100, dtype=np.uint8)
    Image.fromarray(fake_image).save(raw_dir / "test1.jpg")
    Image.fromarray(fake_image).save(raw_dir / "test2.jpg")

    generator = PairGenerator(
        raw_dir=str(raw_dir),
        graded_dir=str(tmp_path / "graded"),
        ungraded_dir=str(tmp_path / "ungraded"),
        red=1.3,
        green=1.05,
        blue=0.8,
    )

    count = generator.run()

    assert count == 2


def test_pair_generator_creates_output_files(tmp_path: Path) -> None:
    """run() should create graded and ungraded files for each input."""
    raw_dir = tmp_path / "raw"
    graded_dir = tmp_path / "graded"
    ungraded_dir = tmp_path / "ungraded"
    raw_dir.mkdir()

    fake_image = np.full((4, 4, 3), 100, dtype=np.uint8)
    Image.fromarray(fake_image).save(raw_dir / "test.jpg")

    generator = PairGenerator(
        raw_dir=str(raw_dir),
        graded_dir=str(graded_dir),
        ungraded_dir=str(ungraded_dir),
        red=1.3,
        green=1.05,
        blue=0.8,
    )

    generator.run()

    assert (graded_dir / "test.jpg").exists()
    assert (ungraded_dir / "test.jpg").exists()