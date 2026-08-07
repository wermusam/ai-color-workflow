"""Tests for CdlTrainer."""

import torch

from src.cdl_trainer import CdlTrainer


def test_cdl_trainer_can_be_constructed() -> None:
    """CdlTrainer should build and store its settings."""
    trainer = CdlTrainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/cdlnet.pt",
    )
    assert trainer.ungraded_dir.name == "ungraded"
    assert trainer.epochs == 100


def test_grade_batch_keeps_the_batch_shape() -> None:
    """grade_batch should apply per-image params and return the same shape."""
    trainer = CdlTrainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/cdlnet.pt",
    )
    images = torch.rand(2, 3, 8, 8)
    params = torch.rand(2, 10)

    graded = trainer.grade_batch(images, params)

    assert graded.shape == images.shape
