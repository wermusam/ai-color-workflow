"""Tests for CdlTrainer."""

import math
from pathlib import Path

import torch

from src.cdl_net import CdlNet
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


def test_run_trains_and_saves_a_loadable_model(tmp_path: Path) -> None:
    """run() should train, return a finite loss, and save loadable weights."""
    output_path = tmp_path / "cdlnet.pt"
    trainer = CdlTrainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path=str(output_path),
        epochs=2,
        image_size=16,
    )

    loss = trainer.run()

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert output_path.exists()
    CdlNet().load_state_dict(torch.load(output_path, weights_only=True))


def test_grade_batch_identity_leaves_image_unchanged() -> None:
    """An identity grade should return the image unchanged."""
    trainer = CdlTrainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/cdlnet.pt",
    )
    images = torch.rand(2, 3, 8, 8)
    identity = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    params = identity.repeat(2, 1)

    graded = trainer.grade_batch(images, params)

    torch.testing.assert_close(graded, images)


def test_grade_batch_applies_each_images_own_params() -> None:
    """Each row of params should affect only its own image."""
    trainer = CdlTrainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/cdlnet.pt",
    )
    images = torch.rand(2, 3, 8, 8)
    desaturate = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    identity = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    params = torch.tensor([desaturate, identity])

    graded = trainer.grade_batch(images, params)

    channel_spread = graded[0].max(dim=0).values - graded[0].min(dim=0).values
    assert torch.allclose(channel_spread, torch.zeros_like(channel_spread), atol=1e-6)
    torch.testing.assert_close(graded[1], images[1])
