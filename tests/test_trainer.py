"""Tests for Trainer."""

from src.trainer import Trainer


def test_trainer_can_be_constructed() -> None:
    trainer = Trainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/colorgrade.pt",
    )
    assert trainer.ungraded_dir.name == "ungraded"
    assert trainer.graded_dir.name == "graded"
    assert trainer.output_path.name == "colorgrade.pt"
    assert trainer.epochs == 100
    assert trainer.batch_size == 4
    assert trainer.learning_rate == 1e-3