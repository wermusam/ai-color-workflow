"""Tests for CdlTrainer."""

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
