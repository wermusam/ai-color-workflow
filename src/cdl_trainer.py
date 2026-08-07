"""Trains CdlNet to predict CDL grades from ungraded/graded image pairs."""

from pathlib import Path


class CdlTrainer:
    """Trains a CdlNet on ungraded/graded image pairs."""

    def __init__(
        self,
        ungraded_dir: str,
        graded_dir: str,
        output_path: str,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        image_size: int = 128,
    ) -> None:
        self.ungraded_dir = Path(ungraded_dir)
        self.graded_dir = Path(graded_dir)
        self.output_path = Path(output_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_size = image_size
