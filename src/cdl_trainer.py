"""Trains CdlNet to predict CDL grades from ungraded/graded image pairs."""

from pathlib import Path

import torch

from src.cdl_torch import TorchCdl


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

    def grade_batch(self, images: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply each image's predicted 10 numbers to that image.

        images is (batch, 3, height, width); params is (batch, 10). Returns the
        graded batch, the same shape as images.
        """
        graded = []
        for image, numbers in zip(images, params, strict=True):
            slope = numbers[0:3]
            offset = numbers[3:6]
            power = numbers[6:9]
            saturation = numbers[9]
            image_hwc = image.permute(1, 2, 0)
            graded_hwc = TorchCdl(slope, offset, power, saturation).apply(image_hwc)
            graded.append(graded_hwc.permute(2, 0, 1))
        return torch.stack(graded)
