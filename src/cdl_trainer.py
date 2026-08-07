"""Trains CdlNet to predict CDL grades from ungraded/graded image pairs."""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.cdl_net import CdlNet
from src.cdl_torch import TorchCdl
from src.dataset import PairImageDataset


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

    def run(self) -> float:
        """Train the model on the pairs and save its weights. Returns final loss."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = PairImageDataset(
            ungraded_dir=self.ungraded_dir,
            graded_dir=self.graded_dir,
            image_size=self.image_size,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = CdlNet()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        print(f"Training CdlNet on {len(dataset)} pairs at {self.image_size}x{self.image_size}")
        print(f"Epochs: {self.epochs}, batch size: {self.batch_size}, lr: {self.learning_rate}")
        print("-" * 60)

        start_time = time.time()
        final_loss = 0.0

        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_loss = 0.0
            num_batches = 0

            for ungraded_batch, graded_batch in dataloader:
                params = model(ungraded_batch)
                prediction = self.grade_batch(ungraded_batch, params)
                loss = loss_fn(prediction, graded_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            final_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                epoch_time = time.time() - epoch_start
                print(
                    f"Epoch {epoch + 1:3d}/{self.epochs}  "
                    f"loss={final_loss:.6f}  "
                    f"({epoch_time:.2f}s/epoch)"
                )

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training complete in {total_time:.1f}s ({total_time / 60:.1f} min)")

        torch.save(model.state_dict(), self.output_path)
        print(f"Saved model to {self.output_path}")

        return final_loss
