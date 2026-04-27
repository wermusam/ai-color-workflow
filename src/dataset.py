"""Dataset class for managing training pairs."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.transforms import GainTransform


class PairDataset:
    """Generates and manages ungraded/graded image pairs for training."""

    def __init__(
        self,
        raw_dir: Path,
        graded_dir: Path,
        ungraded_dir: Path,
        transform: GainTransform,
    ) -> None:
        self.raw_dir = raw_dir
        self.graded_dir = graded_dir
        self.ungraded_dir = ungraded_dir
        self.transform = transform

    def generate_pairs(self) -> int:
        """Read raw images, save graded copies, generate ungraded versions.

        Returns:
            The number of pairs generated.
        """
        self.graded_dir.mkdir(parents=True, exist_ok=True)
        self.ungraded_dir.mkdir(parents=True, exist_ok=True)

        raw_images = sorted(self.raw_dir.glob("*.jpg"))
        count = 0

        for raw_path in raw_images:
            graded = np.array(Image.open(raw_path))
            ungraded = self.transform.invert(graded)

            Image.fromarray(graded).save(self.graded_dir / raw_path.name)
            Image.fromarray(ungraded).save(self.ungraded_dir / raw_path.name)

            count += 1

        return count
    

class PairImageDataset(Dataset):
    """PyTorch Dataset that loads ungraded/graded image pairs as tensors."""

    def __init__(
        self,
        ungraded_dir: Path,
        graded_dir: Path,
        image_size: int = 256,
    ) -> None:
        self.ungraded_dir = ungraded_dir
        self.graded_dir = graded_dir
        self.filenames = sorted(
            f.name for f in self.ungraded_dir.iterdir() if f.is_file()
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.filenames[index]
        ungraded = Image.open(self.ungraded_dir / name).convert("RGB")
        graded = Image.open(self.graded_dir / name).convert("RGB")
        return self.transform(ungraded), self.transform(graded)