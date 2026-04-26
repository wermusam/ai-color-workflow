"""Pair generator class."""

from pathlib import Path

from src.dataset import PairDataset
from src.transforms import GainTransform


class PairGenerator:
    """Generates ungraded/graded training pairs from raw images."""

    def __init__(
        self,
        raw_dir: str,
        graded_dir: str,
        ungraded_dir: str,
        red: float,
        green: float,
        blue: float,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.graded_dir = Path(graded_dir)
        self.ungraded_dir = Path(ungraded_dir)
        self.transform = GainTransform(red=red, green=green, blue=blue)

    def run(self) -> int:
        """Generate the pairs and return the count."""
        dataset = PairDataset(
            raw_dir=self.raw_dir,
            graded_dir=self.graded_dir,
            ungraded_dir=self.ungraded_dir,
            transform=self.transform,
        )
        count = dataset.generate_pairs()
        print(f"Generated {count} pairs.")
        return count