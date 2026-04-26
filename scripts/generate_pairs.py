"""Generate ungraded/graded training pairs from raw sunset images."""

from pathlib import Path

from src.dataset import PairDataset
from src.transforms import GainTransform


def main() -> None:
    """Run pair generation on the raw sunset images."""
    raw_dir = Path("data/raw")
    graded_dir = Path("data/graded")
    ungraded_dir = Path("data/ungraded")

    transform = GainTransform(red=1.3, green=1.05, blue=0.8)

    dataset = PairDataset(
        raw_dir=raw_dir,
        graded_dir=graded_dir,
        ungraded_dir=ungraded_dir,
        transform=transform,
    )

    count = dataset.generate_pairs()
    print(f"Generated {count} pairs.")
    print(f"Graded images saved to: {graded_dir}")
    print(f"Ungraded images saved to: {ungraded_dir}")


if __name__ == "__main__":
    main()