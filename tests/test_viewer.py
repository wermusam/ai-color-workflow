"""Tests for PairViewer."""

from src.viewer import PairViewer


def test_viewer_can_be_constructed() -> None:
    viewer = PairViewer(
        raw_dir="data/raw",
        graded_dir="data/graded",
        ungraded_dir="data/ungraded",
    )
    assert viewer.raw_dir.name == "raw"
    assert viewer.graded_dir.name == "graded"
    assert viewer.ungraded_dir.name == "ungraded"

def test_viewer_lists_all_twelve_sunsets() -> None:
    viewer = PairViewer(
        raw_dir="data/raw",
        graded_dir="data/graded",
        ungraded_dir="data/ungraded",
    )
    pairs = viewer.list_pairs()
    assert len(pairs) == 12
    assert pairs[0] == "sunset_1.jpg"


def test_viewer_lists_sunsets_in_natural_order() -> None:
    viewer = PairViewer(
        raw_dir="data/raw",
        graded_dir="data/graded",
        ungraded_dir="data/ungraded",
    )
    pairs = viewer.list_pairs()
    assert pairs[0] == "sunset_1.jpg"
    assert pairs[1] == "sunset_2.jpg"
    assert pairs[9] == "sunset_10.jpg"
    assert pairs[11] == "sunset_12.jpg"