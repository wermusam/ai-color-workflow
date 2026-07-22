"""Tests for PairViewer."""

from pathlib import Path

from src.viewer import PairViewer


def _make_viewer() -> PairViewer:
    return PairViewer(
        panels=[
            ("Raw", "data/raw"),
            ("Graded", "data/graded"),
            ("Ungraded", "data/ungraded"),
        ]
    )


def test_viewer_can_be_constructed() -> None:
    viewer = _make_viewer()
    assert len(viewer.panels) == 3
    assert viewer.panels[0][0] == "Raw"
    assert viewer.panels[0][1].name == "raw"


def test_viewer_lists_all_sunsets() -> None:
    viewer = _make_viewer()
    pairs = viewer.list_pairs()
    expected = len(list(Path("data/raw").glob("*.jpg")))
    assert len(pairs) == expected


def test_viewer_lists_sunsets_in_natural_order() -> None:
    viewer = _make_viewer()
    pairs = viewer.list_pairs()
    assert pairs[0] == "sunset_1.jpg"
    assert pairs[1] == "sunset_2.jpg"
    assert pairs[9] == "sunset_10.jpg"
    assert pairs[10] == "sunset_11.jpg"


def test_viewer_requires_at_least_one_panel() -> None:
    import pytest

    with pytest.raises(ValueError):
        PairViewer(panels=[])