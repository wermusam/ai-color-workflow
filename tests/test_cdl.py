"""Tests for the color decision list (CDL) class."""

from src.cdl import Cdl


def test_cdl_stores_the_ten_numbers() -> None:
    """A Cdl should remember the slope, offset, power, and saturation it is
given."""
    grade = Cdl(
        slope=(1.2, 1.05, 0.9),
        offset=(0.0, 0.0, 0.0),
        power=(1.0, 1.0, 1.0),
        saturation=1.0,
    )

    assert grade.slope == (1.2, 1.05, 0.9)
    assert grade.offset == (0.0, 0.0, 0.0)
    assert grade.power == (1.0, 1.0, 1.0)
    assert grade.saturation == 1.0