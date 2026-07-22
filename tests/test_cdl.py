"""Tests for the color decision list (CDL) class."""

import numpy as np

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


def test_cdl_apply_runs_slope_offset_and_power() -> None:
    """apply() computes (in * slope + offset) ** power for each channel."""
    image = np.full((1, 1, 3), 0.5, dtype=np.float32)
    grade = Cdl(
        slope=(1.2, 1.0, 1.0),
        offset=(0.0, 0.2, 0.0),
        power=(1.0, 1.0, 2.0),
        saturation=1.0,
    )

    result = grade.apply(image)

    # red   = (0.5 * 1.2 + 0.0) ** 1.0 = 0.6  (slope only)
    # green = (0.5 * 1.0 + 0.2) ** 1.0 = 0.7  (offset only)
    # blue  = (0.5 * 1.0 + 0.0) ** 2.0 = 0.25 (power only)
    np.testing.assert_allclose(result[0, 0], [0.6, 0.7, 0.25], atol=1e-6)


def test_cdl_apply_saturation_zero_makes_gray() -> None:
    """saturation=0 collapses a color to its luma (equal in all channels)."""
    image = np.zeros((1, 1, 3), dtype=np.float32)
    image[0, 0, 0] = 1.0  # pure red pixel
    grade = Cdl(
        slope=(1.0, 1.0, 1.0),
        offset=(0.0, 0.0, 0.0),
        power=(1.0, 1.0, 1.0),
        saturation=0.0,
    )

    result = grade.apply(image)

    # luma of pure red = 0.2126 (Rec. 709), so every channel becomes 0.2126.
    np.testing.assert_allclose(result[0, 0], [0.2126, 0.2126, 0.2126], atol=1e-4)
