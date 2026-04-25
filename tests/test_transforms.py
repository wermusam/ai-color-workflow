"""Tests for color transform classes."""

import numpy as np

from src.transforms import GainTransform


def test_gain_transform_apply_multiplies_channels() -> None:
    """apply() should multiply each channel by its gain value."""
    image = np.array(
        [[[100, 100, 100], [100, 100, 100]],
         [[100, 100, 100], [100, 100, 100]]],
        dtype=np.uint8,
    )
    transform = GainTransform(red=1.5, green=1.0, blue=0.5)

    result = transform.apply(image)

    assert result[0, 0, 0] == 150
    assert result[0, 0, 1] == 100
    assert result[0, 0, 2] == 50


def test_gain_transform_invert_recovers_original() -> None:
    """invert(apply(image)) should approximately equal the original image."""
    image = np.array(
        [[[100, 100, 100], [100, 100, 100]],
         [[100, 100, 100], [100, 100, 100]]],
        dtype=np.uint8,
    )
    transform = GainTransform(red=1.5, green=1.0, blue=0.5)

    graded = transform.apply(image)
    recovered = transform.invert(graded)

    np.testing.assert_allclose(recovered, image, atol=2)