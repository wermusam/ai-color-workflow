"""Color decision list (ASC CDL) for the AI color workflow."""

import numpy as np


class Cdl:
    """Holds the 10 numbers of an ASC CDL primary grade.

    slope, offset, and power are each an (R, G, B) triple; saturation is a
    single value applied across all channels.
    """

    def __init__(
        self,
        slope: tuple[float, float, float],
        offset: tuple[float, float, float],
        power: tuple[float, float, float],
        saturation: float,
    ) -> None:
        self.slope = slope
        self.offset = offset
        self.power = power
        self.saturation = saturation

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the grade to a float image with values in [0, 1].

        For each channel: out = (in * slope + offset) ** power. A value that
        goes negative before the power step is clamped to 0, which the ASC CDL
        spec requires (a fractional power of a negative number isn't real).
        Saturation then blends each channel toward the pixel's luma.
        """
        graded = image.astype(np.float32).copy()

        red = np.clip(image[:, :, 0] * self.slope[0] + self.offset[0], 0.0, None)
        green = np.clip(image[:, :, 1] * self.slope[1] + self.offset[1], 0.0, None)
        blue = np.clip(image[:, :, 2] * self.slope[2] + self.offset[2], 0.0, None)

        graded[:, :, 0] = red ** self.power[0]
        graded[:, :, 1] = green ** self.power[1]
        graded[:, :, 2] = blue ** self.power[2]

        luma = 0.2126 * graded[:, :, 0] + 0.7152 * graded[:, :, 1] + 0.0722 * graded[:, :, 2]
        graded[:, :, 0] = luma + self.saturation * (graded[:, :, 0] - luma)
        graded[:, :, 1] = luma + self.saturation * (graded[:, :, 1] - luma)
        graded[:, :, 2] = luma + self.saturation * (graded[:, :, 2] - luma)

        return np.clip(graded, 0.0, 1.0)
