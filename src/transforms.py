"""Color transform classes for AI color workflow."""

import numpy as np

class GainTransform:
    """Multiplies R, G, B channels by separate values to apply a color grade."""

    def __init__(self, red: float, green: float, blue: float) -> None:
        self.red = red
        self.green = green
        self.blue = blue

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the grade: multiply each channel by its gain value."""
        graded = image.astype(np.float32).copy()
        graded[:, :, 0] *= self.red
        graded[:, :, 1] *= self.green
        graded[:, :, 2] *= self.blue
        return np.clip(graded, 0, 255).astype(np.uint8)

    def invert(self, image: np.ndarray) -> np.ndarray:
        """Invert the grade: divide each channel by its gain value."""
        ungraded = image.astype(np.float32).copy()
        ungraded[:, :, 0] /= self.red
        ungraded[:, :, 1] /= self.green
        ungraded[:, :, 2] /= self.blue
        return np.clip(ungraded, 0, 255).astype(np.uint8)