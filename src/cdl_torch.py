"""PyTorch version of the CDL grade, used for training."""

import torch


class TorchCdl:
    """A CDL grade in PyTorch, so a model can be trained through it.

    Holds the same 10 numbers as Cdl and applies the same math, but on torch
    tensors instead of numpy arrays.
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

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the grade to a float image tensor (height, width, 3) in [0, 1]."""
        slope = self.slope
        offset = self.offset
        power = self.power
        sat = self.saturation

        red = torch.clamp(image[:, :, 0] * slope[0] + offset[0], min=0.0) ** power[0]
        green = torch.clamp(image[:, :, 1] * slope[1] + offset[1], min=0.0) ** power[1]
        blue = torch.clamp(image[:, :, 2] * slope[2] + offset[2], min=0.0) ** power[2]

        luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        red = luma + sat * (red - luma)
        green = luma + sat * (green - luma)
        blue = luma + sat * (blue - luma)

        graded = torch.stack([red, green, blue], dim=2)
        return torch.clamp(graded, 0.0, 1.0)
