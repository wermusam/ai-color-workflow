"""Color decision list (ASC CDL) for the AI color workflow."""

from pathlib import Path

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

    def to_cdl_string(self, cc_id: str = "ai-color-workflow") -> str:
        """Return this grade as an ASC CDL (ColorDecisionList) XML string.

        cc_id names the ColorCorrection so a color tool can reference this grade.
        """
        slope = f"{self.slope[0]} {self.slope[1]} {self.slope[2]}"
        offset = f"{self.offset[0]} {self.offset[1]} {self.offset[2]}"
        power = f"{self.power[0]} {self.power[1]} {self.power[2]}"

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<ColorDecisionList xmlns="urn:ASC:CDL:v1.01">',
            "  <ColorDecision>",
            f'    <ColorCorrection id="{cc_id}">',
            "      <SOPNode>",
            f"        <Slope>{slope}</Slope>",
            f"        <Offset>{offset}</Offset>",
            f"        <Power>{power}</Power>",
            "      </SOPNode>",
            "      <SatNode>",
            f"        <Saturation>{self.saturation}</Saturation>",
            "      </SatNode>",
            "    </ColorCorrection>",
            "  </ColorDecision>",
            "</ColorDecisionList>",
        ]
        return "\n".join(lines) + "\n"

    def export(self, path: str) -> None:
        """Write this grade to an ASC CDL file at the given path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_cdl_string(), encoding="utf-8")
