"""Color decision list (ASC CDL) for the AI color workflow."""


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