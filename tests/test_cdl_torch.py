"""Tests for the PyTorch CDL grade."""

import numpy as np
import torch

from src.cdl import Cdl
from src.cdl_torch import TorchCdl


def test_torch_cdl_matches_numpy_cdl() -> None:
    """TorchCdl.apply should give the same result as the numpy Cdl.apply."""
    slope = (1.2, 1.05, 0.9)
    offset = (0.02, 0.0, -0.01)
    power = (0.95, 1.0, 1.1)
    saturation = 0.8

    image = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)

    numpy_result = Cdl(slope, offset, power, saturation).apply(image)
    torch_result = TorchCdl(slope, offset, power, saturation).apply(torch.from_numpy(image))

    np.testing.assert_allclose(torch_result.numpy(), numpy_result, atol=1e-5)
