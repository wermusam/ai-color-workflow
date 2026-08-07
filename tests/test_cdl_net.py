"""Tests for CdlNet."""

import torch

from src.cdl_net import CdlNet


def test_cdl_net_can_be_constructed() -> None:
    """CdlNet should build without error."""
    model = CdlNet()
    assert model is not None


def test_cdl_net_outputs_ten_numbers_per_image() -> None:
    """CdlNet should turn a batch of images into 10 numbers per image."""
    model = CdlNet()
    dummy_input = torch.randn(2, 3, 64, 64)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_cdl_net_outputs_stay_in_sensible_ranges() -> None:
    """Slope and power stay positive so the grade math can't blow up."""
    model = CdlNet()
    output = model(torch.randn(4, 3, 32, 32))

    slope = output[:, 0:3]
    power = output[:, 6:9]

    assert (slope > 0).all()
    assert (power > 0).all()
