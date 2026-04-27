"""Tests for ColorGradeNet."""

import torch

from src.model import ColorGradeNet


def test_model_can_be_constructed() -> None:
    model = ColorGradeNet()
    assert model is not None


def test_model_output_has_same_shape_as_input() -> None:
    model = ColorGradeNet()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == dummy_input.shape