"""Tests for CdlPredictor."""

from pathlib import Path

import pytest

from src.cdl import Cdl
from src.cdl_predictor import CdlPredictor


def test_cdl_predictor_can_be_constructed() -> None:
    """CdlPredictor should build and store its settings."""
    predictor = CdlPredictor(model_path="models/cdlnet.pt")
    assert predictor.model_path.name == "cdlnet.pt"
    assert predictor.image_size == 128
    assert predictor._model is None


def test_cdl_predictor_predicts_a_cdl() -> None:
    """predict() should return a Cdl with positive slope and power."""
    model_path = Path("models/cdlnet.pt")
    if not model_path.exists():
        pytest.skip("no trained model at models/cdlnet.pt; run scripts.train_cdl first")

    predictor = CdlPredictor(model_path=str(model_path))
    grade = predictor.predict("data/ungraded/sunset_1.jpg")

    assert isinstance(grade, Cdl)
    assert min(grade.slope) > 0
    assert min(grade.power) > 0
