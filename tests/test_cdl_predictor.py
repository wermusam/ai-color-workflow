"""Tests for CdlPredictor."""

from pathlib import Path

import pytest
from PIL import Image

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


def test_cdl_predictor_predict_and_apply_writes_full_res_image(tmp_path: Path) -> None:
    """predict_and_apply writes a graded image at the input's full resolution."""
    model_path = Path("models/cdlnet.pt")
    if not model_path.exists():
        pytest.skip("no trained model at models/cdlnet.pt; run scripts.train_cdl first")

    predictor = CdlPredictor(model_path=str(model_path))
    input_path = "data/ungraded/sunset_1.jpg"
    output_path = tmp_path / "graded.jpg"

    predictor.predict_and_apply(input_path, str(output_path))

    assert output_path.exists()
    assert Image.open(output_path).size == Image.open(input_path).size


def test_cdl_predictor_predict_all_grades_every_image(tmp_path: Path) -> None:
    """predict_all writes one graded image per input."""
    model_path = Path("models/cdlnet.pt")
    if not model_path.exists():
        pytest.skip("no trained model at models/cdlnet.pt; run scripts.train_cdl first")

    predictor = CdlPredictor(model_path=str(model_path))
    output_dir = tmp_path / "cdl_predicted"

    count = predictor.predict_all("data/ungraded", str(output_dir))

    expected = len(list(Path("data/ungraded").glob("*.jpg")))
    assert count == expected
    assert len(list(output_dir.glob("*.jpg"))) == expected
