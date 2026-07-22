"""Tests for Predictor."""

from pathlib import Path

import pytest

from src.predictor import Predictor


def test_predictor_can_be_constructed() -> None:
    predictor = Predictor(model_path="models/colorgrade.pt")
    assert predictor.model_path.name == "colorgrade.pt"
    assert predictor.image_size == 512
    assert predictor._model is None


def test_predictor_predict_all_creates_output_files(tmp_path: Path) -> None:
    model_path = Path("models/colorgrade.pt")
    if not model_path.exists():
        pytest.skip("no trained model at models/colorgrade.pt; run scripts.train first")

    predictor = Predictor(model_path=str(model_path))
    output_dir = tmp_path / "predicted"
    count = predictor.predict_all(
        input_dir="data/ungraded",
        output_dir=str(output_dir),
    )
    expected = len(list(Path("data/ungraded").glob("*.jpg")))
    assert count == expected
    assert len(list(output_dir.iterdir())) == expected
