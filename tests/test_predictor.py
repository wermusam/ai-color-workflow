"""Tests for Predictor."""

from src.predictor import Predictor


def test_predictor_can_be_constructed() -> None:
    predictor = Predictor(model_path="models/colorgrade.pt")
    assert predictor.model_path.name == "colorgrade.pt"
    assert predictor.image_size == 512
    assert predictor._model is None

def test_predictor_predict_all_creates_output_files(tmp_path) -> None:
    predictor = Predictor(model_path="models/colorgrade.pt")
    output_dir = tmp_path / "predicted"
    count = predictor.predict_all(
        input_dir="data/ungraded",
        output_dir=str(output_dir),
    )
    assert count == 12
    assert len(list(output_dir.iterdir())) == 12