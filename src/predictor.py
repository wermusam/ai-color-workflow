"""Predictor class for running inference with a trained color grading model."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import ColorGradeNet


class Predictor:
    """Loads a trained ColorGradeNet and predicts graded images from ungraded ones."""

    def __init__(
        self,
        model_path: str,
        image_size: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self._model: ColorGradeNet | None = None
        self._to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _load_model(self) -> ColorGradeNet:
        """Load the trained model weights, caching after first load."""
        if self._model is None:
            model = ColorGradeNet()
            model.load_state_dict(torch.load(self.model_path, weights_only=True))
            model.eval()
            self._model = model
        return self._model

    def predict(self, input_path: str, output_path: str) -> None:
        """Predict a graded image from a single ungraded input."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load original image and remember its size
        original = Image.open(input_path).convert("RGB")
        original_size = original.size  # (width, height)

        # Convert to tensor, resize to model's training size, add batch dim
        input_tensor = self._to_tensor(original).unsqueeze(0)

        # Run inference
        model = self._load_model()
        with torch.no_grad():
            prediction = model(input_tensor)

        # Remove batch dim, clamp to [0, 1], convert tensor to PIL image
        prediction = prediction.squeeze(0).clamp(0, 1)
        prediction_pil = transforms.ToPILImage()(prediction)

        # Resize prediction back up to original dimensions
        prediction_pil = prediction_pil.resize(original_size, Image.LANCZOS)

        # Save
        prediction_pil.save(output_path)

    def predict_all(self, input_dir: str, output_dir: str) -> int:
        """Predict graded images for every input in input_dir. Returns count."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(f for f in input_dir.iterdir() if f.is_file())
        count = 0
        for input_path in files:
            output_path = output_dir / input_path.name
            self.predict(input_path, output_path)
            count += 1

        print(f"Predicted {count} images, saved to {output_dir}")
        return count