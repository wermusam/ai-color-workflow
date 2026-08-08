"""Predicts a CDL grade for an image using a trained CdlNet."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.cdl import Cdl
from src.cdl_net import CdlNet


class CdlPredictor:
    """Loads a trained CdlNet and predicts a Cdl grade for an image."""

    def __init__(self, model_path: str, image_size: int = 128) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self._model: CdlNet | None = None
        self._to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _load_model(self) -> CdlNet:
        """Load the trained weights, caching after the first call."""
        if self._model is None:
            model = CdlNet()
            model.load_state_dict(torch.load(self.model_path, weights_only=True))
            model.eval()
            self._model = model
        return self._model

    def predict(self, image_path: str) -> Cdl:
        """Predict a Cdl grade for a single image."""
        image = Image.open(image_path).convert("RGB")
        tensor = self._to_tensor(image).unsqueeze(0)

        model = self._load_model()
        with torch.no_grad():
            numbers = model(tensor)[0]

        slope = (float(numbers[0]), float(numbers[1]), float(numbers[2]))
        offset = (float(numbers[3]), float(numbers[4]), float(numbers[5]))
        power = (float(numbers[6]), float(numbers[7]), float(numbers[8]))
        saturation = float(numbers[9])
        return Cdl(slope, offset, power, saturation)
