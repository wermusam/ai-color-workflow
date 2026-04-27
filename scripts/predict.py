from src.predictor import Predictor

if __name__ == "__main__":
    Predictor(
        model_path="models/colorgrade.pt",
    ).predict_all(
        input_dir="data/ungraded",
        output_dir="data/predicted",
    )