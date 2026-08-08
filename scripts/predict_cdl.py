from src.cdl_predictor import CdlPredictor

if __name__ == "__main__":
    CdlPredictor(model_path="models/cdlnet.pt").predict_all(
        input_dir="data/ungraded",
        output_dir="data/cdl_predicted",
    )
