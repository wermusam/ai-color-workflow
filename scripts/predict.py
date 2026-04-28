"""Run inference on a single image or all images in a directory."""

import argparse

from src.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict graded image(s) using the trained color grading model."
    )
    parser.add_argument(
        "--input",
        help="Path to a single input image. If set, --output is required.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the predicted image (used with --input).",
    )
    parser.add_argument(
        "--input-dir",
        default="data/ungraded",
        help="Directory of inputs for batch prediction (default: data/ungraded).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/predicted",
        help="Directory to save batch predictions (default: data/predicted).",
    )
    parser.add_argument(
        "--model",
        default="models/colorgrade.pt",
        help="Path to the trained model file (default: models/colorgrade.pt).",
    )
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model)

    if args.input:
        if not args.output:
            parser.error("--output is required when --input is provided")
        predictor.predict(input_path=args.input, output_path=args.output)
        print(f"Saved prediction to {args.output}")
    else:
        predictor.predict_all(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()