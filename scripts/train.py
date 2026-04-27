from src.trainer import Trainer

if __name__ == "__main__":
    Trainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/colorgrade.pt",
    ).run()