from src.cdl_trainer import CdlTrainer

if __name__ == "__main__":
    CdlTrainer(
        ungraded_dir="data/ungraded",
        graded_dir="data/graded",
        output_path="models/cdlnet.pt",
    ).run()
