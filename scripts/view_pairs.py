from src.viewer import PairViewer

if __name__ == "__main__":
    PairViewer(
        raw_dir="data/raw",
        graded_dir="data/graded",
        ungraded_dir="data/ungraded",
    ).run()