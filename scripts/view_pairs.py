from src.viewer import PairViewer

if __name__ == "__main__":
    PairViewer(
        panels=[
            ("Raw", "data/raw"),
            ("Graded", "data/graded"),
            ("Ungraded", "data/ungraded"),
            ("Prediction", "data/predicted"),
        ]
    ).run()