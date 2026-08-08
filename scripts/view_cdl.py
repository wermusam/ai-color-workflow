from src.viewer import PairViewer

if __name__ == "__main__":
    PairViewer(
        panels=[
            ("Ungraded", "data/ungraded"),
            ("CDL Prediction", "data/cdl_predicted"),
            ("Graded (target)", "data/graded"),
        ]
    ).run()
