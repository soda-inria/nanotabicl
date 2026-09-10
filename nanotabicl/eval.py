"""Evaluate a checkpoint by in-context prediction on small real-world datasets from scikit-learn."""
import sys

from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from .interface import NanoTabICLClassifier, NanoTabICLRegressor, load_model

REAL_DATASETS = {
    "classification": [
        ("iris", lambda: datasets.load_iris(return_X_y=True)),
        ("wine", lambda: datasets.load_wine(return_X_y=True)),
        ("breast_cancer", lambda: datasets.load_breast_cancer(return_X_y=True)),
        ("digits", lambda: datasets.load_digits(return_X_y=True)),
    ],
    "regression": [
        ("diabetes", lambda: datasets.load_diabetes(return_X_y=True)),
        ("friedman1", lambda: datasets.make_friedman1(n_samples=600, noise=1.0, random_state=0)),
        ("linear", lambda: datasets.make_regression(n_samples=600, n_features=20, noise=10.0, random_state=0)),
    ],
}


def evaluate(model, task: str, max_rows: int = 1024, seed: int = 0) -> dict[str, float]:
    """Returns test accuracy (classification) or R^2 (regression) on 50/50 splits of each dataset."""
    regression = task == "regression"
    estimator = (NanoTabICLRegressor if regression else NanoTabICLClassifier)(model=model)
    scores = {}
    for name, load in REAL_DATASETS[task]:
        x, y = load()
        n = min(len(x), max_rows)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n // 2, test_size=n - n // 2, random_state=seed)
        pred = estimator.fit(x_train, y_train).predict(x_test)
        scores[name] = r2_score(y_test, pred) if regression else accuracy_score(y_test, pred)
    return scores


if __name__ == "__main__":  # usage: python -m nanotabicl.eval runs/default/latest.pt
    model, cfg = load_model(sys.argv[1])
    for name, score in evaluate(model, cfg.data.task).items():
        print(f"{name:>14}: {score:.3f}")
