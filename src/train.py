"""Train a RandomForest classifier on the Iris dataset and save it to disk."""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    target_names = data.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": X.columns.tolist(), "target_names": target_names},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    train()
