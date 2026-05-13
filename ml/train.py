"""Тренування моделі класифікації вина (Wine dataset).
Варіант 10: датасет Wine, модель Random Forest Classifier.
"""
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

WINE_CLASSES = ["class_0", "class_1", "class_2"]


def train_and_save(model_path: Path = MODEL_PATH) -> float:
    """Навчає Pipeline (StandardScaler + RandomForest) та повертає точність на тестовій вибірці."""
    X, y = load_wine(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    joblib.dump(pipeline, model_path)
    return accuracy


if __name__ == "__main__":
    acc = train_and_save()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved to: {MODEL_PATH}")
