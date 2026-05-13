from pathlib import Path

import joblib
import numpy as np

from ml.train import train_and_save


def test_train_creates_model_file(tmp_path: Path):
    """Перевіряє, що після тренування файл моделі існує."""
    model_file = tmp_path / "model.joblib"
    accuracy = train_and_save(model_path=model_file)

    assert model_file.exists(), "Файл моделі має бути створений"
    assert 0.0 <= accuracy <= 1.0, "Accuracy має бути коректним числом у діапазоні [0, 1]"
    assert accuracy > 0.85, f"Очікувано accuracy > 0.85, отримано {accuracy}"


def test_model_predicts_three_classes(tmp_path: Path):
    """Перевіряє, що модель повертає клас із {0, 1, 2}."""
    model_file = tmp_path / "model.joblib"
    train_and_save(model_path=model_file)
    model = joblib.load(model_file)

    # Типовий зразок вина класу 0 з датасету Wine
    sample = [[14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]
    pred = model.predict(sample)

    assert pred[0] in (0, 1, 2), f"Клас має бути одним із 0/1/2, отримано {pred[0]}"


def test_model_predict_proba_sums_to_one(tmp_path: Path):
    """Перевіряє, що сума ймовірностей дорівнює 1."""
    model_file = tmp_path / "model.joblib"
    train_and_save(model_path=model_file)
    model = joblib.load(model_file)

    sample = [[13.20, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050.0]]
    proba = model.predict_proba(sample)[0]

    assert abs(proba.sum() - 1.0) < 1e-6, "Сума ймовірностей має дорівнювати 1"
    assert len(proba) == 3, "Має бути 3 класи"
