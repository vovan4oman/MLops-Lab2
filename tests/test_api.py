from fastapi.testclient import TestClient

from ml.train import train_and_save
from app.main import app, MODEL_PATH

# Гарантуємо існування файлу моделі перед запуском API-тестів
if not MODEL_PATH.exists():
    train_and_save(MODEL_PATH)

client = TestClient(app)

# Типовий зразок вина класу 0 (перший клас у Wine dataset)
VALID_PAYLOAD = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.80,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315": 3.92,
    "proline": 1065.0,
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["variant"] == 10


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["class_id"] in (0, 1, 2)
    assert body["class_name"] in ("class_0", "class_1", "class_2")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_returns_class_0_for_known_sample():
    """Перевіряє, що еталонний зразок класифікується як class_0."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["class_name"] == "class_0"


def test_predict_invalid_input():
    """Перевіряє, що Pydantic повертає 422 при некоректних даних."""
    payload = {"alcohol": "not-a-number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_missing_fields():
    """Перевіряє, що відсутні поля також призводять до 422."""
    payload = {"alcohol": 13.5}  # Бракує решти 12 полів
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
