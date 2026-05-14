# MLOps Lab 2: Wine Classification ML API

**Автор:** Миколайчук Володимир
**Група:** ТР-51мп
**Варіант:** 10  
**Дата:** 14.05.2026

![CI](https://github.com/[ТВІЙ_GITHUB_USERNAME]/[НАЗВА_РЕПО]/actions/workflows/ci.yml/badge.svg)

---

## 📋 Опис проєкту

Лабораторна робота № 2 з курсу MLOps. Реалізовано наскрізний MLOps-конвеєр: від навчання ML-моделі до публічного REST API з CI/CD та хмарним деплоєм.

**Варіант 10:** датасет **Wine** (13 хімічних характеристик вина, 3 класи), модель — **Random Forest Classifier** у складі scikit-learn Pipeline з попереднім масштабуванням ознак (StandardScaler).

---

## 🛠 Стек технологій

| Компонент | Технологія |
|-----------|-----------|
| ML-модель | scikit-learn `RandomForestClassifier` + `StandardScaler` |
| REST API | FastAPI + Uvicorn |
| Валідація | Pydantic v2 |
| Тестування | pytest + httpx |
| Контейнеризація | Docker |
| CI/CD | GitHub Actions |
| Деплой | Render (Docker) |

---

## 🗂 Структура репозиторію

```text
ml-api-lab2/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: тести + docker build
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI застосунок (/, /health, /predict)
│   └── schemas.py          # Pydantic-схеми WineFeatures / PredictionResponse
├── ml/
│   ├── __init__.py
│   └── train.py            # Скрипт тренування (Wine dataset → model.joblib)
├── tests/
│   ├── __init__.py
│   ├── test_model.py       # Unit-тести моделі
│   └── test_api.py         # Інтеграційні тести API
├── model.joblib            # Артефакт навченої моделі (генерується скриптом)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 🚀 Як запустити локально

### 1. Клонування та середовище

```bash
git clone https://github.com/[ТВІЙ_GITHUB_USERNAME]/[НАЗВА_РЕПО].git
cd ml-api-lab2

python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Тренування моделі

```bash
python -m ml.train
# Виведе: Model trained. Test accuracy: 1.0000
# Створить файл model.joblib у корені проєкту
```

### 3. Запуск API

```bash
uvicorn app.main:app --reload
```

Відкрийте у браузері: [http://localhost:8000/docs](http://localhost:8000/docs) — інтерактивна Swagger UI документація.

---

## 🐳 Запуск через Docker

```bash
# Збірка образу (модель тренується всередині)
docker build -t ml-api:lab2 .

# Запуск контейнера
docker run --rm -p 8000:8000 ml-api:lab2

# Перевірка
curl http://localhost:8000/health
```

---

## 🧪 Як запустити тести

```bash
pytest -q
```

Очікуваний результат:

```
......                                                               [100%]
6 passed in X.XXs
```

---

## 🔌 Як працює API

### Ендпоінти

| Метод | URL | Опис |
|-------|-----|------|
| `GET` | `/` | Статус сервісу |
| `GET` | `/health` | Health check (liveness probe) |
| `POST` | `/predict` | Класифікація вина |

### Приклад запиту до `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "proline": 1065.0
  }'
```

### Приклад відповіді

```json
{
  "class_id": 0,
  "class_name": "class_0",
  "probability": 1.0
}
```

---

## 📊 Результати тренування

| Параметр | Значення |
|----------|----------|
| Датасет | Wine (sklearn) |
| Розмір датасету | 178 зразків, 13 ознак, 3 класи |
| Розбивка | 80% train / 20% test, stratify, random_state=42 |
| Препроцесинг | StandardScaler |
| Модель | RandomForestClassifier (n_estimators=100, max_depth=10) |
| Test Accuracy | **1.0000** |

---

## ☁️ Посилання на деплой

🌐 **Render:** `https://[НАЗВА-СЕРВІСУ].onrender.com`

Перевірка:

```bash
curl https://[НАЗВА-СЕРВІСУ].onrender.com/health
# {"status":"healthy","model_loaded":true}
```

> ⚠️ Безкоштовний тариф Render може "засипати" при відсутності трафіку. Перший запит після простою займає ~30 секунд.

---

## 🛠 Troubleshooting

### `FileNotFoundError: model.joblib`
**Причина:** Не запущено скрипт тренування перед стартом API.  
**Рішення:** `python -m ml.train`

### `ModuleNotFoundError` при запуску тестів
**Причина:** Неправильна директорія або не активовано venv.  
**Рішення:** Запускати `pytest` з кореня проєкту при активованому venv.

### Помилка MIME type у MLflow на Windows (білий екран)
**Рішення:** `Win + R` → `regedit` → `HKEY_CLASSES_ROOT\.js` → `Content Type`: змінити з `text/plain` на `application/javascript`. Перезапустити термінал.
