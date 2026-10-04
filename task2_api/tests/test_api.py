"""Тесты API.  Запуск: pytest -v"""

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_BATCH_SIZE, MAX_TEXT_LENGTH, app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_response_format():
    response = client.post("/predict", json={"text": "Хороший магазин"})
    assert response.status_code == 200
    body = response.json()
    assert set(body["scores"]) == {"positive", "neutral", "negative"}
    assert sum(body["scores"].values()) == pytest.approx(1, abs=0.01)
    assert body["label"] == max(body["scores"], key=body["scores"].get)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Отличный магазин, вежливые продавцы и большой выбор. Всем советую!", "positive"),
        ("Ужасное обслуживание, продавец нахамил, больше сюда не приду.", "negative"),
    ],
)
def test_predict_obvious_reviews(text, expected):
    response = client.post("/predict", json={"text": text})
    assert response.json()["label"] == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},                                  # нет поля text
        {"text": ""},                        # пустой отзыв
        {"text": "     "},                   # одни пробелы
        {"text": 123},                       # не строка
        {"text": "а" * (MAX_TEXT_LENGTH + 1)},  # слишком длинный
        {"review": "Хороший магазин"},       # поле называется не так
    ],
)
def test_predict_rejects_bad_input(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_keeps_order_and_counts():
    texts = [
        "Прекрасный магазин, всё понравилось!",
        "Обманули с ценой на кассе, отвратительно.",
        "Замечательный выбор и приятные цены.",
    ]
    body = client.post("/predict/batch", json={"texts": texts}).json()
    labels = [r["label"] for r in body["results"]]
    assert labels == ["positive", "negative", "positive"]
    assert body["summary"] == {"positive": 2, "neutral": 0, "negative": 1}


@pytest.mark.parametrize(
    "texts",
    [
        [],                                    # пустой список
        ["Нормально", "   "],                  # в списке пустой отзыв
        ["Нормально"] * (MAX_BATCH_SIZE + 1),  # слишком много отзывов
    ],
)
def test_batch_rejects_bad_input(texts):
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422


def test_metrics_are_collected():
    client.post("/predict", json={"text": "Хороший магазин"})
    metrics = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in metrics
    assert "model_predictions_total" in metrics
    assert "model_inference_seconds" in metrics
