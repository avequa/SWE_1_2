"""API для определения тональности отзывов покупателей

Модель: seara/rubert-tiny2-russian-sentiment из задания 1

Запуск:        uvicorn app.main:app --reload
Документация:  http://127.0.0.1:8000/docs
Метрики:       http://127.0.0.1:8000/metrics (их забирает Prometheus)
"""

import logging
import time
from typing import Annotated

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from transformers import pipeline

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
MAX_TEXT_LENGTH = 2000  # символов в одном отзыве
MAX_BATCH_SIZE = 100    # отзывов в одном пакетном запросе
LABELS = ["positive", "neutral", "negative"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sentiment-api")

app = FastAPI(title="Sentiment API", description="Тональность отзывов покупателей", version="1.0.0")

# Модель загружается один раз при старте сервиса, а не на каждый запрос
classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, device=-1)
log.info("Модель %s загружена", MODEL_NAME)


# МЕТРИКИ ДЛЯ Prometheus
REQUESTS = Counter("http_requests_total", "Количество HTTP-запросов", ["method", "path", "status"])
REQUEST_TIME = Histogram("http_request_duration_seconds", "Время обработки HTTP-запроса", ["path"])
MODEL_TIME = Histogram("model_inference_seconds", "Время работы модели на один вызов")
PREDICTIONS = Counter("model_predictions_total", "Ответы модели по классам", ["label"])

TRACKED_PATHS = {"/health", "/predict", "/predict/batch"}


@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    """для каждого запроса считаем количество, код ответа и время обработки"""
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path == "/metrics":
        return response # запросы самого Prometheus не считаем

    # неизв адреса в "other", чтобы мусорные URL не плодили метрики
    path = request.url.path if request.url.path in TRACKED_PATHS else "other"
    REQUESTS.labels(request.method, path, response.status_code).inc()
    REQUEST_TIME.labels(path).observe(time.perf_counter() - start)
    return response


# СХЕМЫ ЗАПРОСОВ

# отзыв: строка от 1 до 2000 символов
ReviewText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]


class Review(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: ReviewText


class Reviews(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    texts: list[ReviewText] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


# РАБОТА С МОДЕЛЬЮ

def classify(texts):
    start = time.perf_counter()
    # модель читает не больше 512 токенов, более длинный текст обрезается
    outputs = classifier(texts, truncation=True, max_length=512, batch_size=16)
    MODEL_TIME.observe(time.perf_counter() - start)

    results = []
    for class_scores in outputs:
        scores = {item["label"]: round(item["score"], 4) for item in class_scores}
        label = max(scores, key=scores.get)
        PREDICTIONS.labels(label).inc()
        results.append({"label": label, "scores": scores})
    return results


# АДРЕСА API

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(review: Review):
    """тональность одного отзыва"""
    result = classify([review.text])[0]
    log.info("predict: label=%s length=%d", result["label"], len(review.text))
    return result


@app.post("/predict/batch")
def predict_batch(body: Reviews):
    """тональность списка отзывов, например всех отзывов на один товар"""
    results = classify(body.texts)
    summary = {label: sum(r["label"] == label for r in results) for label in LABELS}
    log.info("predict_batch: size=%d summary=%s", len(results), summary)
    return {"results": results, "summary": summary}


@app.get("/metrics")
def metrics():
    """значения метрик в формате, который понимает prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
