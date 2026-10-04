"""Нагрузка на API, чтобы на графиках в Grafana появились данные

Сервис должен быть запущен (docker compose up)
Запуск:  python load_test.py

"""

import random
import time

import httpx2

URL = "http://localhost:8000"
REQUESTS = 300

REVIEWS = [
    "Отличный магазин, большой выбор и вежливые продавцы.",
    "Всё понравилось, цены приятные, приду ещё.",
    "Нормальный магазин, ничего особенного.",
    "Работают до девяти, парковка рядом.",
    "Долго стояла в очереди на кассе, работала одна касса.",
    "Продавец нахамил, товар оказался просроченным. Ужасно.",
    "Цена на полке не совпала с ценой на кассе, обманывают.",
    "Хороший ассортимент, но бывает грязно в торговом зале.",
]

for i in range(1, REQUESTS + 1):
    chance = random.random()
    if chance < 0.7:
        # обычный запрос с одним отзывом
        response = httpx2.post(f"{URL}/predict", json={"text": random.choice(REVIEWS)})
    elif chance < 0.9:
        # пакет из пяти отзывов
        response = httpx2.post(f"{URL}/predict/batch", json={"texts": random.sample(REVIEWS, 5)})
    else:
        # специально неправильный запрос, чтобы на графике ошибок было видно 422
        response = httpx2.post(f"{URL}/predict", json={"text": ""})

    if i % 25 == 0:
        print(f"Отправлено {i} из {REQUESTS}, последний ответ: {response.status_code}")
    time.sleep(random.uniform(0.05, 0.3))
