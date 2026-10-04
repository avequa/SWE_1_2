# Task1 & Task2

1. Четыре задачи на готовых моделях - `task1_models/` - тональность отзывов, распознавание речи, категория товара по фото, подсчёт людей на видео
2. Сервис тональности отзывов - `task2_api/`- API на FastAPI, веб-интерфейс на Go, тесты, Docker, Prometheus, Grafana
3. Отчёт - `report/report.pdf` - отчёт по заданиям 1 и 2 

## Установка

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r task2_api/requirements.txt
```

## Запуск

Задание 1:

```bash
cd task1_models
python text_sentiment.py
python audio_speech.py
python image_products.py
python video_people.py
```

Задание 2:

```bash
cd task2_api
pytest -v
docker compose up --build -d
```

```bash
http://localhost:8080 - веб-интерфейс
http://localhost:8000/docs - документация API
http://localhost:9090 - Prometheus
http://localhost:3000 - Grafana
```

## Структура

```
.
├── task1_models/        задание 1
├── task2_api/           задание 2
├── report/              отчёт и скриншоты
├── .github/workflows/   автозапуск тестов
├── requirements.txt     зависимости задания 1
└── README.md
```
