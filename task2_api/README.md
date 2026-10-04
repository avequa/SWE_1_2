# Задание 2. Сервис тональности отзывов

Выбрана модель rubert-tiny2
Веб-интерфейс на Go, тесты в GitHub Actions, сборка в Docker, мониторинг в Prometheus и Grafana

В `task1_models/text_sentiment.py` rubert-tiny2 оказалась и точнее (macro-F1 0.84 против 0.73), и в 27 раз быстрее большой модели. Здесь та же модель с доступом по HTTP

## Архитектура

Как проходит запрос:

1. Пользователь вводит отзывы на странице веб-интерфейса
2. Веб-интерфейс отправляет их в API: `POST /predict/batch`
3. API проверяет вход; некорректный запрос получает 422 и до модели не доходит
4. Модель возвращает вероятности трёх классов, API считает сводку и обновляет метрики
5. Веб-интерфейс показывает таблицу «отзыв - тональность - уверенность» и добавляет новые отзывы к сводке за сессию
6. Prometheus забирает метрики с `/metrics`, Grafana рисует по ним графики

**Тесты:**

```bash
cd task2_api
pytest -v
```

**Docker:**

```bash
cd task2_api
docker compose up --build
python load_test.py         # нагрузка для графиков
docker compose logs api web # логи
docker compose down
```

| Адрес | Что там |
|---|---|
| http://localhost:8080 | веб-интерфейс: проверка отзывов, сводка за сессию, кнопки в Grafana и Swagger |
| http://localhost:8000/docs | Swagger: документация API и запросы из браузера |
| http://localhost:8000/metrics | метрики для Prometheus |
| http://localhost:9090/targets | Prometheus: цель `sentiment-api` должна быть UP |
| http://localhost:3000 | Grafana, дашборд *Sentiment API* |

## API

| Метод | Адрес | Что делает |
|---|---|---|
| GET | `/health` | проверка, что сервис жив |
| POST | `/predict` | тональность одного отзыва |
| POST | `/predict/batch` | тональность до 100 отзывов и сводка по классам |
| GET | `/metrics` | метрики в формате Prometheus |

## Мониторинг

| Метрика | Тип | Метки | Что показывает |
|---|---|---|---|
| `http_requests_total` | Counter | method, path, status | число запросов и коды ответа |
| `http_request_duration_seconds` | Histogram | path | время обработки запроса |
| `model_inference_seconds` | Histogram | - | время работы модели |
| `model_predictions_total` | Counter | label | сколько отзывов в каждом классе |

## Файлы

**`app/main.py`** - сам сервис. Модель загружается один раз при старте, а не на каждый запрос. Middleware вокруг каждого запроса считает число запросов, код ответа и время; неизвестные адреса пишутся как `other`, чтобы мусорные URL не плодили метрики.

**`tests/test_api.py`** — тесты, `TestClient` вызывает приложение напрямую, без запуска сервера. Проверяется: сервис жив; формат ответа; явно хороший и плохой отзыв; неправильный ввод 422; пакет сохраняет порядок и верно считает сводку; 3 нарушения ограничений пакета

**`requirements.txt`** - зависимости API и тестов

**`Dockerfile`** - образ API: Python 3.12, зависимости

**`.dockerignore`** - докеригнор

**`docker-compose.yml`** - сборка четырех сервисов: `api`, `web`, `prometheus`, `grafana`

**`web/main.go`** веб на Go. `ML_API_URL` - куда отправлять отзывы, `GRAFANA_URL` и `API_DOCS_URL` - куда ведут кнопки на странице

**`web/index.html`** - шаблон страницы

**`web/Dockerfile`** - двухэтапная сборка: компиляция `golang`, итоговый образ на `alpine` только с бинарником

**`monitoring/prometheus.yml`** - Prometheus каждые 5 секунд забирает `api:8000/metrics`

**`monitoring/grafana/provisioning/datasources/prometheus.yml`** - Grafana при старте сама подключает Prometheus

**`monitoring/grafana/provisioning/dashboards/dashboards.yml`** - Grafana при старте загружает дашборды из папки `dashboards`

**`monitoring/grafana/dashboards/sentiment-api.json`** - дашборд из шести панелей

**`load_test.py`** - нагрузка для графиков: 70% одиночных запросов, 20% пакетных, 10% заведомо неправильных

**`../.github/workflows/tests.yml`** - GitHub Actions на каждый push