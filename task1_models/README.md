# Задание 1. Четыре задачи на готовых моделях

Использованы три фреймворка: Hugging Face Transformers, TensorFlow / Keras и PyTorch / Ultralytics. В каждой задаче сравниваются модели, и выбирается лучшая

| Скрипт | Задача | Фреймворк | Модели | Данные | Метрика |
|---|---|---|---|---|---|
| `text_sentiment.py` | тональность отзывов | Hugging Face | rubert-tiny2, rubert-base | 30 отзывов в коде | accuracy, F1 |
| `audio_speech.py` | голосовой поиск | Hugging Face | Whisper tiny, base, small | 10 записей в `data/audio` | WER |
| `image_products.py` | категория товара по фото | TensorFlow / Keras | MobileNetV2, EfficientNetB0 | 20 фото в `data/images` | доля верных категорий |
| `video_people.py` | подсчёт покупателей | PyTorch / Ultralytics | YOLO11n, YOLO11s + ByteTrack | видео в `data/` | скорость, число людей |

## Запуск

```bash
cd task1_models
python text_sentiment.py
python audio_speech.py
python image_products.py
python video_people.py
```

## Данные

| Данные | Откуда |
|---|---|
| 30 отзывов (список `REVIEWS`) | написаны вручную, по 10 позитивных, нейтральных и негативных |
| `data/audio/01.wav` … `10.wav` | первые 10 записей корпуса [SberDevices Golos](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd), расшифровки — в словаре `QUERIES` |
| `data/images/*.jpg` | каталог интернет-магазина одежды [Fashion Product Images](https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512), по 2 фото на 10 категорий |
| `data/store-aisle-detection.mp4` | [Intel sample-videos](https://github.com/intel-iot-devkit/sample-videos): проход в магазине |

## Файлы

**`text_sentiment.py`** - список `REVIEWS` из пар «отзыв, правильный класс». Для каждой модели `pipeline` скачивает её и прогоняет все отзывы

**`audio_speech.py`** - словарь `QUERIES` «файл -> что в нём сказано». Язык задан явно, `simplify` убирает заглавные буквы и знаки препинания, чтобы модель не штрафовалась за запятые. `jiwer.wer` считает долю ошибочных слов

**`image_products.py`** - словарь `CATALOG` сопоставляет классы ImageNet категориям магазина, словарь `PHOTOS` - фото и правильную категорию. Фото уменьшается до 224×224, нормализуется функцией `preprocess_input` своей модели, модель возвращает пять самых вероятных классов, и берётся первый, который есть в каталоге

**`video_people.py`** - `model.track` сама читает видео, на каждом втором кадре (`vid_stride=2`) ищет только людей (`classes=[0]`) и передаёт рамки трекеру ByteTrack, который выдаёт каждому человеку постоянный номер.

**`requirements.txt`** - зависимости

## Краткое ТЗ по задачам

### 1. Тональность отзывов покупателей - автоматически понять, доволен покупатель, недоволен или пишет нейтрально
**Вход:** текст отзыва
**Выход:** `positive`, `neutral` или `negative`
**Модели:** `seara/rubert-tiny2-russian-sentiment` и `blanchefort/rubert-base-cased-sentiment`

### 2. Голосовой поиск - покупатель говорит запрос, сервис превращает его в текст для поиска
**Вход:** аудиофайл с русской речью
**Выход:** текст
**Модели:** `openai/whisper-tiny`, `whisper-base`, `whisper-small`

### 3. Категория товара по фото - продавец загружает фото, сервис предлагает категорию каталога
**Вход:** фото товара
**Выход:** категория каталога
**Модели:** MobileNetV2 и EfficientNetB0 с весами ImageNet

### 4. Подсчёт покупателей в торговом зале - по видео узнать, сколько людей прошло и сколько было одновременно
**Вход:** видеофайл
**Выход:** число уникальных людей, максимум в кадре, видео с рамками
**Модели:** YOLO11n и YOLO11s + трекер ByteTrack
