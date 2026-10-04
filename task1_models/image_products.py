"""Задача 3. Категория товара по фото

Две готовые модели из Keras Applications (TensorFlow) обучены на ImageNet - 1000 классов предметов
Данные - 20 фото из каталога интернет-магазина одежды, по 2 на категорию (папка data/images)

Запуск:  python image_products.py
"""

import time

import numpy as np
from keras.applications import efficientnet, mobilenet_v2
from keras.utils import img_to_array, load_img

# Модель и модуль Keras, в котором лежит её функция подготовки картинки
MODELS = {
    "MobileNetV2": (mobilenet_v2.MobileNetV2, mobilenet_v2),
    "EfficientNetB0": (efficientnet.EfficientNetB0, efficientnet),
}

# Класс ImageNet -> категория магазина
CATALOG = {
    "backpack": "Рюкзаки",
    "sunglasses": "Очки",
    "sunglass": "Очки",
    "running_shoe": "Кроссовки",
    "sandal": "Сандалии",
    "jean": "Джинсы",
    "jersey": "Футболки",
    "wallet": "Кошельки",
    "sock": "Носки",
    "purse": "Сумки",
    "mailbag": "Сумки",
    "perfume": "Парфюмерия",
}

# Фото и правильная категория
PHOTOS = {
    "data/images/backpack_1.jpg": "Рюкзаки",
    "data/images/backpack_2.jpg": "Рюкзаки",
    "data/images/sunglasses_1.jpg": "Очки",
    "data/images/sunglasses_2.jpg": "Очки",
    "data/images/sneakers_1.jpg": "Кроссовки",
    "data/images/sneakers_2.jpg": "Кроссовки",
    "data/images/sandals_1.jpg": "Сандалии",
    "data/images/sandals_2.jpg": "Сандалии",
    "data/images/jeans_1.jpg": "Джинсы",
    "data/images/jeans_2.jpg": "Джинсы",
    "data/images/tshirt_1.jpg": "Футболки",
    "data/images/tshirt_2.jpg": "Футболки",
    "data/images/wallet_1.jpg": "Кошельки",
    "data/images/wallet_2.jpg": "Кошельки",
    "data/images/socks_1.jpg": "Носки",
    "data/images/socks_2.jpg": "Носки",
    "data/images/handbag_1.jpg": "Сумки",
    "data/images/handbag_2.jpg": "Сумки",
    "data/images/perfume_1.jpg": "Парфюмерия",
    "data/images/perfume_2.jpg": "Парфюмерия",
}

for name, (create_model, module) in MODELS.items():
    print(f"\n=== {name} ===")
    model = create_model(weights="imagenet")

    model.predict(np.zeros((1, 224, 224, 3)), verbose=0)

    correct = 0
    start = time.perf_counter()
    for path, answer in PHOTOS.items():
        image = img_to_array(load_img(path, target_size=(224, 224)))
        batch = module.preprocess_input(np.expand_dims(image, axis=0))
        probabilities = model.predict(batch, verbose=0)
        top5 = module.decode_predictions(probabilities, top=5)[0]

        category = "не определена"
        for _, class_name, _ in top5:
            if class_name in CATALOG:
                category = CATALOG[class_name]
                break

        if category == answer:
            correct += 1
        else:
            seen = [class_name for _, class_name, _ in top5[:3]]
            print(f"  ошибка: {path}: {answer} -> {category}, модель видит {seen}")
    seconds = time.perf_counter() - start

    print(f"Верная категория: {correct} из {len(PHOTOS)}")
    print(f"Время: {seconds / len(PHOTOS) * 1000:.0f} мс на фото")
