"""Задача 4. Подсчёт покупателей в торговом зале

Две модели YOLO11 (PyTorch, библиотека Ultralytics) находят людей на каждом кадре, 
а трекер ByteTrack даёт каждому человеку постоянный номер
Данные - видео из магазина data/store-aisle-detection.mp4.

Запуск:  python video_people.py
"""

import time

from ultralytics import YOLO

VIDEO = "data/store-aisle-detection.mp4"

MODELS = ["yolo11n.pt", "yolo11s.pt"]

for name in MODELS:
    print(f"\n=== {name} ===")
    model = YOLO(name)

    people = set()
    max_on_frame = 0
    frames = 0
    start = time.perf_counter()

    for result in model.track(VIDEO, classes=[0], conf=0.35, tracker="bytetrack.yaml",
                              vid_stride=2, stream=True, save=True, verbose=False):
        frames += 1
        max_on_frame = max(max_on_frame, len(result.boxes))
        if result.boxes.id is not None:
            people.update(result.boxes.id.int().tolist())
    seconds = time.perf_counter() - start

    print(f"Обработано кадров: {frames}, скорость {frames / seconds:.1f} кадров/с")
    print(f"Уникальных людей (разных номеров трекера): {len(people)}")
    print(f"Больше всего людей в одном кадре: {max_on_frame}")
