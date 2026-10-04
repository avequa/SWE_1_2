"""Задача 2. Голосовой поиск: распознавание русской речи

Три размера модели Whisper с Hugging Face
Данные - 10 коротких голосовых запросов 
из корпуса SberDevices Golos (папка data/audio)

Запуск:  python audio_speech.py
"""

import re
import time

import jiwer
import librosa
from transformers import pipeline

MODELS = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"]

QUERIES = {
    "data/audio/01.wav": "можешь включить сериал теория большого взрыва",
    "data/audio/02.wav": "секретная миссия санты показывай",
    "data/audio/03.wav": "салют макс пояснит",
    "data/audio/04.wav": "открой клетка ай эс ди",
    "data/audio/05.wav": "мелодрама вуди аллена",
    "data/audio/06.wav": "стоимость обмена доллара на гуарани",
    "data/audio/07.wav": "найди передачу рбк на телевизоре",
    "data/audio/08.wav": "нью касл против лестера",
    "data/audio/09.wav": "афина что с ними",
    "data/audio/10.wav": "на номер девятьсот счет номер телефона пополнить",
}

def simplify(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

for name in MODELS:
    print(f"\n=== {name} ===")
    model = pipeline("automatic-speech-recognition", model=name)

    first_audio, _ = librosa.load("data/audio/01.wav", sr=16000)
    model(first_audio)

    answers, recognized = [], []
    start = time.perf_counter()
    for path, answer in QUERIES.items():
        audio, _ = librosa.load(path, sr=16000)
        text = simplify(model(audio, generate_kwargs={"language": "russian"})["text"])
        answers.append(answer)
        recognized.append(text)
        mark = "  " if text == answer else "x "
        print(f"  {mark}{answer}  ->  {text}")
    seconds = time.perf_counter() - start

    print(f"WER (доля ошибочных слов): {jiwer.wer(answers, recognized):.2f}")
    print(f"Время: {seconds / len(QUERIES):.1f} с на запрос")
