import time
import json
import string

punctuations = string.punctuation


def clean_text(text):
    cleaned_text = []
    for word in text:
        word = word.lower()
        word = "".join([char for char in word if char not in punctuations])
        cleaned_text.append(word)
    return cleaned_text


def measure_time(start_time=None):
    if start_time is None:
        return time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def load_configs():
    with open("./config/preprocess.json", "r") as f:
        preprocess_config = json.load(f)
    with open("./config/train.json", "r") as f:
        train_config = json.load(f)
    with open("./config/deepspeed.json", "r") as f:
        deepspeed_config = json.load(f)
    return preprocess_config, train_config, deepspeed_config
