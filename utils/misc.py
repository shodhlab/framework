import os
import time
import json
import string
from sentencepiece import SentencePieceProcessor

punctuations = string.punctuation


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


def tokenizer(config):
    sp = SentencePieceProcessor(
        os.path.join(config["vocab_dir"], config["tokenizer"]) + ".model"
    )
    return sp


def vocab(config):
    vocab = json.load(open(f"{config['vocab_dir']}/vocab.json", "r"))
    return vocab


def rev_vocab(config):
    rev_vocab = json.load(open(f"{config['vocab_dir']}/rev_vocab.json", "r"))
    return rev_vocab
