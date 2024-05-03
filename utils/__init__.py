import os
import json
import torch

from utils.vocab import vocab_builder
from utils.tokenization import tokenize
from utils.process_file import process_file
from utils.dataset_creation import dataset_creation


def preprocess(config, seq_len):
    vocab_dir = config["vocab_dir"]
    data_dir = config["data_dir"]
    if os.path.exists(f"{data_dir}/data.pt") and os.path.exists(
        f"{vocab_dir}/vocab.json"
    ):
        data = torch.load(f"{data_dir}/data.pt")
        vocab = json.load(open(f"{vocab_dir}/vocab.json", "r"))
    else:
        words = process_file(config["input"])
        words = tokenize(words)
        vocab = vocab_builder(words, vocab_dir)
        data = dataset_creation(words, vocab, seq_len, data_dir)
    return data, vocab
