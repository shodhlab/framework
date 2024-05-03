import json


def vocab_builder(words, vocab_dir):
    vocab = {}
    rev_vocab = {}
    i = 0
    for word in words:
        if word not in vocab:
            vocab[word] = i + 1
            rev_vocab[i + 1] = word
            i += 1
    json.dump(vocab, open(f"{vocab_dir}/vocab.json", "w"))
    json.dump(rev_vocab, open(f"{vocab_dir}/rev_vocab.json", "w"))
    return vocab
