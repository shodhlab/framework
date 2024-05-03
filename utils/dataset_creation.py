import torch


def dataset_creation(words, vocab, seq_len, data_dir):
    x = []
    y = []
    temp = torch.zeros(seq_len, dtype=torch.long)

    for i in range(len(words) - 1):
        x.append(temp)
        y.append(vocab[words[i]])
        temp = temp[1:]
        temp = torch.cat((temp, torch.tensor([vocab[words[i]]], dtype=torch.long)))

    x = torch.stack(x)
    y = torch.tensor(y)
    torch.save((x, y), f"{data_dir}/data.pt")
    return x, y
