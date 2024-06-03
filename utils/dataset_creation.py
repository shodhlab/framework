import torch


def dataset_creation(tokens, config, seq_len):
    x = []
    y = []
    temp = torch.zeros(seq_len, dtype=torch.long)

    for i in range(len(tokens) - 1):
        x.append(temp)
        y.append(tokens[i])
        temp = temp[1:]
        temp = torch.cat((temp, torch.tensor([tokens[i]], dtype=torch.long)))

    x = torch.stack(x)
    y = torch.tensor(y)
    data_dir = config["data_dir"]
    torch.save((x, y), f"{data_dir}/data.pt")
    print("Total tokens:", len(tokens))
    return x, y
