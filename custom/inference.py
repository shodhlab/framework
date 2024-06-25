import os
import torch
from utils.misc import load_checkpoint
from sentencepiece import SentencePieceProcessor

class Inference:
    def __init__(self, modelClass, config, checkpoint_path=None):
        self.modelClass = modelClass
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = load_checkpoint(self.checkpoint_path)
        self.model = (
            self.modelClass.load_from_checkpoint(
                self.checkpoint_path,
                config=self.config.train,
                # Ensure these parameters are correct
                vocabSize=self.config.preprocess["vocab_size"],
                dtype=self.config.dtype,
            )
            .to(self.config.dtype)
            .to(self.device)
        ).eval()
        self.tokenizer = SentencePieceProcessor(
            model_file=os.path.join(
                config.preprocess["vocab_dir"],
                config.preprocess["tokenizer"] + ".model",
            )
        )

    def pad_sequence(self, x, max_length):
        if len(x) < max_length:
            x = [self.tokenizer.pad_id()] * (max_length - len(x)) + x
        if len(x) > max_length:
            x = x[:max_length]  # Fix the slicing to take the first max_length elements
        return x

    def get_output(self, x):
        inp = self.pad_sequence(x, self.config.train["context_length"])
        inp = torch.tensor(inp, dtype=torch.int64).unsqueeze(0).to(self.device)
        with torch.no_grad():  # Ensure no gradient computation
            op = self.model.predict_step(inp)
        return op

    def test(self):
        x = torch.randint(
            0,
            self.config.preprocess["vocab_size"],
            (self.config.train["batch_size"], self.config.train["context_length"]),
        ).to(self.device)
        print(x.shape)
        with torch.no_grad():  # Ensure no gradient computation
            op = self.model.predict_step(x)
        print(op.shape)

    def infer(self, text, max_length):
        x = self.tokenizer.encode(text, out_type=int)
        outputs = []
        for _ in range(max_length):
            op = self.get_output(x)
            op = torch.argmax(op, dim=-1)
            op = op.squeeze().tolist()[-1]
            x.append(op)
            outputs.append(op)
            if op == self.tokenizer.eos_id():
                break
        return self.tokenizer.decode(outputs)

