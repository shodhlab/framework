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
        
    def top_k_sampling(self,logits, k):
        values, indices = torch.topk(logits, k)
        probs = torch.softmax(values, dim=-1)
        chosen_index = torch.multinomial(probs, 1)
        return indices[chosen_index].item()

    def top_p_sampling(self, logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        chosen_index = torch.multinomial(probs, 1)
        return chosen_index.item()
    
    def penalize_repeats(self, logits, token_ids, penalty, N):
        recent_tokens = token_ids[-N:]
        for token in set(recent_tokens):
            logits[token] -= penalty
        return logits

    def infer(self, text, max_length, top_k=40, top_p=.5, temperature=0.7, penalty=1.18, N=5):
        x = self.tokenizer.encode(text, out_type=int)
        outputs = []
        for _ in range(max_length):
            logits = self.get_output(x)
            logits = logits.squeeze()[-1]

            # Apply temperature scaling
            logits = logits / temperature

            # Penalize repeating tokens
            logits = self.penalize_repeats(logits, x, penalty, N)

            if top_k > 0:
                op = self.top_k_sampling(logits, top_k)
            elif top_p > 0.0:
                op = self.top_p_sampling(logits, top_p)
            else:
                op = torch.argmax(torch.softmax(logits,dim=-1), dim=-1).item()

            x.append(op)
            outputs.append(op)
            if op == self.tokenizer.eos_id():
                break
        return self.tokenizer.decode(outputs)