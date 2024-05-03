import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from model.Decoder import Decoder
from model.Embedding import InputEmbedding


class Transformer(pl.LightningModule):
    def __init__(self, config, vocabSize):
        super().__init__()
        self.config = config
        self.batchSize = config["batch_size"]
        self.context = config["sequence_length"]
        self.embedding = config["embedding_size"]
        self.heads = config["heads"]
        self.layers = config["layers"]
        self.groups = config["groups"]
        self.vocabSize = vocabSize

        self.input_embedding = InputEmbedding(
            self.context, self.vocabSize, self.embedding
        )
        self.decoder = Decoder(
            self.batchSize,
            self.context,
            self.embedding,
            self.heads,
            self.layers,
            self.groups,
        )
        self.linear = nn.Linear(self.embedding, self.vocabSize)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.vocabSize
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=self.vocabSize
        )
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=self.vocabSize
        )
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.vocabSize)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.decoder(x)
        x = self.linear(x)
        return x

    def training_step(self, batch):
        x, y = batch
        output = self.forward(x)[:, 0, :]

        loss = self.loss_fn(output, y)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        output = self.forward(x)[:, 0, :]
        loss = self.loss_fn(output, y)
        accuracy = self.accuracy(output, y)
        f1_score = self.f1_score(output, y)
        precision = self.precision(output, y)
        recall = self.recall(output, y)
        dict_log = {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "val_f1_score": f1_score,
            "val_precision": precision,
            "val_recall": recall,
        }
        self.log_dict(dict_log, sync_dist=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        output = self.forward(x)[:, 0, :]
        loss = self.loss_fn(output, y)
        accuracy = self.accuracy(output, y)
        f1_score = self.f1_score(output, y)
        precision = self.precision(output, y)
        recall = self.recall(output, y)
        dict_log = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_f1_score": f1_score,
            "test_precision": precision,
            "test_recall": recall,
        }
        self.log_dict(dict_log, sync_dist=True)
        return loss

    def predict_step(self, x):
        output = self.forward(x)[:, 0, :]
        output = torch.argmax(output, dim=1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10, eta_min=0.001
            ),
            "name": "lr_sched",
        }
        return [optimizer], [lr_scheduler]
