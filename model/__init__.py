import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from model.Decoder import Decoder
from model.PositionalEncoding import Learned, RoPE, Cosine
from model.Loss import ChunkedCrossEntropyLoss, CrossEntropyLoss

import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0

            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)


class Transformer(pl.LightningModule):
    def __init__(self, config, vocabSize, dtype):
        super().__init__()
        self.config = config
        self.batchSize = config["batch_size"]
        self.contextLength = config["context_length"]
        self.embeddingDim = config["embedding_dimension"]
        self.numHeads = config["num_heads"]
        self.numLayers = config["num_layers"]
        self.dropout = config["dropout"]
        self.vocabSize = vocabSize
        self.external_dtype = dtype

        self.inputEmbed = nn.Embedding(
            self.vocabSize, self.embeddingDim, dtype=self.external_dtype
        )
        # self.pe = RoPE(self.contextLength, self.embeddingDim, self.external_dtype)
        self.pe = Learned(self.contextLength, self.embeddingDim, self.external_dtype)
        self.decoder = Decoder(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.numHeads,
            self.numLayers,
            self.dropout,
            self.external_dtype,
        )
        self.linear = nn.Linear(
            self.embeddingDim, self.vocabSize, dtype=self.external_dtype
        )

        self.loss_fn = ChunkedCrossEntropyLoss()
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
        x = self.inputEmbed(x)
        x = self.pe(x)
        x = self.decoder(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        y_full = torch.cat([x,y.unsqueeze(-1)],dim=-1)[:,1:]
        loss = self.loss_fn(output.reshape(output.shape[0]*output.shape[1], 32_000), y_full)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        y_full = torch.cat([x,y.unsqueeze(-1)],dim=-1)[:,1:]
        loss = self.loss_fn(output.reshape(output.shape[0]*output.shape[1], 32_000), y_full)
        accuracy = self.accuracy(output[:,-1,:], y)
        dict_log = {
            "val_loss": loss,
            "val_accuracy": accuracy
        }
        self.log_dict(dict_log, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        y_full = torch.cat([x,y.unsqueeze(-1)],dim=-1)[:,1:]
        loss = self.loss_fn(output.reshape(output.shape[0]*output.shape[1], 32_000), y_full)
        accuracy = self.accuracy(output[:,-1, :], y)
        f1_score = self.f1_score(output[:,-1, :], y)
        precision = self.precision(output[:,-1, :], y)
        recall = self.recall(output[:,-1, :], y)
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
        with torch.no_grad():
            output = self.forward(x)
        # output = nn.Softmax(dim=2)(output)
        return output

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.config["lr"],
    #         weight_decay=self.config["weight_decay"],
    #     )
    #     lr_scheduler = {
    #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6
    #         ),
    #         "monitor": "val_loss",
    #         "name": "lr_scheduler",
    #     }
    #     return [optimizer], [lr_scheduler]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        lr_scheduler = {
            "scheduler": CosineAnnealingWarmRestartsDecay(
                optimizer, 
                T_0=self.config["T_0"], 
                T_mult=self.config["T_mult"], 
                eta_min=self.config["eta_min"], 
                decay=self.config["decay"]
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]