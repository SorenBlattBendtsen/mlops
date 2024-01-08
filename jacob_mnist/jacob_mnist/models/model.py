from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import wandb
from torch import nn
import torch
from pytorch_lightning import LightningModule


class MyAwesomeModel(LightningModule):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10), nn.LogSoftmax(dim=1)
        )

        self.criterium = nn.NLLLoss()

    def forward(self, x):
        if x.shape[1:] != torch.Size([1, 28, 28]):
            raise ValueError("Input tensor has incorrect shape. Expected shape is (batch_size, 1, 28, 28)")
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        images, labels = batch
        # images = images.unsqueeze(1)
        out = self(images)
        # self.logger.experiment.log({'logits': wandb.Histogram(out.detach().cpu().numpy())})
        loss = self.criterium(out, labels)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch):
        images, labels = batch
        #    images = images.unsqueeze(1)
        out = self(images)
        loss = self.criterium(out, labels)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        path = "./data/processed/"
        train_images = torch.load(path + "train_images.pt")
        train_labels = torch.load(path + "train_target.pt")

        train = torch.utils.data.TensorDataset(train_images, train_labels)
        train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
        return train

    def test_dataloader(self):
        path = "./data/processed/"
        test_images = torch.load(path + "test_images.pt")
        test_labels = torch.load(path + "test_target.pt")

        test = torch.utils.data.TensorDataset(test_images, test_labels)
        test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)
        return test
