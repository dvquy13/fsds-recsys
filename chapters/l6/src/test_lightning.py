# poetry run python src/test_lightning.py --accelerator cpu --devices 2

import argparse
import sys

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.strategies import DDPStrategy  # For multi-GPU training
from loguru import logger
from torch.utils.data import DataLoader, Dataset


# Custom Dataset (Example using random data)
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        # For simplicity, targets are also random
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


# Simple model
class ToyModel(L.LightningModule):  # Updated to use L.LightningModule
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)


# DataModule to handle data loading
class RandomDataModule(L.LightningDataModule):  # Updated to use L.LightningDataModule
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = RandomDataset(size=10, length=64)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PyTorch Lightning Example")
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size per process"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="type of device to use: 'cpu', 'gpu'",
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="number of devices to use (GPUs or CPUs)"
    )
    args = parser.parse_args()

    # Initialize logger
    logger.remove()  # Remove default logger
    logger.add(
        sys.stderr,
        format=f"<green>{{time}}</green> | <level>{{level}}</level> | <cyan>{{message}}</cyan>",
        level="INFO",
    )

    logger.info("Starting training with PyTorch Lightning")

    # Initialize the model and datamodule
    model = ToyModel()
    datamodule = RandomDataModule(batch_size=args.batch_size)

    # Create a Trainer using L.Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,  # 'cpu' or 'gpu'
        devices=args.devices,  # Number of devices
        strategy=(
            DDPStrategy(find_unused_parameters=False) if args.devices > 1 else None
        ),
        logger=False,  # You can integrate custom loggers if needed
    )

    # Start training
    trainer.fit(model, datamodule)

    logger.info("Training completed.")
