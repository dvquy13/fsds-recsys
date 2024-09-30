import os
import sys
import argparse

import lightning as L
import mlflow
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

load_dotenv()

sys.path.insert(0, "..")

from src.id_mapper import IDMapper
from src.skipgram.dataset import SkipGramDataset
from src.skipgram.model import SkipGram
from src.skipgram.trainer import LitSkipGram


class Args(BaseModel):
    testing: bool = False
    log_to_mlflow: bool = False
    experiment_name: str = "FSDS RecSys - L6 - Scale training"
    run_name: str = "006-lightning-ddp-multi-gpu"
    notebook_persist_dp: str = None
    random_seed: int = 41

    top_K: int = 100
    top_k: int = 10

    max_epochs: int = 1000
    batch_size: int = 1028

    num_negative_samples: int = 2
    window_size: int = 1

    embedding_dim: int = 128
    early_stopping_patience: int = 5
    learning_rate: float = 0.01
    l2_reg: float = 1e-5

    def init(self):
        self.notebook_persist_dp = os.path.abspath(f"data/{self.run_name}")
        os.makedirs(self.notebook_persist_dp, exist_ok=True)

        if not os.environ.get("MLFLOW_TRACKING_URI"):
            logger.warning(
                f"Environment variable MLFLOW_TRACKING_URI is not set. Setting self.log_to_mlflow to false."
            )
            self.log_to_mlflow = False

        if self.log_to_mlflow:
            logger.info(
                f"Setting up MLflow experiment {self.experiment_name} - run {self.run_name}..."
            )
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name)

        return self


def init_model(n_items, embedding_dim):
    model = SkipGram(n_items, embedding_dim)
    return model


def prepare_dataloaders(args, idm, sequences_fp, val_sequences_fp):
    dataset = SkipGramDataset(
        sequences_fp,
        window_size=args.window_size,
        negative_samples=args.num_negative_samples,
        id_to_idx=idm.item_to_index,
    )
    val_dataset = SkipGramDataset(
        val_sequences_fp,
        dataset.interacted,
        dataset.item_freq,
        window_size=args.window_size,
        negative_samples=args.num_negative_samples,
        id_to_idx=idm.item_to_index,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=val_dataset.collate_fn,
    )

    assert dataset.id_to_idx == idm.item_to_index, "ID Mappings are not matched!"

    return train_loader, val_loader, len(dataset.items)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a SkipGram model with Lightning")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type: cpu, gpu, etc.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs/CPUs) to use")

    args_cli = parser.parse_args()

    # Initialize model arguments
    args = Args().init()

    # Load ID Mapper
    idm = IDMapper().load("../data/idm.json")

    # Prepare DataLoaders
    sequences_fp = "../data/item_sequence.jsonl"
    val_sequences_fp = "../data/val_item_sequence.jsonl"
    train_loader, val_loader, n_items = prepare_dataloaders(
        args, idm, sequences_fp, val_sequences_fp
    )

    # Initialize Model
    model = init_model(n_items, args.embedding_dim)
    lit_model = LitSkipGram(
        model,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        log_dir=args.notebook_persist_dp,
    )

    # Setup Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
        mode="min",
        verbose=False,
    )

    # Setup Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.notebook_persist_dp}/checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Train model
    log_dir = f"{args.notebook_persist_dp}/logs/run"
    trainer = L.Trainer(
        default_root_dir=log_dir,
        max_epochs=args.max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator=args_cli.accelerator,
        devices=args_cli.devices,
        strategy="ddp",
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    logger.info(f"Logs available at {trainer.log_dir}")

    # Save the final model checkpoint
    final_checkpoint_path = f"{args.notebook_persist_dp}/skipgram-model.ckpt"
    trainer.save_checkpoint(final_checkpoint_path)

    logger.info(f"Logs available at {trainer.log_dir}")
    logger.info(f"Model checkpoint saved to {final_checkpoint_path}")

    id_mapping_path = f"{args.notebook_persist_dp}/skipgram_id_mapping.json"
    logger.info(f"Saving id_mapping to {id_mapping_path}...")
    train_loader.dataset.save_id_mappings(id_mapping_path)


if __name__ == "__main__":
    main()