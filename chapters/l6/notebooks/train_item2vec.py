import argparse
import os
import sys
from typing import Literal

import mlflow
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

load_dotenv()

sys.path.insert(0, "..")

from src.id_mapper import IDMapper
from src.skipgram.dataset import SkipGramDistributedDataset
from src.skipgram.model import SkipGram
from src.train_utils import MetricLogCallback, MLflowLogCallback, train


class Args(BaseModel):
    testing: bool = False
    log_to_mlflow: bool = True
    experiment_name: str = "FSDS RecSys - L6 - Scale training"
    run_name: str = "002-test-ddp"
    notebook_persist_dp: str = None
    random_seed: int = 41
    device: Literal["cpu", "cuda"] = "cpu"

    top_K: int = 100
    top_k: int = 10

    epochs: int = 1000
    batch_size: int = 128

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


def init_model(n_items, embedding_dim, device):
    model = SkipGram(n_items, embedding_dim).to(device)
    return model


def train_ddp(args):
    # Get rank, world size, and local rank from the environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize process group backend depending on the device
    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend)

    # Determine device type for logging
    device_type = "GPU" if args.device == "cuda" else "CPU"

    # Configure logger to display device information
    logger.remove()
    logger.add(
        sys.stderr,
        format=f"<green>{{time}}</green> | <level>{{level}}</level> | Device: {device_type} {rank} | <cyan>{{message}}</cyan>",
        level="INFO",
    )

    logger.info(
        f"Starting training on device: {device_type}. World size: {world_size}, local rank: {local_rank}"
    )

    # Set the device based on the argument
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Using GPU: cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU")

    # Prepare data
    sequences_fp = "item_sequence.jsonl"
    val_sequences_fp = "val_item_sequence.jsonl"
    idm = IDMapper().load("../data/idm.json")

    dataset = SkipGramDistributedDataset(
        sequences_fp,
        window_size=args.window_size,
        negative_samples=args.num_negative_samples,
        id_to_idx=idm.item_to_index,
    )
    val_dataset = SkipGramDistributedDataset(
        val_sequences_fp,
        dataset.interacted,
        dataset.item_freq,
        window_size=args.window_size,
        negative_samples=args.num_negative_samples,
        id_to_idx=idm.item_to_index,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=val_dataset.collate_fn,
    )

    assert dataset.id_to_idx == idm.item_to_index, "ID Mappings are not matched!"

    # Initialize model and wrap with DDP
    n_items = len(dataset.items)
    model = init_model(n_items, args.embedding_dim, device)

    if args.device == "cuda":
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)  # No device_ids for CPU

    # Training loop
    n_epochs = args.epochs

    metric_log_cb = MetricLogCallback()
    callbacks = [metric_log_cb.process_payload]
    if args.log_to_mlflow:
        mlflow_log_cb = MLflowLogCallback()
        callbacks.append(mlflow_log_cb.process_payload)

    train(
        ddp_model,
        dataloader,
        val_dataloader,
        epochs=n_epochs,
        patience=args.early_stopping_patience,
        update_steps=100,
        lr=args.learning_rate,
        l2_reg=args.l2_reg,
        gradient_clipping=False,
        device=device,
        callbacks=callbacks,
    )

    # Save the model only on rank 0
    if rank == 0:
        model_path = f"{args.notebook_persist_dp}/skipgram_model_full.pth"
        logger.info(f"Saving model to {model_path}...")
        torch.save(model.state_dict(), model_path)

        id_mapping_path = f"{args.notebook_persist_dp}/skipgram_id_mapping.json"
        logger.info(f"Saving id_mapping to {id_mapping_path}...")
        dataset.save_id_mappings(id_mapping_path)

    # Clean up
    dist.destroy_process_group()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PyTorch DDP SkipGram Training")
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size per process"
    )
    parser.add_argument("--log-to-mlflow", type=bool, default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for training",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    log_to_mlflow = args.log_to_mlflow
    if rank != 0:
        log_to_mlflow = False

    # Initialize the training arguments
    training_args = Args(
        batch_size=args.batch_size,
        device=args.device,
        epochs=args.epochs,
        log_to_mlflow=log_to_mlflow,
    ).init()

    print(training_args.model_dump_json(indent=2))

    train_ddp(args=training_args)


if __name__ == "__main__":
    main()
