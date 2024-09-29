import os
import sys

import mlflow
import torch
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

load_dotenv()

sys.path.insert(0, "..")

from src.id_mapper import IDMapper
from src.skipgram.dataset import SkipGramDataset
from src.skipgram.model import SkipGram
from src.train_utils import MetricLogCallback, MLflowLogCallback, train


class Args(BaseModel):
    log_to_mlflow: bool = True
    experiment_name: str = "FSDS RecSys - L6 - Scale training"
    run_name: str = "001-test-run-large-data"
    notebook_persist_dp: str = None
    random_seed: int = 41
    top_K: int = 100
    top_k: int = 10
    batch_size: int = 128
    num_epochs: int = 1
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


args = Args().init()


def init_model(n_items, embedding_dim, device):
    model = SkipGram(n_items, embedding_dim).to(device)
    return model


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using {device} device")

# Prepare data
sequences_fp = "item_sequence.jsonl"
val_sequences_fp = "val_item_sequence.jsonl"
idm = IDMapper().load("../data/idm.json")

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

# Train
n_items = len(dataset.items)
n_epochs = args.num_epochs

metric_log_cb = MetricLogCallback()
mlflow_log_cb = MLflowLogCallback()

model = init_model(n_items, args.embedding_dim, device)

train(
    model,
    dataloader,
    val_dataloader,
    epochs=n_epochs,
    patience=args.early_stopping_patience,
    update_steps=100,
    lr=args.learning_rate,
    l2_reg=args.l2_reg,
    gradient_clipping=False,
    device=device,
    callbacks=[metric_log_cb.process_payload, mlflow_log_cb.process_payload],
)

# Clean up
all_params = [args]

if args.log_to_mlflow:
    for params in all_params:
        params_dict = params.model_dump()
        params_ = {f"{params.__repr_name__()}.{k}": v for k, v in params_dict.items()}
        mlflow.log_params(params_)

    mlflow.end_run()

# Persist model
model_path = f"{args.notebook_persist_dp}/skipgram_model_full.pth"
logger.info(f"Saving model to {model_path}...")
torch.save(model, model_path)

id_mapping_path = f"{args.notebook_persist_dp}/skipgram_id_mapping.json"
logger.info(f"Saving id_mapping to {id_mapping_path}...")
dataset.save_id_mappings(id_mapping_path)
