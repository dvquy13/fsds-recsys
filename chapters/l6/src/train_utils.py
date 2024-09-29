import time

import mlflow
import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from tqdm.auto import tqdm

from src.id_mapper import IDMapper


class MLflowLogCallback:
    def __init__(self):
        pass

    def process_payload(self, payload: dict):
        step = payload.get("step", None)
        dataset = payload.get("dataset", None)

        if dataset == "train":
            mlflow.log_metric("train_global_loss", payload["global_loss"], step=step)
            if "total_grad_norm" in payload:
                mlflow.log_metric(
                    "total_grad_norm", payload["total_grad_norm"], step=step
                )
            for key, value in payload.items():
                if key.startswith("grad_norm_"):
                    mlflow.log_metric(key, value, step=step)

        # Log validation loss at epoch level
        if "val_loss" in payload:
            mlflow.log_metric(
                "val_loss", payload["val_loss"], step=payload.get("epoch", 0)
            )

        # Log epoch-level metrics for training loss
        if "train_loss" in payload:
            mlflow.log_metric(
                "train_loss", payload["train_loss"], step=payload.get("epoch", 0)
            )

        if "total_train_time_seconds" in payload:
            mlflow.log_metrics(payload)

        if "learning_rate" in payload:
            mlflow.log_metric(
                "learning_rate", payload["learning_rate"], step=payload["epoch"]
            )


class MetricLogCallback:
    def __init__(self):
        self.payloads = []

    def process_payload(self, payload: dict):
        self.payloads.append(payload)


def log_gradients(model):
    total_norm = 0
    param_count = 0
    gradient_metrics = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            param_count += 1
            gradient_metrics[f"grad_norm_{name}"] = param_norm

    total_norm = total_norm**0.5
    gradient_metrics["total_grad_norm"] = total_norm
    return gradient_metrics


def map_indice(df, idm: IDMapper, user_col="user_id", item_col="parent_asin"):
    return df.assign(
        **{
            "user_indice": lambda df: df[user_col].apply(
                lambda user_id: idm.get_user_index(user_id)
            ),
            "item_indice": lambda df: df[item_col].apply(
                lambda item_id: idm.get_item_index(item_id)
            ),
        }
    )


def train(
    model,
    dataloader,
    val_dataloader=None,
    loss_fn=None,  # The dataloader.dataset should implement default loss_fn
    early_stopping=True,
    patience=5,
    epochs=10,
    lr=0.001,
    delta_perc=0.01,
    update_steps=100,
    device="cpu",
    l2_reg=1e-5,
    gradient_clipping=False,
    callbacks=[],
    verbose=False,
):
    if (dataset_type := model.get_expected_dataset_type()) != (
        actual := type(dataloader.dataset)
    ):
        raise Exception(f"Expected dataset type {dataset_type} but got {actual}")
    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    # scheduler_patience = (
    #     patience // 2
    # )  # If equal or greater than early stopping patience then no use
    scheduler_patience = (
        1  # This is like after 1 + 1 rounds of intolerable -> reduce the learning rate
    )
    scheduler_factor = 0.3  # Andrew Ng magic learning rate step number
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience
    )
    best_val_loss = np.inf
    no_improvement_count = 0
    step = 0

    stop_training = False
    total_train_time = 0  # To track the total time

    epoch_iterator = tqdm(
        range(epochs),
        desc=f"Epochs",
        position=0,
        total=epochs,
    )

    for epoch in epoch_iterator:
        epoch_start_time = time.time()  # Start time for this epoch

        model.train()
        total_loss = 0
        num_batches = len(dataloader)
        batch_iterator = tqdm(
            enumerate(dataloader),
            desc=f"Training Epoch {epoch+1}",
            position=1,
            leave=None,
            total=num_batches,
        )

        for batch_idx, batch_input in batch_iterator:
            step += 1

            optimizer.zero_grad()
            loss = dataset_type.forward(
                model, batch_input, loss_fn=loss_fn, device=device
            )

            loss.backward()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if step % update_steps == 0:
                # Collect metric values
                metric_log_payload = {"step": step, "dataset": "train"}
                global_loss = total_loss / (batch_idx + 1)
                metric_log_payload["global_loss"] = global_loss
                gradient_metrics = log_gradients(model)
                metric_log_payload.update(gradient_metrics)

                # Log metrics
                if verbose:
                    logger.info(
                        f"Step {step}, Global Loss: {metric_log_payload['global_loss']:.4f}"
                    )
                    logger.info(
                        f"Step {step}, Learning Rate: {metric_log_payload['learning_rate']:.6f}"
                    )
                    logger.info(f"Step {step}, Gradient Norms: {gradient_metrics}")

                for callback in callbacks:
                    callback(metric_log_payload)

            batch_iterator.set_postfix(Global_Loss=total_loss / (batch_idx + 1))

        epoch_iterator.set_postfix(Global_Loss=total_loss / (batch_idx + 1))

        avg_train_loss = total_loss / num_batches
        epoch_metric_log_payload = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
        }
        epoch_metric_log_payload["learning_rate"] = scheduler.get_last_lr()[0]
        if verbose:
            logger.info(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")

        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_input in val_dataloader:
                    loss = dataset_type.forward(model, batch_input, device=device)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            epoch_metric_log_payload["val_loss"] = val_loss
            epoch_iterator.set_postfix(Val_Loss=val_loss)
            if verbose:
                logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if early_stopping:
                if val_loss < best_val_loss * (1 - delta_perc):
                    best_val_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    stop_training = True
        else:
            scheduler.step(avg_train_loss, epoch=epoch + 1)

        # Log the time taken for this epoch
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        if verbose:
            logger.info(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds")

        for callback in callbacks:
            callback(epoch_metric_log_payload)

        if stop_training:
            break

    # Log the total training time
    avg_train_time_per_epoch = total_train_time / (epoch + 1)
    logger.info(f"Total training time: {total_train_time:.2f} seconds")
    logger.info(
        f"Average training time per epoch: {avg_train_time_per_epoch:.2f} seconds"
    )
    for callback in callbacks:
        callback(
            {
                "num_train_epochs": epoch + 1,
                "total_train_time_seconds": total_train_time,
                "avg_train_time_seconds_per_epoch": avg_train_time_per_epoch,
            }
        )
